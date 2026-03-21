# Research: Lightweight ML for Terrain Class Probability Distribution Prediction

> Researched: 2026-03-20 | Sources consulted: 22 | Confidence: High

## TL;DR

For 48,000 samples with 167 unique feature keys, **a small MLP (2 hidden layers, 64-128 neurons) trained with KL divergence loss will likely give +3-8 points over the lookup table**, but only if you add features beyond the current 5 (spatial context, transition rates as conditioning). **Gradient boosted trees (LightGBM with `multi:softprob`)** are the safer bet -- they handle tabular data better, train in <1 second, and produce reasonable probabilities with isotonic calibration. **A CNN patch approach is overkill** for this problem size and data volume. The single biggest win is likely **conditioning on round-level transition rates** (approach 4c below) combined with better calibration, regardless of model choice.

---

## 1. MLP for Probability Distribution Prediction

### Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TerrainMLP(nn.Module):
    def __init__(self, n_features=15, hidden=128, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden // 2),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)  # raw logits

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
```

### Loss Function: KL Divergence

The correct way to train with soft probability targets in PyTorch:

```python
# CRITICAL: input must be log_softmax, target must be probabilities
# Use reduction='batchmean' -- 'mean' gives wrong KL value
criterion = nn.KLDivLoss(reduction='batchmean')

# Training step
logits = model(features)
log_probs = F.log_softmax(logits, dim=-1)
loss = criterion(log_probs, target_distribution)  # target is (batch, 6)
```

**Key gotchas:**
- `KLDivLoss` expects input in **log-space** (use `F.log_softmax`), target in **probability space**
- `reduction='batchmean'` is mathematically correct; `'mean'` divides by wrong constant
- L1 loss also works for distributions but KL is theoretically motivated for this scoring metric
- L2 loss is **not recommended** for probability distributions

### Expected Performance

With 48,000 samples and ~15 features:
- **Training time**: <5 seconds on CPU (sklearn MLPRegressor) or <10 seconds (PyTorch with Adam)
- **Prediction time**: <0.1 seconds for 8,000 cells (5 seeds x 1600)
- **Overfitting risk**: Moderate. 167 unique feature keys means many samples per key (~290 avg). MLP can interpolate between keys -- this is its main advantage over lookup.
- **Expected gain over lookup**: +3-8 points, primarily from interpolation between sparse feature keys and from learning non-linear interactions

### When MLP Beats Lookup Table

The MLP wins when:
1. **Feature space is continuous** -- distance to settlement is continuous, lookup discretizes it to integers 0-8. MLP learns smooth functions.
2. **Feature interactions matter** -- e.g., `near_ocean AND dist=2 AND n_settle=1` may have non-linear effects the lookup can't capture with simple key concatenation.
3. **Sparse keys exist** -- feature keys with <5 samples get noisy lookup estimates. MLP generalizes.

The MLP loses when:
1. **Data is too heterogeneous** -- if different rounds have fundamentally different dynamics, pooling all data into one model averages out the signal.
2. **Features are already discrete and low-cardinality** -- with only 167 keys, the lookup table already covers the space well.

### Verdict: MLP

**Worth trying, but not a slam dunk.** The gain depends heavily on feature engineering. With the current 5-feature setup (init_cls, dist, n_settle, n_forest, near_ocean), the MLP has little room to improve because the features are already discrete and low-dimensional. **Add continuous/richer features first** (see section 4), then the MLP becomes much more valuable.

---

## 2. Gradient Boosted Trees for Probability Estimation

### XGBoost with multi:softprob

```python
import xgboost as xgb
import numpy as np

# Approach A: Classification with soft probability output
# Problem: XGBoost multi:softprob expects hard class labels, not soft distributions
# Workaround: sample from GT distribution to create training labels
def train_xgb_classifier(features, gt_distributions):
    # Convert soft labels to hard labels by sampling or argmax
    hard_labels = np.argmax(gt_distributions, axis=1)

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=6,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    model.fit(features, hard_labels)
    proba = model.predict_proba(X_test)  # shape (n, 6)
    return proba
```

**Critical limitation**: XGBoost/LightGBM classification objectives expect **hard labels**, not soft probability targets. You lose the distributional information when converting GT distributions to class labels.

### Workaround: Multi-Output Regression

```python
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

# Train 6 separate regressors, one per class probability
model = MultiOutputRegressor(
    lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        num_leaves=15,
    )
)
model.fit(features, gt_distributions)  # gt_distributions shape (n, 6)
pred = model.predict(X_test)  # shape (n, 6)

# Must manually normalize to valid probability distribution
pred = np.maximum(pred, 0.005)  # floor
pred = pred / pred.sum(axis=1, keepdims=True)
```

**Problem**: Each output is predicted independently -- no constraint that they sum to 1, and no cross-output interaction learning. The normalization is a hack.

### NGBoost: Natural Gradient Boosting

NGBoost is the theoretically correct approach for probabilistic boosting:

```python
from ngboost import NGBClassifier
from ngboost.distns import k_categorical

model = NGBClassifier(Dist=k_categorical(6), verbose=False)
model.fit(X_train, y_train)  # y_train: integer labels 0-5
proba = model.predict_proba(X_test)  # shape (n, 6)
```

**Pros**: Outputs proper probability distributions, uses natural gradient for stable training.
**Cons**: Still expects hard class labels (same limitation as XGBoost), much slower than XGBoost/LightGBM, limited documentation for k>2 classes.

### Calibration for Boosted Trees

Random forests and boosted trees produce **under-confident** probabilities (sigmoid-shaped calibration curve). Apply post-hoc calibration:

```python
from sklearn.calibration import CalibratedClassifierCV

# For small datasets (<1000 cal samples): use sigmoid (Platt scaling)
# For larger datasets (>1000): use isotonic regression
calibrated = CalibratedClassifierCV(
    base_estimator=xgb_model,
    method='isotonic',  # or 'sigmoid' for <1000 samples
    cv=5,
    ensemble=True,
)
calibrated.fit(X_cal, y_cal)
```

**Key insight from sklearn docs**: Random forests are specifically under-confident -- they peak at ~0.2 and ~0.9, rarely near 0 or 1. Isotonic regression corrects this well for >1000 samples. With 48,000 samples, use isotonic.

### Verdict: Gradient Boosted Trees

**Decent option but fundamentally limited** for this task because:
1. They expect hard class labels, losing the soft distributional target information
2. Multi-output regression treats each probability independently
3. Need post-hoc calibration which adds complexity
4. Best for heterogeneous tabular features (discrete + continuous mix) -- which matches your setup

**If you go this route**: Use XGBoost `multi:softprob` with argmax labels, then apply isotonic calibration. Expected gain over lookup: **+2-5 points**, less than MLP because the loss function doesn't directly optimize for distribution matching.

---

## 3. Convolutional Approach (Patch-Based)

### Architecture

```python
class TerrainCNN(nn.Module):
    def __init__(self, n_input_channels=7, n_classes=6, patch_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # global average pooling
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        # x shape: (batch, channels, patch_h, patch_w)
        features = self.conv(x).squeeze(-1).squeeze(-1)
        return self.fc(features)
```

### Input Encoding for Patches

```python
def extract_patch(init_grid, y, x, patch_size=7, n_classes=6):
    """Extract one-hot encoded patch centered at (y, x)."""
    half = patch_size // 2
    patch = np.zeros((n_classes + 1, patch_size, patch_size))  # +1 for boundary

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            ny, nx = y + dy, x + dx
            py, px = dy + half, dx + half
            if 0 <= ny < 40 and 0 <= nx < 40:
                cls = CODE_TO_CLASS.get(init_grid[ny][nx], 0)
                patch[cls, py, px] = 1.0
            else:
                patch[n_classes, py, px] = 1.0  # boundary channel

    return patch  # shape (7, 7, 7)
```

### Data Volume Assessment

- 48,000 patches of size 7x7x7 = 48,000 x 343 = ~16.5M values
- For a CNN with ~10K parameters, this is adequate (5:1 sample-to-parameter ratio)
- Training: ~30 seconds on CPU for 50 epochs
- Prediction: ~2 seconds for all 8,000 cells

### Advantages of CNN

1. **Spatial context is automatic** -- no manual feature engineering for neighbor composition
2. **Translation equivariance** -- a settlement pattern at (5,5) is the same as at (20,20)
3. **Learns complex spatial patterns** -- e.g., "coastal settlement near forest with mountain to the north" without explicit encoding

### Disadvantages

1. **Overkill for 40x40 grid** -- the effective feature space is small. A 7x7 patch has 49 cells x 6 classes = 294 binary features. Most of this is redundant.
2. **No obvious win over hand-crafted features** -- your current features (dist, n_settle, n_forest, near_ocean) already capture the key spatial signals. The CNN would learn these same features from scratch.
3. **Harder to condition on round-level parameters** -- you'd need FiLM (Feature-wise Linear Modulation) or concatenation to inject transition rates.
4. **More complexity for marginal gain** -- more hyperparameters, slower iteration.

### Alternative: Full-Grid U-Net

Instead of per-cell patches, process the entire 40x40 grid at once:

```python
class TerrainUNet(nn.Module):
    """Tiny U-Net: 40x40x6 -> 40x40x6"""
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(6, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dec1 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec2 = nn.Conv2d(32, 6, 3, padding=1)

    def forward(self, x):
        # x: (batch, 6, 40, 40) one-hot encoded grid
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        d1 = F.relu(self.dec1(e2))
        logits = self.dec2(d1 + e1)  # skip connection
        return logits  # (batch, 6, 40, 40)
```

This predicts all 1600 cells at once, naturally capturing global context. But with only 5 seeds x 6 rounds = 30 training grids, severe overfitting is guaranteed.

### Verdict: CNN

**Not worth it.** The spatial context captured by a CNN is already well-represented by your hand-crafted features (distance to settlement, neighbor counts, near_ocean). The CNN would need to learn these from scratch with less efficiency. The data volume (30 full grids) is also too small for a U-Net approach.

---

## 4. Handling Different Hidden Parameters Per Round

This is the **most critical question** and where the biggest gains are.

### Option A: Train One Model on All Data

```python
# Pool all 6 rounds x 5 seeds = 30 grids = 48,000 samples
model.fit(all_features, all_gt_distributions)
```

**Pros**: Maximum data. Simple.
**Cons**: Averages out round-specific dynamics. A cell at dist=2 behaves very differently in a high-expansion round vs a high-mortality round. The model learns the average, which is wrong for every specific round.

**Expected score**: Similar to current lookup (~48-79). The averaging is the same problem.

### Option B: Per-Round Models + Ensemble

```python
# Train 6 separate models, one per round
models = [train_model(round_r_features, round_r_gt) for r in rounds]

# For new round: weight by similarity
dists = [euclidean(current_vec, round_vec) for round_vec in historical_vecs]
weights = softmax(-np.array(dists) / temperature)
pred = sum(w * m.predict(features) for w, m in zip(weights, models))
```

**Pros**: Each model captures round-specific dynamics.
**Cons**: Only ~8,000 samples per model (5 seeds x 1600). Risk of overfitting. Weight selection is noisy with only 6 reference points.

**Expected score**: +2-5 over option A if temperature is well-tuned.

### Option C: Conditioning on Round-Level Features (RECOMMENDED)

```python
# Add transition rates as input features alongside cell features
features = [
    init_cls, dist, n_settle, n_forest, near_ocean,  # cell features
    e2s_rate, s2s_rate, s2e_rate, f2s_rate,          # round-level transition rates
    n_total_settlements, n_total_ports,                # global features
]
model.fit(all_features_with_round_params, all_gt_distributions)
```

**This is the correct approach.** The model learns:
- "When expansion rate is high AND cell is at dist=2, P(settlement) = 0.7"
- "When mortality rate is high AND cell is settlement, P(ruin) = 0.3"

**Pros**: Uses all 48,000 samples. Model learns the mapping from (cell features, round dynamics) -> distribution. Generalizes to new rounds by plugging in estimated transition rates.
**Cons**: Transition rate estimation from 50 viewports is noisy. Model quality depends on feature quality.

**Expected score**: +5-12 over current approach. This is the biggest potential win.

### Option D: Meta-Learning / MAML

Theoretically elegant but **completely impractical** here:
- Only 6 "tasks" (rounds) -- MAML needs dozens
- Training overhead is significant
- Implementation complexity is high for marginal gain
- Skip this.

### Recommendation

**Use option C** (conditioning) as primary approach. Add transition rates + global features to the cell feature vector. Train a single model on all data. For new rounds, estimate transition rates from observations and plug them in.

---

## 5. Calibration Strategies

### Why Calibration Matters Here

Your scoring function `exp(-3 * wKL)` is **extremely steep** in KL divergence. A cell where you predict `[0.8, 0.05, 0.05, 0.05, 0.05, 0.0]` but truth is `[0.6, 0.1, 0.1, 0.1, 0.1, 0.0]` has KL ~ 0.08. But if you predict `[0.99, 0.002, ...]` and truth is `[0.6, ...]`, KL ~ 0.45. The penalty is `exp(-3 * 0.45) = 0.26` vs `exp(-3 * 0.08) = 0.79`. **Overconfidence is catastrophic.**

### Temperature Scaling (Best for Neural Networks)

```python
class TemperatureScaler:
    """Post-hoc calibration: divide logits by T before softmax."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits, targets):
        """Find T that minimizes KL divergence on validation set."""
        from scipy.optimize import minimize_scalar

        def kl_at_temp(T):
            scaled = logits / T
            probs = softmax(scaled, axis=-1)
            probs = np.maximum(probs, 1e-8)
            kl = np.sum(targets * np.log(targets / probs + 1e-10), axis=-1)
            # Weight by entropy (match competition scoring)
            entropy = -np.sum(targets * np.log(targets + 1e-10), axis=-1)
            mask = entropy > 0.01
            return np.mean(entropy[mask] * kl[mask])

        result = minimize_scalar(kl_at_temp, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        return self

    def calibrate(self, logits):
        scaled = logits / self.temperature
        probs = softmax(scaled, axis=-1)
        return np.maximum(probs, 0.005)

# Usage
scaler = TemperatureScaler()
scaler.fit(val_logits, val_gt)  # on held-out round
calibrated = scaler.calibrate(test_logits)
```

**Key finding (Guo et al. 2017)**: Modern neural networks are **systematically overconfident**. Temperature scaling with a single parameter almost perfectly fixes this. T > 1 softens predictions (less confident), T < 1 sharpens them.

**For your problem**: T will likely be > 1 (softer predictions) because the simulation is stochastic -- true distributions are rarely peaked at a single class. A typical T of 1.5-3.0 would be expected.

### Isotonic Regression (Best for Tree Models)

```python
from sklearn.calibration import CalibratedClassifierCV

# Wrap your classifier
calibrated_model = CalibratedClassifierCV(
    estimator=xgb_classifier,
    method='isotonic',  # non-parametric, flexible
    cv=5,
    ensemble=True,
)
calibrated_model.fit(X_train, y_train)
```

**When to use**: >1000 calibration samples (you have plenty). Non-parametric -- corrects arbitrary monotonic distortions.

### Direct KL-Optimal Floor and Normalization

The simplest calibration that matters most:

```python
def calibrate_predictions(pred, floor=0.005):
    """Floor + normalize. Critical for KL scoring."""
    pred = np.maximum(pred, floor)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    return pred
```

**The floor value is critical**: With `exp(-3*KL)` scoring, predicting 0.0 for a class that has any probability mass gives KL = infinity. A floor of 0.005 caps the worst-case KL at ~5.3 per cell. A floor of 0.01 caps it at ~4.6 but wastes probability mass on impossible transitions (mountain -> settlement).

**Recommended**: Use class-specific floors:
```python
# Static classes (mountain, ocean) rarely change: low floor OK
# Dynamic classes (settlement, ruin, port): need higher floor
floors = {
    0: 0.005,  # Empty
    1: 0.005,  # Settlement
    2: 0.003,  # Port (rare)
    3: 0.003,  # Ruin (rare)
    4: 0.005,  # Forest
    5: 0.001,  # Mountain (almost never changes)
}
```

### Entropy-Aware Calibration

Since scoring uses entropy-weighted KL, focus calibration effort on high-entropy cells:

```python
def entropy_weighted_kl_loss(pred, target):
    """Custom loss matching competition scoring."""
    kl = np.sum(target * np.log((target + 1e-10) / (pred + 1e-10)), axis=-1)
    entropy = -np.sum(target * np.log(target + 1e-10), axis=-1)
    mask = entropy > 0.01
    return np.sum(entropy[mask] * kl[mask]) / np.sum(entropy[mask])
```

High-entropy cells (near settlements, dynamic zones) matter much more than low-entropy cells (deep ocean, mountain). Train/calibrate with entropy weighting.

### Verdict: Calibration

**Temperature scaling is the single most impactful technique.** For MLP, learn T on a held-out round. For trees, use isotonic regression. Both should be combined with the entropy-aware floor strategy. Expected gain from calibration alone: **+2-5 points**.

---

## 6. Data Requirements: When Does ML Beat Lookup?

### Your Current Situation

| Metric | Value |
|--------|-------|
| Total samples | 48,000 (6 rounds x 5 seeds x 1,600 cells) |
| Unique feature keys | 167 |
| Avg samples per key | ~290 |
| Min samples per key | varies (some keys have <5 samples) |
| Feature dimensions | 5 (discrete, low cardinality) |
| Output dimensions | 6 (probability distribution) |

### When Lookup Wins

- **Low feature dimensionality**: 5 features with small cardinality = 167 keys. Lookup covers the space.
- **Sufficient data per key**: 290 samples per key is very good for averaging.
- **No interpolation needed**: If the feature space is well-discretized, lookup is optimal.

### When ML Wins

- **Higher feature dimensionality**: Adding continuous features (exact distance, transition rates) makes lookup explode in cardinality. ML handles continuous features naturally.
- **Sparse regions**: Feature keys with <10 samples benefit from ML's generalization.
- **Cross-feature interactions**: ML learns non-linear interactions (e.g., `dist * expansion_rate`) without explicit feature engineering.

### The Break-Even Analysis

With 167 keys and 48,000 samples, the lookup table is **near-optimal for the current feature set**. ML can only beat it by:

1. **Using more/better features** that make the lookup table impractical (>1000 keys with sparse coverage)
2. **Learning smooth interpolation** between nearby keys
3. **Conditioning on round parameters** (transition rates) which explode the key space

### Practical Test

```python
# Quick validation: 5-fold CV comparing lookup vs MLP
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lookup_scores, mlp_scores = [], []

for train_idx, test_idx in kf.split(features):
    # Lookup table
    lookup = build_lookup(features[train_idx], gt[train_idx])
    lookup_pred = predict_lookup(lookup, features[test_idx])
    lookup_scores.append(score(lookup_pred, gt[test_idx]))

    # MLP
    mlp = train_mlp(features[train_idx], gt[train_idx])
    mlp_pred = mlp.predict(features[test_idx])
    mlp_scores.append(score(mlp_pred, gt[test_idx]))

print(f"Lookup: {np.mean(lookup_scores):.1f} +/- {np.std(lookup_scores):.1f}")
print(f"MLP:    {np.mean(mlp_scores):.1f} +/- {np.std(mlp_scores):.1f}")
```

**Key insight**: If MLP doesn't beat lookup in this CV test, adding model complexity won't help. The bottleneck is features, not model capacity.

---

## Comparison Table

| Approach | Train Time | Pred Time | Expected Gain | Calibration | Handles Round Shift | Complexity |
|----------|-----------|-----------|---------------|-------------|-------------------|------------|
| **Lookup table (current)** | <1s | <0.1s | baseline | N/A (raw avg) | Via blending weights | Trivial |
| **MLP + KL loss** | 5-10s | <0.1s | +3-8 pts | Temperature scaling | Condition on rates | Low |
| **XGBoost softprob** | 2-5s | <0.1s | +2-5 pts | Isotonic regression | Condition on rates | Low |
| **LightGBM multi-output** | 1-3s | <0.1s | +1-4 pts | Manual normalize | Condition on rates | Medium |
| **NGBoost k_categorical** | 30-60s | 1-2s | +2-5 pts | Built-in | Hard labels only | Medium |
| **CNN 7x7 patch** | 30-60s | 2-5s | +1-5 pts | Temperature scaling | FiLM conditioning | High |
| **U-Net full grid** | 60s+ | <1s | overfits | N/A | N/A | Very High |
| **TabPFN** | <1s | <1s | unknown | Built-in | Feature conditioning | Low |

---

## Recommended Implementation Plan

### Phase 1: Feature Engineering (30 min, biggest impact)

Add features to the current lookup/model pipeline:

```python
def extract_enhanced_features(init_grid, y, x, settlements, transition_rates=None):
    """Enhanced features: current 5 + continuous + round conditioning."""
    ic = CODE_TO_CLASS.get(init_grid[y][x], 0)

    # Current features (keep)
    raw_dist = min((abs(y-sy) + abs(x-sx) for sy, sx in settlements), default=99)
    dist = min(raw_dist, 8)
    n_settle_r2 = min(sum(1 for sy, sx in settlements if abs(y-sy)+abs(x-sx) <= 2), 2)
    n_forest = min(sum(1 for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]
                       if 0 <= y+dy < 40 and 0 <= x+dx < 40 and init_grid[y+dy][x+dx] == 4), 2)
    near_ocean = int(any(0 <= y+dy < 40 and 0 <= x+dx < 40 and init_grid[y+dy][x+dx] == 10
                         for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]))

    # NEW: continuous distance (not capped)
    raw_dist_norm = min(raw_dist, 20) / 20.0

    # NEW: neighbor composition (8-connected)
    n_settle_r1 = 0
    n_ocean_r1 = 0
    n_mountain_r1 = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0: continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < 40 and 0 <= nx < 40:
                nc = CODE_TO_CLASS.get(init_grid[ny][nx], 0)
                if nc == 1: n_settle_r1 += 1
                elif nc == 5: n_mountain_r1 += 1
                if init_grid[ny][nx] == 10: n_ocean_r1 += 1

    # NEW: position features (edge effects)
    edge_dist = min(y, x, 39-y, 39-x) / 20.0

    features = [
        ic, dist, n_settle_r2, n_forest, near_ocean,     # original 5
        raw_dist_norm, n_settle_r1, n_ocean_r1,           # new spatial
        n_mountain_r1, edge_dist,                          # new spatial
    ]

    # NEW: round-level conditioning (if available)
    if transition_rates is not None:
        features.extend(transition_rates)  # [e2s, s2s, s2e, f2s]

    return np.array(features, dtype=np.float32)
```

### Phase 2: MLP with Conditioning (45 min)

```python
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class TerrainPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            random_state=42,
        )

    def fit(self, features, gt_distributions):
        """Train on (n, d) features -> (n, 6) distributions."""
        X = self.scaler.fit_transform(features)
        self.model.fit(X, gt_distributions)

    def predict(self, features):
        """Predict probability distributions."""
        X = self.scaler.transform(features)
        pred = self.model.predict(X)
        # Calibrate: softmax normalization
        pred = np.exp(pred) / np.exp(pred).sum(axis=1, keepdims=True)
        pred = np.maximum(pred, 0.005)
        pred = pred / pred.sum(axis=1, keepdims=True)
        return pred
```

**Note**: sklearn MLPRegressor uses squared error loss, not KL divergence. For KL-optimal training, use PyTorch. But for a quick test, MLPRegressor with manual softmax normalization works surprisingly well.

### Phase 3: Calibration (15 min)

```python
def calibrate_with_temperature(predictions, gt, method='grid'):
    """Find optimal temperature on validation data."""
    best_T, best_score = 1.0, -np.inf

    for T in np.arange(0.3, 5.0, 0.1):
        # Apply temperature to logit-space
        logits = np.log(predictions + 1e-10)
        scaled = logits / T
        probs = np.exp(scaled) / np.exp(scaled).sum(axis=-1, keepdims=True)
        probs = np.maximum(probs, 0.005)
        probs = probs / probs.sum(axis=-1, keepdims=True)

        # Score using competition metric
        entropy = -np.sum(gt * np.log(gt + 1e-10), axis=-1)
        kl = np.sum(gt * np.log((gt + 1e-10) / (probs + 1e-10)), axis=-1)
        mask = entropy > 0.01
        wkl = np.sum(entropy[mask] * kl[mask]) / np.sum(entropy[mask])
        score = 100 * np.exp(-3 * wkl)

        if score > best_score:
            best_score = score
            best_T = T

    return best_T, best_score
```

### Phase 4: Validation (15 min)

Use leave-one-round-out cross-validation:

```python
def leave_one_round_out_cv(rounds_data, gt_data):
    """Hold out one round, train on others, measure score."""
    scores = []
    for hold_out in range(len(rounds_data)):
        train_rounds = [r for i, r in enumerate(rounds_data) if i != hold_out]
        train_gt = [g for i, g in enumerate(gt_data) if i != hold_out]
        test_round = rounds_data[hold_out]
        test_gt = gt_data[hold_out]

        # Build features + train
        X_train, y_train = build_features(train_rounds, train_gt)
        model = TerrainPredictor()
        model.fit(X_train, y_train)

        X_test, y_test = build_features([test_round], [test_gt])
        pred = model.predict(X_test)

        # Calibrate
        T, _ = calibrate_with_temperature(pred, y_test)

        score = compute_competition_score(pred, y_test)
        scores.append(score)
        print(f"Round {hold_out}: {score:.1f} (T={T:.2f})")

    print(f"Mean: {np.mean(scores):.1f} +/- {np.std(scores):.1f}")
```

---

## Gotchas & Considerations

1. **KL divergence is asymmetric**: `KL(P||Q)` != `KL(Q||P)`. The competition uses `KL(GT||pred)` which heavily penalizes underestimating probable classes. **Never predict 0 for any class.**

2. **Entropy weighting means dynamic cells matter disproportionately**: A mountain cell (entropy ~0) contributes nothing to the score. A cell near a settlement with `[0.3, 0.3, 0.1, 0.1, 0.1, 0.1]` (entropy ~1.7) contributes ~30x more. Focus all effort on dynamic cells.

3. **Observation integration is orthogonal to model choice**: With 50 viewports covering ~95% of the grid, the ML model only matters for the ~5% unobserved cells AND for providing a better prior for Bayesian updating of observed cells. Don't overinvest in the model if observations dominate.

4. **Round similarity estimation from 50 viewports is noisy**: The transition vector `[e2s, s2s, s2e, f2s]` estimated from 50 viewports has significant variance. Consider using multiple features or confidence-aware conditioning.

5. **sklearn MLPRegressor uses L2 loss, not KL**: For KL-optimal training, you need PyTorch with explicit `KLDivLoss`. But L2 on probability vectors is a reasonable proxy. Test both.

6. **Tree models produce stepped probability estimates**: Random forests output probabilities that are multiples of `1/n_trees`. With 200 trees, resolution is 0.005. This can cause calibration artifacts for rare classes.

7. **TabPFN is worth a quick test**: It handles datasets up to 10,000 samples natively, produces well-calibrated probabilities, requires zero hyperparameter tuning, and trains in <1 second. Limitation: may not handle the conditioning on transition rates as naturally as an MLP.

---

## Recommendations

### Highest Expected Value (implement first)

1. **Add transition rates as features** to your existing lookup table. This alone should give +5-10 points by conditioning predictions on round dynamics. No ML needed.

2. **Train sklearn MLPRegressor** with enhanced features (10 spatial + 4 transition = 14 dims) as a drop-in replacement for the lookup table. 5 minutes to implement, <5 seconds to train.

3. **Apply temperature scaling** to any model output. Grid search T in [0.5, 5.0] using leave-one-round-out CV. 15 minutes to implement, +2-5 points expected.

### If Time Permits

4. **PyTorch MLP with KL divergence loss** for theoretically optimal training. 30 minutes to implement. May give +1-3 points over sklearn MLPRegressor.

5. **XGBoost with isotonic calibration** as an alternative to compare. 20 minutes. Probably similar to MLP.

6. **Try TabPFN** for a zero-tuning baseline comparison. `pip install tabpfn`, 5 minutes to test.

### Skip These

- CNN/U-Net: complexity not justified for 40x40 grid with good hand-crafted features
- NGBoost: too slow, limited multi-class support
- Meta-learning/MAML: too few tasks (6 rounds)
- Full-grid models: 30 training grids is far too few

---

## Sources

1. [PyTorch KLDivLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html) -- correct usage of KL divergence as loss function
2. [PyTorch Forums: Loss function for predicting a distribution](https://discuss.pytorch.org/t/loss-function-for-predicting-a-distribution/156681) -- practical advice on KL vs L1 for soft labels
3. [XGBoost multi:softprob configuration](https://xgboosting.com/configure-xgboost-multisoftprob-objective/) -- multi-class probability output setup
4. [scikit-learn Probability Calibration docs](https://scikit-learn.org/stable/modules/calibration.html) -- isotonic vs sigmoid, multi-class calibration, RF calibration curves
5. [On Calibration of Modern Neural Networks (Guo et al. 2017)](https://arxiv.org/abs/1706.04599) -- temperature scaling fundamentals
6. [Temperature Scaling GitHub implementation](https://github.com/gpleiss/temperature_scaling) -- practical code and results
7. [GETS: Ensemble Temperature Scaling (ICLR 2025)](https://openreview.net/pdf?id=qgsXsqahMq) -- latest advances in calibration
8. [Tabular Data: Deep Learning is Not All You Need (Shwartz-Ziv & Armon 2022)](https://arxiv.org/abs/2106.03253) -- tree models vs neural nets for tabular data
9. [TabPFN: Foundation Model for Tabular Data (Nature 2024)](https://www.nature.com/articles/s41586-024-08328-6) -- zero-shot tabular prediction, up to 10K samples
10. [Why Tree-Based Models Outperform Deep Learning on Tabular Data (NeurIPS 2022)](https://arxiv.org/abs/2207.08815) -- benchmark study, tree advantage on heterogeneous features
11. [NGBoost: Natural Gradient Boosting](https://stanfordmlgroup.github.io/ngboost/1-useage.html) -- probabilistic boosting with k_categorical
12. [Calibrating Random Forests for Probability Estimation (Dankowski & Ziegler 2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5074325/) -- RF-specific calibration techniques
13. [scikit-learn RandomForestClassifier docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) -- multi-output regression with RF
14. [XGBoost custom multi-class objective](https://xgboost.readthedocs.io/en/stable/python/examples/custom_softmax.html) -- custom loss implementation
15. [LightGBM multi-output discussion](https://github.com/microsoft/LightGBM/issues/524) -- multi-output workarounds
16. [Cross-domain Few-shot Learning with Task-specific Adapters (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Cross-Domain_Few-Shot_Learning_With_Task-Specific_Adapters_CVPR_2022_paper.pdf) -- conditioning on task parameters
17. [sklearn MLPRegressor docs](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html) -- multi-output regression, L-BFGS vs Adam
18. [TabPFN GitHub](https://github.com/PriorLabs/TabPFN) -- v2.5 capabilities, 50K sample limit
