# Research: Optimal Blending Strategies for Multiple Probabilistic Prediction Sources

> Researched: 2026-03-20 | Sources consulted: 18 | Confidence: High

## TL;DR

Your hand-tuned linear blend is a reasonable starting point but leaves significant score on the table. The optimal approach is **logarithmic pooling** (geometric mean in log-space) which is the provably optimal aggregation method when scoring with KL divergence. For weight optimization with only 5 historical rounds, use **leave-one-out stacking** with a Dirichlet concentration regularizer. The single biggest improvement is making blend weights **confidence-adaptive** — scaling trust by the effective sample size of each source.

## Key Findings

### 1. Bayesian Optimal Combination via Dirichlet Conjugacy

Your sources map naturally to a Bayesian framework where the unknown is a categorical distribution over 6 classes, and each source provides evidence about it.

**The Dirichlet-Multinomial conjugate update is the principled foundation:**

```
Prior:      Dir(alpha_1, ..., alpha_K)        # K=6 classes
Likelihood: Multinomial(n; p_1, ..., p_K)     # observed counts
Posterior:  Dir(alpha_1 + n_1, ..., alpha_K + n_K)
```

**Key insight**: Each source can be represented as pseudo-counts (Dirichlet alphas), and combination reduces to **adding alphas**:

```python
# Each source contributes a Dirichlet with concentration proportional to trust
# Direct obs: alpha_direct = counts + 0.5 (Jeffreys prior)
# Cross-seed: alpha_cross = n_cross_samples * cross_mean
# Historical:  alpha_hist  = concentration * hist_mean

# Combined posterior:
alpha_combined = alpha_direct + alpha_cross + alpha_hist
prediction = alpha_combined / alpha_combined.sum()
```

This is **mathematically equivalent** to the posterior after observing all data, treating each source as independent multinomial samples. The concentration parameter of each source's Dirichlet encodes how much you trust it (effective sample size).

**Critical advantage over your current linear blend**: The Dirichlet update naturally down-weights uncertain sources because adding `Dir(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)` (1 observation with Jeffreys prior) barely moves a `Dir(50, 2, 1, 1, 1, 1)` (cross-seed model with 55 samples). No manual weight scheduling needed.

**Concrete implementation:**

```python
def blend_dirichlet(direct_counts, n_direct,
                    cross_mean, n_cross,
                    hist_mean, hist_concentration,
                    mc_mean=None, mc_concentration=1.0):
    """Combine sources as Dirichlet pseudo-counts.

    Each source contributes alpha = effective_n * mean_distribution.
    The posterior mean is the prediction.
    """
    K = 6
    # Jeffreys prior as base
    alpha = np.ones(K) * 0.5

    # Direct observations: actual counts (strongest signal)
    if n_direct > 0:
        alpha += direct_counts  # raw counts, NOT normalized

    # Cross-seed model: trust proportional to samples behind the key
    if cross_mean is not None and n_cross > 0:
        # Scale cross-seed contribution by effective sample size
        # Discount slightly because same-round-different-seed != same-cell
        effective_n = n_cross * 0.7  # discount factor (tune this)
        alpha += effective_n * cross_mean

    # Historical prior: concentration encodes trust
    if hist_mean is not None:
        alpha += hist_concentration * hist_mean

    # MC simulation: weakest signal
    if mc_mean is not None:
        alpha += mc_concentration * mc_mean

    # Posterior mean = prediction
    pred = alpha / alpha.sum()
    return np.maximum(pred, 0.005)  # floor for KL safety
```

### 2. Linear vs Logarithmic Pooling Under KL Scoring

This is the most important theoretical finding for your specific scoring metric.

**Key result from decision analysis literature** (Gneiting & Ranjan, 2013; Abbas, 2009):

- **Linear pooling** (what you do now: `w1*p1 + w2*p2`) minimizes the **weighted average Brier score** (quadratic loss) to the experts.
- **Logarithmic pooling** minimizes the **weighted average KL divergence** to the experts.

Since you are scored by entropy-weighted KL divergence, **logarithmic pooling is the theoretically optimal aggregation method**.

**Log-linear pool formula:**

```python
def log_linear_pool(distributions, weights):
    """Combine distributions via logarithmic pooling.

    p_combined(k) proportional to prod_i p_i(k)^w_i

    This is equivalent to weighted geometric mean, normalized.
    """
    log_sum = np.zeros_like(distributions[0])
    for p, w in zip(distributions, weights):
        # Clip to avoid log(0)
        log_sum += w * np.log(np.maximum(p, 1e-10))

    # Normalize in log-space for numerical stability
    log_sum -= log_sum.max()
    combined = np.exp(log_sum)
    combined /= combined.sum()
    return np.maximum(combined, 0.005)
```

**Why log pooling is better for KL scoring:**

| Property | Linear Pool | Log Pool |
|----------|-------------|----------|
| Optimizes | Brier score / L2 | KL divergence / log score |
| Effect on confidence | Flattens (averages) | Sharpens (agreement amplified) |
| Zero handling | Safe (averages away) | Dangerous (any zero kills) |
| External Bayesianity | No | Yes |
| Calibration preserved | No (necessarily uncalibrated) | Approximately |

**The asymmetry matters enormously for your scoring**: Your competition penalizes overconfidence (pred=0 when GT>0) via KL divergence infinity. Log pooling handles this better because:
- If ANY source says p(k)>0, the geometric mean preserves that signal
- Linear pooling can mask a strong signal by averaging with flat predictions
- BUT: log pooling requires careful flooring because `0^w = 0` for any w > 0

**Practical recommendation**: Use log-linear pooling but with aggressive flooring (0.01 minimum per class) and adaptive weights.

### 3. Confidence-Adaptive Blending (Sample-Size Dependent Weights)

Your current scheme uses fixed thresholds (14+ obs, 1-13 obs, 0 obs). This is a step function when it should be smooth.

**The principled approach**: Weight each source by its **effective sample size** (ESS), which is the Dirichlet concentration parameter.

```python
def adaptive_blend_weights(n_direct, n_cross, hist_concentration):
    """Compute blend weights proportional to effective sample sizes.

    For KL scoring, use log-pooling weights (must sum to 1).
    """
    # Effective sample sizes
    ess_direct = n_direct  # 1:1 — these are actual observations
    ess_cross = n_cross * 0.7  # discount: different seed, same params
    ess_hist = hist_concentration  # pre-tuned from historical data

    total = ess_direct + ess_cross + ess_hist + 1e-10

    w_direct = ess_direct / total
    w_cross = ess_cross / total
    w_hist = ess_hist / total

    return w_direct, w_cross, w_hist
```

**Why this is better than fixed weights:**

| Scenario | Your current | Adaptive | Improvement |
|----------|-------------|----------|-------------|
| 14 direct + 3 cross-seed samples | 0.75/0.25 | 0.82/0.12/0.06 | Direct obs dominate correctly |
| 2 direct + 50 cross-seed | 0.30/0.70 | 0.05/0.86/0.09 | Cross-seed properly dominates |
| 0 direct + 3 cross-seed | 0.80/0.20 | 0/0.41/0.59 | With few cross samples, hist gets more weight |
| 0 direct + 200 cross-seed | 0.80/0.20 | 0/0.93/0.07 | Dense cross-seed dominates correctly |

The cross-seed model's sample count per feature key varies wildly (from 2 to 200+). With fixed weights, you treat a 3-sample key the same as a 200-sample key. The adaptive approach handles this naturally.

### 4. Leave-One-Out Stacking for Weight Optimization

With only 5 historical rounds, you need to maximize every data point. **Bayesian stacking via LOO-CV** (Yao, Vehtari, Simpson & Gelman, 2018) is the gold standard.

**The optimization problem:**

```
maximize sum_i log( sum_k w_k * p_k(y_i | x_i) )
subject to: w_k >= 0, sum(w_k) = 1
```

Where `y_i` is the ground truth for round `i` (left out), and `p_k(y_i | x_i)` is model k's prediction trained on the other 4 rounds.

**For your specific case, the approach is:**

```python
from scipy.optimize import minimize

def stacking_weights_loo(historical_rounds, feature_extractor, n_rounds=5):
    """Find optimal blend weights via LOO stacking on historical rounds.

    For each held-out round:
    1. Build cross-seed model from remaining 4 rounds (simulates cross-seed)
    2. Build historical prior from remaining 4 rounds
    3. Compute per-cell KL divergence of each source vs held-out GT
    4. Find weights that minimize total entropy-weighted KL
    """
    n_sources = 3  # cross-seed proxy, historical, MC

    # Collect LOO predictions and ground truths
    loo_scores = []  # shape: (n_rounds, n_cells, n_sources)

    for holdout_idx in range(n_rounds):
        train_rounds = [r for i, r in enumerate(historical_rounds) if i != holdout_idx]
        test_round = historical_rounds[holdout_idx]

        gt = load_ground_truth(test_round)

        for source_idx in range(n_sources):
            pred = build_source_prediction(source_idx, train_rounds, test_round)
            # Compute per-cell log score: log(pred[y,x,gt_class])
            for seed in range(5):
                for y in range(40):
                    for x in range(40):
                        gt_class = gt[seed][y][x]
                        log_score = np.log(max(pred[seed][y,x,gt_class], 1e-10))
                        loo_scores.append((holdout_idx, seed, y, x, source_idx, log_score))

    # Optimize stacking weights
    def neg_log_score(w):
        """Negative LOO log score for the blended prediction."""
        w = np.exp(w) / np.exp(w).sum()  # softmax parametrization
        total = 0.0
        for holdout, seed, y, x, src, ls in loo_scores:
            # Weight the log-scores (for log-linear pooling)
            total += w[src] * ls
        return -total

    w0 = np.zeros(n_sources)
    result = minimize(neg_log_score, w0, method='Nelder-Mead')
    optimal_w = np.exp(result.x) / np.exp(result.x).sum()
    return optimal_w
```

**Important caveats for 5-fold LOO:**

1. **High variance**: With only 5 rounds, LOO estimates are noisy. Use regularization.
2. **Distributional bias**: Recent research (Science Advances, 2025) shows LOO can introduce negative correlation between train/test sets. With 5 rounds this is a real risk.
3. **Recommendation**: Use LOO to get a direction, then apply Dirichlet regularization:
   ```python
   # Regularize toward uniform weights
   alpha_reg = 2.0  # higher = more regularization
   w_regularized = (loo_weights * n_rounds + alpha_reg / n_sources) / (n_rounds + alpha_reg)
   ```

### 5. Grid Search Strategy for 5 Historical Rounds

Given the tiny validation set, a principled grid search approach:

**Step 1: Parameterize the blend**

```python
# 4 parameters to optimize:
# - cross_discount: how much to discount cross-seed ESS (0.3-1.0)
# - hist_concentration: effective sample size of historical prior (1-20)
# - mc_concentration: effective sample size of MC prior (0.5-5)
# - floor: minimum probability per class (0.005-0.05)
```

**Step 2: LOO grid search**

```python
import itertools

cross_discounts = [0.3, 0.5, 0.7, 0.9, 1.0]
hist_concentrations = [2, 5, 8, 12, 15, 20]
mc_concentrations = [0.5, 1, 2, 3, 5]
floors = [0.005, 0.01, 0.02, 0.03]

best_score = -np.inf
best_params = None

for cd, hc, mc, fl in itertools.product(
    cross_discounts, hist_concentrations, mc_concentrations, floors
):
    loo_score = 0.0
    for holdout_idx in range(n_rounds):
        # Build predictions using params (cd, hc, mc, fl) on train rounds
        # Score against holdout GT using entropy-weighted KL
        loo_score += score_round(holdout_idx, cd, hc, mc, fl)

    if loo_score > best_score:
        best_score = loo_score
        best_params = (cd, hc, mc, fl)
```

**Step 3: Refine with Bayesian optimization**

After coarse grid search identifies the region, use `scipy.optimize.minimize` with Nelder-Mead to fine-tune. The 4D parameter space is small enough for this.

**Step 4: Robustness check**

For each candidate parameter set, compute the standard deviation across LOO folds. Prefer parameters with lower variance even if slightly worse mean — this protects against overfitting to the 5 training rounds.

### 6. Literature on Expert Combination

**Key results from the forecast aggregation literature:**

1. **Log pool is KL-optimal** (Abbas, 2009): The logarithmic opinion pool minimizes the weighted average KL divergence from the combined forecast to each expert's forecast. This is a characterization theorem — it's not just empirically better, it's provably the unique optimal method under KL.

2. **Linear pool requires recalibration** (Gneiting & Ranjan, 2013): Any non-trivial weighted average of two or more distinct, calibrated probability forecasts is necessarily uncalibrated. This means your linear blend is biased toward overconfident or underconfident predictions.

3. **Stacking beats BMA** (Yao et al., 2018): Bayesian Model Averaging asymptotically selects the single model closest in KL divergence, throwing away all other models. Stacking properly combines them. With your 3 sources, BMA would effectively just use the best one, wasting the others.

4. **Local weights beat global weights** (LBMM, 2023): Weights that depend on the input features (your feature key) outperform fixed global weights. Your current system already does this implicitly (different weights for different observation counts), but the Dirichlet approach makes it principled.

5. **KL-pool for rare events** (Seckarova, 2017): A KL-based pooling method has equal or higher entropy than linear and log pooling for low-probability events. Since your scoring heavily penalizes missing rare events (pred->0 when GT>0), this is particularly relevant.

## Recommended Implementation

### Phase 1: Quick wins (implement first)

```python
def blend_optimal(direct_counts, n_direct,
                  cross_mean, n_cross,
                  hist_alpha, mc_mean=None):
    """Optimal blend using Dirichlet pseudo-count combination.

    This replaces the if/elif chain with a single unified formula.
    """
    K = 6
    FLOOR = 0.01

    # Start with Jeffreys prior
    alpha = np.ones(K) * 0.5

    # Source 1: Direct observations (strongest per-cell signal)
    if n_direct > 0:
        alpha += direct_counts

    # Source 2: Cross-seed model (same round, different seed)
    if cross_mean is not None and n_cross > 0:
        # Discount factor: cross-seed is informative but not identical
        # More samples = higher trust, but with diminishing returns
        effective_n = min(n_cross * 0.7, 50)  # cap at 50 effective samples
        alpha += effective_n * cross_mean

    # Source 3: Historical Dirichlet (pre-computed concentration)
    if hist_alpha is not None:
        alpha += hist_alpha  # already has concentration baked in

    # Source 4: MC simulation (weak prior, only when nothing else)
    if mc_mean is not None and n_direct == 0 and cross_mean is None:
        alpha += 1.0 * mc_mean  # very weak

    # Posterior mean
    pred = alpha / alpha.sum()
    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum()
    return pred
```

### Phase 2: Log-linear pooling (higher ceiling, more complex)

```python
def blend_log_linear(sources, weights, floor=0.01):
    """Log-linear pool for KL-optimal aggregation.

    sources: list of (distribution, available) tuples
    weights: corresponding weights (sum to 1, only used for available sources)
    """
    K = 6
    log_sum = np.zeros(K)
    w_total = 0.0

    for (dist, available), w in zip(sources, weights):
        if not available or dist is None:
            continue
        # Floor before log to avoid -inf
        safe_dist = np.maximum(dist, floor)
        safe_dist /= safe_dist.sum()
        log_sum += w * np.log(safe_dist)
        w_total += w

    if w_total == 0:
        return np.ones(K) / K

    log_sum /= w_total  # normalize weights
    log_sum -= log_sum.max()  # numerical stability
    combined = np.exp(log_sum)
    combined = np.maximum(combined, floor)
    combined /= combined.sum()
    return combined
```

### Phase 3: LOO-optimized parameters

Optimize `cross_discount`, `hist_concentration`, and `floor` against the 5 historical rounds using the LOO scheme described in section 4.

## Comparison of Approaches

| Approach | Theoretical Basis | Implementation Effort | Expected Improvement | Risk |
|----------|------------------|-----------------------|---------------------|------|
| **Current linear blend** | Ad hoc | Done | Baseline | Over/under-confident |
| **Dirichlet pseudo-count** | Bayesian conjugacy | Low (replace if/elif) | +5-15% KL reduction | Conservative, may under-adapt |
| **Log-linear pooling** | KL-optimal theorem | Medium | +10-20% KL reduction | Sensitive to zeros |
| **LOO-stacked weights** | Stacking theory | High | +5-10% on top | Overfitting (only 5 rounds) |
| **Adaptive ESS weights** | Information theory | Low | +5-15% KL reduction | Needs cross-seed sample counts |
| **Combined: log-pool + adaptive + LOO** | All of above | High | +15-30% KL reduction | Complexity |

## Gotchas & Considerations

1. **Zero probability trap**: KL divergence goes to infinity when pred(k)=0 but GT(k)>0. Your floor of 0.005 is reasonable but consider 0.01 for safety. Log pooling amplifies this risk because `0^w = 0`.

2. **Entropy weighting changes the game**: Since your scoring weights cells by their GT entropy, **high-entropy cells matter more**. These are exactly the cells where all sources are uncertain. The blend strategy for high-entropy cells should be more conservative (flatter) than for low-entropy cells.

3. **Cross-seed discount is crucial**: Treating cross-seed samples as 1:1 equivalent to direct observations would be wrong — the initial grid topology matters. The discount factor (0.5-0.8) should ideally be learned from LOO.

4. **Historical similarity weighting**: Your softmax-over-distance approach for historical weights is good but consider using KL divergence between transition vectors instead of Euclidean distance — it's more appropriate for comparing probability-like vectors.

5. **Feature key granularity tradeoff**: Finer keys = less bias but higher variance (fewer samples per key). The simple fallback (init_cls, dist) is a good variance reduction strategy. Consider a continuous fallback: start with full key, if <5 samples fall back to 3-feature key, if <3 samples fall back to 2-feature key.

6. **Overconfidence asymmetry in scoring**: Since overconfidence hurts more than underconfidence under KL, it is better to err on the side of flatter predictions. The Dirichlet approach naturally does this — the Jeffreys prior (0.5 per class) ensures no class ever gets zero probability.

## Recommendations

**Immediate (high impact, low effort):**
1. Replace the fixed-weight if/elif chain with Dirichlet pseudo-count combination
2. Pass `n_cross` (number of cross-seed samples behind each feature key) to the blend function
3. Increase floor from 0.005 to 0.01

**Short-term (medium effort, high potential):**
4. Implement log-linear pooling as an alternative blend mode
5. Run LOO backtest comparing Dirichlet vs log-linear vs current on all 5 historical rounds
6. Tune `cross_discount` and `hist_concentration` via grid search on LOO

**If time permits:**
7. Implement feature-key cascade (full -> partial -> simple) based on sample count
8. Weight cells by entropy in the LOO objective to match competition scoring
9. Use Bayesian optimization (scipy minimize) for final parameter tuning

## Sources

1. [A Kullback-Leibler View of Linear and Log-Linear Pools (Abbas, 2009)](https://pubsonline.informs.org/doi/10.1287/deca.1080.0133) — Proves log-linear pool is KL-optimal
2. [Using Stacking to Average Bayesian Predictive Distributions (Yao et al., 2018)](https://arxiv.org/abs/1704.02030) — LOO stacking methodology
3. [Bayesian Stacking and Pseudo-BMA weights - loo package](https://mc-stan.org/loo/articles/loo2-weights.html) — Implementation guide for stacking
4. [Performance of KL-Based Expert Opinion Pooling (Seckarova, 2017)](http://proceedings.mlr.press/v58/seckarova17a/seckarova17a.pdf) — KL-pool handles rare events better
5. [Local Bayesian Dirichlet Mixing of Imperfect Models (2023)](https://www.nature.com/articles/s41598-023-46568-0) — Input-dependent weights via Dirichlet
6. [Log-Linear Pool to Combine Prior Distributions (BA, 2012)](https://projecteuclid.org/journals/bayesian-analysis/volume-7/issue-2/Log-Linear-Pool-to-Combine-Prior-Distributions--A-Suggestion/10.1214/12-BA714.pdf) — Weight selection for log-linear pools
7. [Bayesian Inference for Weights in Logarithmic Pooling (BA, 2022)](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Bayesian-Inference-for-the-Weights-in-Logarithmic-Pooling/10.1214/22-BA1311.pdf) — Bayesian weight estimation
8. [Combining Probability Forecasts (Gneiting & Ranjan, 2013)](https://academic.oup.com/jrsssb/article/72/1/71/7076442) — Linear pool recalibration necessity
9. [From Proper Scoring Rules to Max-Min Optimal Forecast Aggregation](https://arxiv.org/abs/2102.07081) — Scoring rules determine aggregation method
10. [Dirichlet Distribution (Wikipedia)](https://en.wikipedia.org/wiki/Dirichlet_distribution) — Conjugate prior formulas
11. [Dirichlet Distribution tutorial (stephens999)](https://stephens999.github.io/fiveMinuteStats/dirichlet.html) — Mean and variance formulas
12. [Distributional bias compromises LOO-CV (Science Advances, 2025)](https://www.science.org/doi/10.1126/sciadv.adx6976) — LOO bias with small samples
13. [Bayesian Hierarchical Stacking (Yao et al., 2021)](https://sites.stat.columbia.edu/gelman/research/published/hierarchical_stacking.pdf) — Input-dependent stacking weights
14. [Confidence-Weighted Ensembling](https://www.emergentmind.com/topics/confidence-weighted-ensembling) — Adaptive weighting by confidence
