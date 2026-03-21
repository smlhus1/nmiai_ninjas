# Research: Cross-Seed Transfer Learning for Stochastic Simulation Prediction

> Researched: 2026-03-20 | Sources consulted: 14 | Confidence: High

## TL;DR

The 5-10 point gap between perfect-data and stochastic-observation cross-seed models is primarily addressable through **Bayesian shrinkage** (Dirichlet-multinomial posterior with Jeffreys prior alpha=0.5) and **sample-count confidence weighting**. These two techniques alone should recover 3-7 points. Feature-key k-NN fallback and spatial transfer add incremental gains but are more complex to implement.

## Key Findings

### 1. Confidence Weighting — Bayesian Shrinkage is the Answer

**The core formula** (Dirichlet-multinomial posterior predictive):

```
P(class = k | observations) = (N_k + alpha) / (N + K * alpha)
```

Where:
- `N_k` = number of times class k was observed for this feature key
- `N` = total observations for this feature key
- `K` = number of possible classes
- `alpha` = smoothing parameter (prior strength)

**Choosing alpha:**
| Alpha | Name | When to use |
|-------|------|-------------|
| 0 | MLE (no smoothing) | Large N (50+), trust data completely |
| 0.5 | Jeffreys prior | **Recommended default** — minimax optimal, invariant under reparameterization |
| 1.0 | Laplace smoothing | Conservative, spreads probability more evenly |

**Why this works for your problem:** With 1-4 observations per cell, the MLE is extremely noisy — a single observation of class A gives P(A)=1.0, which is clearly wrong for a stochastic simulation. The Jeffreys prior (alpha=0.5) adds 0.5 pseudocounts to each class, effectively saying "I've seen half an observation of each class before looking at the data." This is mathematically optimal for the bias-variance tradeoff.

**Practical implementation:**

```python
def bayesian_smoothed_distribution(observations, n_classes, alpha=0.5):
    """
    observations: list of observed class indices for this feature key
    n_classes: total number of possible classes (e.g., 7 terrain types)
    alpha: Jeffreys prior = 0.5
    """
    counts = [0] * n_classes
    for obs in observations:
        counts[obs] += 1
    N = len(observations)
    total = N + n_classes * alpha
    return [(counts[k] + alpha) / total for k in range(n_classes)]
```

### 2. Observation Debiasing — Shrinkage Toward Global Prior

Beyond per-key smoothing, you should **shrink** low-sample keys toward the global distribution using empirical Bayes:

```
P_final(k) = B * P_local(k) + (1 - B) * P_global(k)
```

Where `B` is the shrinkage factor:

```
B = N / (N + kappa)
```

- `N` = observations for this feature key
- `kappa` = "strength of prior" — tune empirically, start with kappa = 5-10
- `P_local` = observed distribution for this feature key (with Jeffreys smoothing)
- `P_global` = distribution across ALL observations regardless of feature key

**Effect by sample size:**

| N (observations) | B (kappa=5) | Weight on local | Weight on global |
|---|---|---|---|
| 1 | 0.17 | 17% | 83% |
| 2 | 0.29 | 29% | 71% |
| 5 | 0.50 | 50% | 50% |
| 10 | 0.67 | 67% | 33% |
| 50 | 0.91 | 91% | 9% |

**Key insight:** A feature key with 2 observations should be 71% global prior and only 29% local data. This is the James-Stein shrinkage principle — individual estimates with small samples are improved by "borrowing strength" from the overall population.

**Choosing kappa empirically:** Run leave-one-out cross-validation on your observed data. For each observed cell, hold it out, predict using the remaining observations with different kappa values, measure accuracy. Pick the kappa that minimizes prediction error.

### 3. Cross-Seed Spatial Transfer

**Can you transfer spatial patterns across seeds?**

Yes, but carefully. Two approaches:

**Approach A: Spatial context features (recommended)**
Instead of just feature keys based on the cell's own properties, include spatial context:
- Distance to nearest observed settlement
- Density of observed land types in a radius
- Position relative to map edges/water bodies

This implicitly transfers spatial patterns because "cell near a settlement cluster" carries information regardless of which seed it's in.

**Approach B: Direct coordinate transfer (risky)**
If seeds share the same underlying terrain generator with different random seeds, absolute coordinates are meaningless. However, **relative spatial structure** may transfer:
- Cluster detection in observed seeds → predict cluster probability in unobserved seeds
- Spatial autocorrelation range estimation → use as a prior on spatial smoothness

**Recommendation:** Stick with Approach A. Add 2-3 spatial features to your feature key rather than trying explicit coordinate transfer. The cross-seed model already handles this implicitly if the feature keys capture enough spatial context.

### 4. Feature Key Similarity — k-NN Fallback

When a feature key has 0 observations, instead of dropping to a simpler key, find the K nearest observed keys:

**Distance metric for mixed feature keys:**
```python
def feature_key_distance(key_a, key_b, feature_weights):
    """
    For categorical features: Hamming distance (0 if match, 1 if not)
    For numerical features: normalized absolute difference
    Weighted combination (Gower distance)
    """
    dist = 0.0
    for i, (a, b) in enumerate(zip(key_a, key_b)):
        if isinstance(a, str):  # categorical
            dist += feature_weights[i] * (0 if a == b else 1)
        else:  # numerical
            dist += feature_weights[i] * abs(a - b) / feature_ranges[i]
    return dist
```

**k-NN prediction for missing keys:**
```python
def knn_predict(query_key, observed_keys, k=5, alpha=0.5):
    """Find k nearest observed feature keys and blend their distributions"""
    distances = [(feature_key_distance(query_key, ok), ok) for ok in observed_keys]
    distances.sort()
    nearest = distances[:k]

    # Inverse-distance weighting
    weights = [1.0 / (d + 1e-6) for d, _ in nearest]
    total_w = sum(weights)

    blended = [0.0] * n_classes
    for w, (_, key) in zip(weights, nearest):
        dist = get_distribution(key)  # already smoothed
        for c in range(n_classes):
            blended[c] += (w / total_w) * dist[c]
    return blended
```

**Practical note:** This is more complex but addresses the coverage gap directly. Start with k=3-5. Weight by inverse distance so closer keys dominate.

**When to use k-NN vs simpler key fallback:**
- If your feature key hierarchy (full → partial → minimal) works well and covers >95% of cases, stick with the hierarchy — it's simpler and faster
- If coverage gaps are >5% of cells, k-NN is worth the complexity
- You can combine both: try hierarchy first, k-NN only for cells where even the simplest key has <3 observations

### 5. Observation Strategy — Which Cells to Observe

**Cells near settlements ARE more informative**, but for a subtle reason:

High-entropy cells (those with uncertain predictions) give the most information gain per observation. In a land-use simulation:
- **Interior forest/water cells** are highly predictable (low entropy) → low value to observe
- **Edge cells near settlements** are highly uncertain (could be settlement, farmland, forest) → high information gain
- **Cells with rare feature keys** are high-value because they fill coverage gaps

**Optimal observation strategy (if you can choose):**

1. **First priority:** Observe cells with feature keys that have 0 observations in other seeds (fills coverage gaps)
2. **Second priority:** Observe cells with high predicted entropy (most uncertain predictions)
3. **Third priority:** Observe cells near settlement boundaries (highest class diversity)
4. **Low priority:** Interior cells of large homogeneous regions (redundant information)

**Quantitative criterion (expected information gain):**
```
Value(cell) = H(prediction) - E[H(prediction | observation)]
```
Where H is entropy. In practice, approximate this as:
```
Value(cell) ≈ H(current_prediction) * (1 / N_key)
```
High uncertainty AND low sample count = highest value cell to observe.

### 6. Ensemble of Cross-Seed + Historical Priors

**Optimal combination formula:**

```
P_final(k) = w_cross * P_cross_seed(k) + w_hist * P_historical(k) + w_spatial * P_spatial(k)
```

**How to set weights — reliability-based:**

```python
def ensemble_weights(n_cross_observations, n_historical_observations, spatial_confidence):
    """
    Weight each source by its estimated reliability.
    More observations = higher weight.
    """
    # Effective sample sizes
    ess_cross = n_cross_observations
    ess_hist = n_historical_observations * decay_factor  # older = less relevant
    ess_spatial = spatial_confidence  # 0-1 based on spatial model R²

    total = ess_cross + ess_hist + ess_spatial + epsilon
    return ess_cross/total, ess_hist/total, ess_spatial/total
```

**Key insight:** Don't use fixed weights. The optimal blend depends on how much data each source has FOR THAT SPECIFIC CELL. A cell with 20 cross-seed observations should weight cross-seed heavily. A cell with 0 cross-seed observations should fall back to historical + spatial.

**Stacking approach (more principled):**
Use cross-validated predictions from each source as features in a simple logistic regression or softmax model:
```python
# For each cell with known ground truth:
features = [P_cross_seed, P_historical, P_spatial]  # each is a K-dim vector
target = true_class

# Train a simple meta-learner on held-out data
# This automatically learns optimal weights per class
```

## Recommended Implementation Priority

| Priority | Technique | Expected gain | Complexity |
|----------|-----------|---------------|------------|
| 1 | Jeffreys smoothing (alpha=0.5) | +1-2 points | Trivial (5 lines) |
| 2 | Shrinkage toward global prior (kappa=5-10) | +2-4 points | Easy (10 lines) |
| 3 | Sample-count weighted ensemble | +1-2 points | Easy (15 lines) |
| 4 | k-NN fallback for missing keys | +0.5-1 point | Medium (30 lines) |
| 5 | Spatial context features in key | +0.5-1 point | Medium (depends on features) |
| 6 | Observation strategy optimization | +1-2 points | Hard (requires query selection logic) |

**Do items 1-3 first. They're simple and address the biggest sources of the gap.**

## Combined Implementation Sketch

```python
class CrossSeedModel:
    def __init__(self, n_classes, alpha=0.5, kappa=7):
        self.n_classes = n_classes
        self.alpha = alpha
        self.kappa = kappa
        self.key_observations = defaultdict(list)  # key -> [class, class, ...]
        self.global_counts = [0] * n_classes
        self.global_total = 0

    def add_observation(self, feature_key, observed_class):
        self.key_observations[feature_key].append(observed_class)
        self.global_counts[observed_class] += 1
        self.global_total += 1

    def predict(self, feature_key):
        obs = self.key_observations.get(feature_key, [])
        N = len(obs)

        if N == 0:
            # Fallback: k-NN or simpler key or global prior
            return self._global_prior()

        # Step 1: Jeffreys-smoothed local distribution
        counts = [0] * self.n_classes
        for o in obs:
            counts[o] += 1
        local = [(counts[k] + self.alpha) / (N + self.n_classes * self.alpha)
                 for k in range(self.n_classes)]

        # Step 2: Shrink toward global prior
        B = N / (N + self.kappa)
        glob = self._global_prior()

        return [B * local[k] + (1 - B) * glob[k] for k in range(self.n_classes)]

    def _global_prior(self):
        N = self.global_total
        return [(self.global_counts[k] + self.alpha) / (N + self.n_classes * self.alpha)
                for k in range(self.n_classes)]
```

## Gotchas & Considerations

- **Over-smoothing risk:** If kappa is too high, you lose all local signal and everything predicts the global average. Cross-validate kappa on held-out observed cells.
- **Feature key granularity tradeoff:** More specific keys = better predictions when you have data, but more coverage gaps. Less specific keys = fewer gaps but noisier predictions. Hierarchical keys (try specific first, fall back to coarser) handle this well.
- **Non-stationarity across seeds:** If the simulation parameters differ significantly between seeds, cross-seed transfer hurts. Monitor prediction accuracy per seed — if one seed consistently deviates, down-weight its contribution to the model for other seeds.
- **Observation order matters:** Early observations should maximize coverage (diverse feature keys). Later observations should reduce uncertainty on high-entropy cells. This is a classic explore-exploit tradeoff.
- **Computational cost:** All techniques above are O(N) or O(N*K) per prediction. k-NN is O(M) per query where M = number of unique observed keys. For grids of ~1000 cells, all of this runs in microseconds.

## Recommendations

1. **Start with Jeffreys + shrinkage** (priority 1-2). This is the theoretically principled approach and will close most of the gap. Alpha=0.5, kappa=5-10.
2. **Add reliability-weighted ensemble** (priority 3) if you have multiple prediction sources (cross-seed + historical + spatial).
3. **k-NN fallback** (priority 4) only if coverage gaps are a significant problem (>5% of cells with 0 matching observations).
4. **Observation strategy** (priority 6) is the highest-ceiling improvement but hardest to implement. If you can choose which 2-3 cells to observe in shallow seeds, pick cells with rare/missing feature keys first, then high-entropy cells.

## Sources

1. [Additive Smoothing — Grokipedia](https://grokipedia.com/page/Additive_smoothing) — Jeffreys prior formula, bias-variance analysis, Dirichlet posterior
2. [Dirichlet-Multinomial Models — Gregory Gundersen](https://gregorygundersen.com/blog/2020/12/24/dirichlet-multinomial/) — Posterior predictive formula, conjugacy, alpha's role
3. [Shrinkage and Empirical Bayes — Kiwi Damien](https://kiwidamien.github.io/shrinkage-and-empirical-bayes-to-improve-inference.html) — Shrinkage factor formula, B = tau²/(tau²+epsilon²), sample size effects
4. [James-Stein Estimator — Wikipedia](https://en.wikipedia.org/wiki/James%E2%80%93Stein_estimator) — Shrinkage toward grand mean, positive-part estimator
5. [Shrinkage and Empirical Bayes — Peter Hoff (Duke)](https://www2.stat.duke.edu/~pdh10/Teaching/732/Notes/shrinkage.pdf) — Hierarchical Bayes framework, empirical Bayes estimation
6. [Empirical Bayes Method — Wikipedia](https://en.wikipedia.org/wiki/Empirical_Bayes_method) — Prior estimation from data, hierarchical model approximation
7. [KNNImputer — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) — k-NN imputation for missing values, distance metrics
8. [Locality-Sensitive Hashing for Categorical+Numerical Data](https://www.researchgate.net/publication/291225452_Locality-Sensitive_Hashing_for_Data_with_Categorical_and_Numerical_Attributes_Using_Dual_Hashing) — Dual hashing for mixed feature types
9. [Parameter Estimation for Cellular Automata](https://arxiv.org/abs/2301.13320) — Stochastic simulation parameter estimation, intractable likelihood
10. [Bayesian Active Learning](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter21/readings/BayesianAL.pdf) — Information gain for observation selection
11. [Unifying Active Learning via Fisher Information](https://arxiv.org/abs/2208.00549) — Information-theoretic observation strategies
12. [Bayesian Model Averaging Tutorial — Hoeting et al.](https://www.stat.colostate.edu/~jah/papers/statsci.pdf) — Ensemble combination with posterior weights
13. [Area of Applicability — CAST](https://hannameyer.github.io/CAST/articles/cast04-AOA-tutorial.html) — Spatial prediction model applicability domains
14. [Posterior Predictive for Dirichlet-Categorical](https://blog.jakuba.net/posterior-predictive-distribution-for-the-dirichlet-categorical-model/) — Bag-of-words posterior predictive derivation
