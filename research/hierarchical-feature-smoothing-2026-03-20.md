# Research: Hierarchical/Cascading Feature Models for Sparse Lookup Tables

> Researched: 2026-03-20 | Sources consulted: 14 | Confidence: High

## TL;DR

Use **Dirichlet-smoothed interpolation** (not hard fallback) with the formula `alpha = n/(n+k)` to blend child and parent distributions. Drop features one at a time in order of least importance: frontier -> density_bin -> ocean_dist -> food_bin -> distance -> init_class. Use k=5 for the historical model and k=2 for the cross-seed model. This is directly analogous to n-gram language model smoothing (Jelinek-Mercer / Dirichlet prior), which is one of the best-studied problems in ML.

## Key Findings

### 1. Your Problem IS N-gram Smoothing

Your 6-tuple lookup table is structurally identical to a 6-gram language model. The NLP community has spent 40+ years optimizing exactly this problem. The key insight from that literature:

- **Interpolation always beats hard backoff** for sparse data (Chen & Goodman 1998, the definitive study)
- **Kneser-Ney smoothing** is the gold standard for n-grams, but it requires continuation counts that don't apply to your feature structure
- **Jelinek-Mercer interpolation** and **Dirichlet prior smoothing** are the most applicable to your case

### 2. Optimal Fallback Hierarchy (Feature Drop Order)

Given your stated insight that init_class and distance account for 90%+ of prediction power, the optimal cascade drops **least informative features first**:

```
Level 0: (init_class, distance, food_bin, ocean_dist, frontier, density_bin)  -- full 6-tuple
Level 1: (init_class, distance, food_bin, ocean_dist, density_bin)            -- drop frontier
Level 2: (init_class, distance, food_bin, ocean_dist)                         -- drop density_bin
Level 3: (init_class, distance, food_bin)                                     -- drop ocean_dist
Level 4: (init_class, distance)                                               -- drop food_bin
Level 5: (init_class,)                                                        -- drop distance
Level 6: uniform prior                                                        -- drop everything
```

**Rationale for this order:**
- `frontier` (binary: explored or not) provides minimal information once other features are known
- `density_bin` is a discretized continuous value with high sparsity contribution
- `ocean_dist` and `food_bin` are moderate-information features
- `distance` and `init_class` are the core features and should be dropped last

**Alternative consideration:** You could also try dropping features that cause the most key fragmentation first (i.e., the feature with the most unique values creates the most sparse keys). If `density_bin` has 5 bins and `frontier` is binary, dropping `density_bin` first reduces keys by 5x vs 2x.

### 3. Smoothing Between Levels (The Core Formula)

Instead of hard fallback (`if n < threshold: use parent`), use **interpolated smoothing**:

```python
def smoothed_distribution(counts_child, counts_parent, k):
    """
    Dirichlet-style interpolation between child and parent distributions.

    counts_child: Counter of class observations at this key level
    counts_parent: Counter of class observations at parent level
    k: smoothing strength (pseudocount weight)

    Returns: smoothed probability distribution over classes
    """
    n = sum(counts_child.values())  # total child observations
    alpha = n / (n + k)             # child weight: 0 when n=0, approaches 1 for large n

    # Child MLE distribution
    p_child = {cls: c / n for cls, c in counts_child.items()} if n > 0 else {}

    # Parent distribution (already smoothed from its parent)
    n_parent = sum(counts_parent.values())
    p_parent = {cls: c / n_parent for cls, c in counts_parent.items()} if n_parent > 0 else uniform

    # Blend
    p_smoothed = {}
    for cls in all_classes:
        child_prob = p_child.get(cls, 0.0)
        parent_prob = p_parent.get(cls, 1.0 / num_classes)  # uniform fallback
        p_smoothed[cls] = alpha * child_prob + (1 - alpha) * parent_prob

    return p_smoothed
```

**The key parameter is `k`** (sometimes called mu in Dirichlet smoothing):
- `k = 1`: very light smoothing, child dominates with just 1 observation
- `k = 5`: moderate — need ~5 observations before child distribution is trusted at 50%
- `k = 10`: conservative — need 10 observations for 50% child weight
- `k = 20`: very conservative — heavy parent influence

**The weight schedule:**

| n (child samples) | k=2 | k=5 | k=10 | k=20 |
|-------------------|-----|-----|------|------|
| 0                 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1                 | 0.33 | 0.17 | 0.09 | 0.05 |
| 2                 | 0.50 | 0.29 | 0.17 | 0.09 |
| 5                 | 0.71 | 0.50 | 0.33 | 0.20 |
| 10                | 0.83 | 0.67 | 0.50 | 0.33 |
| 20                | 0.91 | 0.80 | 0.67 | 0.50 |
| 50                | 0.96 | 0.91 | 0.83 | 0.71 |

### 4. Recursive Multi-Level Smoothing

The real power comes from applying this **recursively through the hierarchy**:

```python
def get_distribution(key_tuple, level_tables, k_values, all_classes):
    """
    Recursively compute smoothed distribution through hierarchy.

    key_tuple: full (init_class, distance, food_bin, ocean_dist, frontier, density_bin)
    level_tables: dict of {level: {key: Counter}} precomputed from training data
    k_values: list of k per level (can differ!)
    all_classes: set of all terrain classes
    """
    num_classes = len(all_classes)
    uniform = {cls: 1.0 / num_classes for cls in all_classes}

    # Build keys for each level (dropping features from the right)
    keys = [
        key_tuple,                    # L0: full 6-tuple
        key_tuple[:5],                # L1: drop frontier (or whichever is last)
        key_tuple[:4],                # L2: drop density_bin
        key_tuple[:3],                # L3: drop ocean_dist
        key_tuple[:2],                # L4: (init_class, distance)
        key_tuple[:1],                # L5: (init_class,)
    ]

    # Start from most general (uniform prior)
    dist = dict(uniform)

    # Walk from general to specific, blending at each level
    for level in range(len(keys) - 1, -1, -1):
        key = keys[level]
        if key in level_tables[level]:
            counts = level_tables[level][key]
            n = sum(counts.values())
            k = k_values[level]
            alpha = n / (n + k)

            child_dist = {cls: counts.get(cls, 0) / n for cls in all_classes}
            dist = {
                cls: alpha * child_dist[cls] + (1 - alpha) * dist[cls]
                for cls in all_classes
            }

    return dist
```

This is essentially **Jelinek-Mercer interpolation** generalized to a feature hierarchy. Every level contributes, weighted by how much data it has. Levels with 0 samples contribute nothing (alpha=0). Levels with lots of data dominate (alpha~1).

### 5. Empirical Bayes: Learning k from Data

Instead of hand-tuning k, you can learn it from held-out data:

```python
def learn_k(train_data, held_out_data, level_tables, all_classes, k_candidates):
    """
    Find optimal k per level using held-out log-likelihood.
    """
    best_k = []
    for level in range(len(level_tables)):
        best_ll = -float('inf')
        best_k_val = 5  # default
        for k in k_candidates:  # e.g., [0.5, 1, 2, 5, 10, 20, 50]
            ll = 0
            for sample in held_out_data:
                key = make_key(sample, level)
                dist = get_smoothed_dist(key, level_tables, k, all_classes)
                true_class = sample['terrain_class']
                ll += math.log(max(dist[true_class], 1e-10))
            if ll > best_ll:
                best_ll = ll
                best_k_val = k
        best_k.append(best_k_val)
    return best_k
```

This is standard empirical Bayes — use the marginal likelihood on held-out data to set hyperparameters. The NLP community calls this "deleted interpolation" (Jelinek & Mercer 1980).

### 6. Minimum Samples for Reliable Multinomial Estimates

For 6 terrain classes, the theoretical and practical guidance:

| Metric | Threshold | Reasoning |
|--------|-----------|-----------|
| **Bare minimum** | 6 samples | At least 1 per class on average (Laplace) |
| **Rule of thumb** | 5 per category = 30 | Goodman CI method requires >= 5 per cell |
| **Reliable MLE** | 10 per category = 60 | MSE of MLE beats smoothed estimate |
| **High confidence** | 20 per category = 120 | Narrow confidence intervals |

**For your case (6 classes, median 37 samples):** The median key is borderline adequate for raw MLE. The 20% of keys with <10 samples definitely need smoothing.

**Practical recommendation:** Don't use a hard threshold. The Dirichlet smoothing formula `alpha = n/(n+k)` handles this automatically:
- n=0: pure parent distribution (alpha=0)
- n=5, k=5: 50/50 blend
- n=37, k=5: 88% child, 12% parent — small correction
- n=100, k=5: 95% child — almost pure MLE

### 7. Cross-Seed Model (2-50 samples, many 0s)

For the cross-seed model with very few samples, use **higher k values** (stronger parent prior):

```python
# Historical model: lots of data, light smoothing
k_historical = [5, 5, 5, 5, 3, 2]  # less smoothing at general levels

# Cross-seed model: sparse data, heavy smoothing
k_cross_seed = [10, 8, 5, 3, 2, 1]  # much stronger parent influence at specific levels
```

Or simpler: `k_cross_seed = 2 * k_historical` everywhere.

The cross-seed model benefits most from the hierarchy because:
- Many full 6-tuple keys will have 0 samples (alpha=0, pure parent)
- 5-tuple and 4-tuple will have a few samples each (alpha=0.1-0.3, mostly parent)
- 2-tuple (init_class, distance) will have decent coverage (alpha=0.5-0.8)
- This gracefully degrades without losing information at any level

### 8. Alternative: Additive Smoothing (Simpler)

If the full hierarchical approach is too complex, a simpler middle ground:

```python
def additive_smoothed(counts, alpha=0.5, num_classes=6):
    """
    Dirichlet prior with symmetric alpha (Jeffreys prior at alpha=0.5).
    """
    n = sum(counts.values())
    return {
        cls: (counts.get(cls, 0) + alpha) / (n + alpha * num_classes)
        for cls in all_classes
    }
```

This doesn't use the hierarchy but handles sparse keys. With alpha=0.5 (Jeffreys prior):
- n=0: uniform (1/6 each)
- n=10: slight smoothing
- n=50: almost pure MLE

**But the hierarchical approach is strictly better** because it uses parent-level information instead of just a uniform prior.

## Implementation Recommendation

### Concrete Implementation Plan

```python
class HierarchicalPredictor:
    """
    Hierarchical feature model with Dirichlet-smoothed interpolation.

    Feature hierarchy (most to least important):
    L0: (init_class, distance, food_bin, ocean_dist, frontier, density_bin)
    L1: (init_class, distance, food_bin, ocean_dist, density_bin)
    L2: (init_class, distance, food_bin, ocean_dist)
    L3: (init_class, distance, food_bin)
    L4: (init_class, distance)
    L5: (init_class,)
    L6: uniform
    """

    FEATURE_ORDER = ['init_class', 'distance', 'food_bin', 'ocean_dist', 'density_bin', 'frontier']
    # Drop from the right: frontier first, then density_bin, etc.

    def __init__(self, num_classes=6, k_values=None):
        self.num_classes = num_classes
        self.k_values = k_values or [5, 5, 5, 5, 3, 2]  # per level, L0..L5
        self.level_tables = [{} for _ in range(6)]  # L0..L5

    def add_observation(self, features, terrain_class):
        """Add a single observation to all hierarchy levels."""
        for level in range(6):
            key = tuple(features[:6 - level])  # drop rightmost features
            if key not in self.level_tables[level]:
                self.level_tables[level][key] = Counter()
            self.level_tables[level][key][terrain_class] += 1

    def predict(self, features):
        """Return smoothed probability distribution over terrain classes."""
        uniform_p = 1.0 / self.num_classes
        dist = {c: uniform_p for c in range(self.num_classes)}

        # Walk from general (L5) to specific (L0)
        for level in range(5, -1, -1):
            key = tuple(features[:6 - level]) if level < 6 else ()
            if key in self.level_tables[level]:
                counts = self.level_tables[level][key]
                n = sum(counts.values())
                k = self.k_values[level]
                alpha = n / (n + k)

                for c in range(self.num_classes):
                    child_p = counts.get(c, 0) / n
                    dist[c] = alpha * child_p + (1 - alpha) * dist[c]

        return dist
```

### Precomputation for Speed

Since this runs during game rounds with a 2-second budget, precompute the smoothed tables:

```python
def precompute_all_distributions(self):
    """Precompute smoothed distributions for all observed keys at L0."""
    self.cache = {}
    for key in self.level_tables[0]:
        self.cache[key] = self.predict(list(key))

    # Also cache L1-L5 keys for fallback
    for level in range(1, 6):
        for key in self.level_tables[level]:
            if key not in self.cache:
                features = list(key) + [0] * level  # pad with defaults
                self.cache[key] = self.predict(features)
```

## Gotchas & Considerations

- **Feature ordering matters**: The hierarchy assumes features are ordered by importance. If your importance ranking is wrong, the smoothing will be suboptimal. Validate with cross-validation.
- **k sensitivity**: The model is moderately sensitive to k. Too low (k=1) = under-smoothing (noisy with small n). Too high (k=50) = over-smoothing (ignores data). Start with k=5 and tune.
- **Non-symmetric Dirichlet**: If terrain classes have very unequal base rates, consider using class-specific alpha values (non-symmetric Dirichlet prior) rather than uniform pseudocounts.
- **Cross-seed cold start**: For the first few rounds of a new game, the cross-seed model has near-zero data. It will naturally fall back to the historical model's predictions through the hierarchy. This is a feature, not a bug.
- **Overfitting risk at L0**: With only 37 median samples and 6 classes, L0 estimates are borderline. The smoothing naturally handles this, but be aware that L0 distributions for rare keys (n<10) are mostly parent-derived.

## Comparison: Approaches

| Approach | Complexity | Quality | Speed |
|----------|-----------|---------|-------|
| Hard fallback (current) | Low | Poor (loses 4 features at once) | Fast |
| Additive smoothing (flat) | Low | Medium (no hierarchy) | Fast |
| Interpolated hierarchy (recommended) | Medium | High | Fast (precomputed) |
| Full Bayesian (MCMC) | High | Highest | Too slow for real-time |
| Kneser-Ney | High | N/A (wrong structure) | N/A |

## Sources

1. [Smoothing for Language Models - ML Wiki](http://mlwiki.org/index.php/Smoothing_for_Language_Models) — Jelinek-Mercer and Dirichlet smoothing formulas, equivalence between them
2. [Understanding Dirichlet-Multinomial Models - Gundersen](https://gregorygundersen.com/blog/2020/12/24/dirichlet-multinomial/) — Posterior mean formula, pseudocount interpretation
3. [Additive Smoothing - Grokipedia](https://grokipedia.com/page/Additive_smoothing) — Alpha parameter roles, MSE tradeoff, Bayesian connection
4. [An Empirical Study of Smoothing Techniques - Chen & Goodman 1998](https://aclanthology.org/P96-1041.pdf) — Definitive comparison: interpolation beats backoff
5. [Bayesian Inference for Categorical Data Analysis - Agresti](https://users.stat.ufl.edu/~aa/cda/bayes.pdf) — Hierarchical Bayesian approaches for categorical data
6. [Stanford NLP Smoothing Tutorial - MacCartney 2005](https://nlp.stanford.edu/~wcmac/papers/20050421-smoothing-tutorial.pdf) — Overview of all major smoothing techniques
7. [Bayesian Inference for Dirichlet-Multinomials - Johnson](https://users.cecs.anu.edu.au/~ssanner/MLSS2010/Johnson1.pdf) — Dirichlet-multinomial conjugate prior mechanics
8. [Cornell Smoothing + Backoff Lecture](https://www.cs.cornell.edu/courses/cs4740/2014sp/lectures/smoothing+backoff.pdf) — Interpolation vs backoff comparison
9. [MultinomialCI R Package](https://cran.r-project.org/web/packages/MultinomialCI/MultinomialCI.pdf) — Minimum sample requirements for multinomial CIs
10. [Rule of Succession in Multinomial Contexts - FasterCapital](https://fastercapital.com/content/Laplace-s-Rule-of-Succession--Predicting-the-Next-Success--Laplace-s-Rule-in-Multinomial-Contexts.html) — Laplace smoothing extended to multinomial
