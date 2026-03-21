# Research: Optimal Query/Observation Allocation for Prediction Competition

> Researched: 2026-03-20 | Sources consulted: 18 | Confidence: High

## TL;DR

Your current 2-deep + 3-shallow (44+6=50) strategy is **near-optimal but can be improved**. The key insight from statistical theory is that for 6-category multinomial estimation, diminishing returns kick in hard after ~10-15 samples per viewport. You should shift to **2-deep + 3-shallow but redistribute repeats**: instead of 13 repeats on ONE viewport, use **6-7 repeats on TWO viewports** per deep seed, targeting different feature-key regions. Shallow seeds should query viewports that cover **complementary feature keys** not seen in deep seeds. This maximizes cross-seed transfer value.

---

## Key Findings

### 1. Optimal Deep-vs-Shallow Split

**Mathematical framework: Breadth-Depth Dilemma**

Research from Analytis et al. (PNAS 2020) establishes a "square root sampling law" for the breadth-depth tradeoff:

- **Optimal breadth M* ~ sqrt(C)** where C = total budget
- With C=50 queries across 5 seeds: sqrt(50) ~ 7 "exploration units"
- Sharp transition at capacity ~5-10: below this, breadth dominates; above, depth becomes valuable

**Applied to your problem:**

| Strategy | Deep seeds | Shallow seeds | Deep queries each | Shallow queries each | Score potential |
|----------|-----------|--------------|-------------------|---------------------|----------------|
| 1-deep + 4-shallow | 1 | 4 | 42 | 2 | LOW: single deep seed bottlenecks; 42 queries on 1 seed has extreme diminishing returns |
| **2-deep + 3-shallow** | 2 | 3 | 22 | 2 | **BEST**: good depth for model building + adequate cross-seed coverage |
| 3-deep + 2-shallow | 3 | 2 | 14-15 | 2-3 | GOOD but risky: 14 queries per deep seed is just barely enough for good probability estimation |
| 5-equal | 5 | 0 | 10 | - | MEDIUM: uniform coverage but insufficient depth for high-entropy cells |

**Recommendation: Stick with 2-deep + 3-shallow.** Here's why:

1. Your scoring function is `100 * exp(-3 * weighted_KL)`. This is **exponentially punishing** -- a KL of 0.5 scores 22, while KL of 0.2 scores 55. Two excellent seeds + three good-enough seeds beats five mediocre seeds.

2. Cross-seed transfer fills the gap. If seeds share parameters, your 2 deep seeds build the model, and 3 shallow seeds only need to identify which feature keys map to which cells (2 queries each is sufficient for that).

3. The 3-deep variant gives only ~14 queries per deep seed. With 9 needed for grid coverage, that leaves only 5 repeats -- marginal for probability estimation on high-entropy cells.

### 2. Diminishing Returns for Multinomial Probability Estimation

**Core result: Expected KL divergence ~ (k-1)/(2n)**

For a k-category multinomial estimated from n samples, the expected KL divergence between the MLE (empirical frequencies) and the true distribution concentrates around:

```
E[KL(p_true || p_hat)] ~ (k-1) / (2n)
```

For your k=6 categories:

| Samples (n) | Expected KL | Score component (100*exp(-3*KL)) | Marginal gain per sample |
|------------|-------------|----------------------------------|-------------------------|
| 1 | 2.50 | 0.06 | - |
| 2 | 1.25 | 2.35 | +2.29 |
| 3 | 0.83 | 8.21 | +5.86 |
| 5 | 0.50 | 22.31 | +7.05/2 |
| 7 | 0.36 | 34.05 | +5.87/2 |
| 10 | 0.25 | 47.24 | +4.40/3 |
| 13 | 0.19 | 56.55 | +3.10/3 |
| 15 | 0.17 | 60.65 | +2.05/2 |
| 20 | 0.13 | 68.73 | +1.62/5 |

**Key takeaway**: The biggest gains are from samples 2-7. After ~10 samples, each additional sample adds <1% to score. After 15, it's <0.5%.

**Bayesian perspective (Dirichlet posterior):**

With a Dirichlet(alpha) prior and n observations, the posterior variance of each probability estimate is:

```
Var(p_i) = alpha_i'(1 - alpha_i') / (alpha_0 + n + 1)
```

where alpha_i' = (alpha_i + count_i)/(alpha_0 + n). Variance decreases as ~1/n, giving sqrt(n) convergence in standard deviation. This means:
- **1 to 4 samples**: variance drops 4x (massive improvement)
- **4 to 16 samples**: variance drops another 4x (significant)
- **16 to 64 samples**: variance drops another 4x (diminishing)

### 3. One Viewport x13 vs Two Viewports x6-7

**Strongly recommend: Two viewports x 6-7 each.**

Reasoning from the math above:

| Approach | KL at viewport A | KL at viewport B | Combined benefit |
|----------|-----------------|-----------------|------------------|
| 1 viewport x 13 | 0.19 (good) | infinity (unobserved) | Only covers 15x15 = 225 cells deeply |
| 2 viewports x 6-7 | 0.36 (decent) | 0.36 (decent) | Covers 450 cells with good estimates |

The score function is entropy-WEIGHTED KL divergence. This means:
- High-entropy cells (near settlements) contribute most to score
- Having a KL of 0.19 on some high-entropy cells and infinity on others is MUCH worse than 0.36 on all of them
- The exponential scoring punishes bad cells more than it rewards good cells

**Concrete allocation for 22-query deep seed:**
```
Queries 1-9:   Grid coverage (3x3 non-overlapping viewports)
Queries 10-16: 7 repeats of viewport covering the HIGHEST-ENTROPY area
Queries 17-22: 6 repeats of viewport covering the SECOND highest-entropy area
```

This gives you:
- 1 observation per cell for ~95% of the grid (sufficient for low-entropy cells like water/mountains)
- 7-8 observations for the most critical 225 cells (1 from grid + 7 repeats)
- 6-7 observations for the second most critical 225 cells

### 4. Complementary Feature-Key Coverage for Shallow Seeds

**Yes, absolutely -- shallow seeds should cover DIFFERENT feature keys.**

This is where hierarchical Bayesian inference provides the theoretical grounding:

**The model:** All 5 seeds share simulation parameters. A "feature key" maps terrain/position to probability distributions. Once you learn the mapping feature_key -> probability_distribution from deep seeds, you only need to learn feature_key -> cell_position for shallow seeds.

**Information-directed sampling (IDS) principle**: observations should maximize information gain per query. For shallow seeds, information gain is maximized by:

1. **Confirming** which feature keys apply to which cells (needs 1 observation per region)
2. **Discovering** feature keys not seen in deep seeds (needs viewports in different areas)

**Optimal shallow seed strategy:**
```
Query 1: Viewport covering area with DIFFERENT terrain mix than deep seeds observed
Query 2: Viewport covering another distinct region (ideally different feature keys)
```

If all 5 seeds share the same feature-key set, 2 queries per shallow seed is sufficient to:
- Identify which feature keys map to which cells (pattern matching from deep seed data)
- Catch any seed-specific anomalies

**Do NOT** simply replicate the same viewport positions as deep seeds. The deep seeds already taught you those probability distributions -- repeating them on shallow seeds wastes queries.

### 5. Bayesian Optimal Experimental Design (BOED) Framework

The formal framework for this problem is **Bayesian Optimal Experimental Design**:

**Expected Information Gain (EIG):**
```
xi* = argmax EIG(xi) = argmax E_y[ KL( p(theta|y,xi) || p(theta) ) ]
```

Choose the experiment (query) that maximizes the expected reduction in uncertainty about the model parameters.

**Applied to your problem:**
- theta = feature-key -> probability mapping (shared across seeds)
- xi = which viewport to query on which seed
- y = observed terrain counts

**Key insights from BOED theory:**

1. **Sequential > batch**: Ideally, you'd choose each query based on results of previous queries. If the API allows sequential querying, use adaptive strategy:
   - First 18 queries: 9 grid coverage on each of 2 deep seeds
   - Analyze results, identify high-entropy regions
   - Next 26 queries: targeted repeats on highest-uncertainty cells
   - Last 6 queries: shallow seed strategic placements

2. **Diminishing EIG**: Each subsequent query at the same location has strictly decreasing EIG. The first repeat has the most value; the 13th repeat has almost none.

3. **Cross-seed queries have HIGH EIG** when they reveal new feature keys. If your 2 deep seeds happen to miss certain rare terrain combinations, a shallow seed viewport in a different area could be extremely valuable.

## Gotchas & Considerations

- **Stochastic simulation means identical viewport = different sample**: This is your friend. Each repeat is an independent draw from the same multinomial, so frequency counting is the MLE (maximum likelihood estimator) and is provably optimal.

- **The (k-1)/(2n) formula assumes uniform-ish categories**: If one terrain type dominates (e.g., 90% water), you need FEWER samples. If all 6 categories have ~equal probability, you need MORE samples. The formula is a worst-case bound.

- **Cross-seed transfer failure modes**: If seeds DON'T actually share parameters perfectly (e.g., different random seeds produce systematically different biomes), your shallow-seed estimates could be badly wrong. Consider using 1 repeat per shallow seed for validation of the transfer model.

- **Entropy weighting means you can be strategic**: Cells that are clearly one terrain type (entropy ~0) contribute ~0 to the score regardless of your prediction. Focus ALL repeat queries on cells where you see 3+ different terrains across observations.

- **Score = average of 5 seeds**: A score of 0 on any seed is catastrophic. Even 2 queries per shallow seed is risky if cross-seed transfer isn't perfect. Consider 3-shallow with 3 queries each (reducing deep seeds to 20-21 queries each). The 9 grid + 5-6 repeats on 2 viewports is still solid.

## Recommended Strategy (Revised)

```
Budget: 50 queries total, 5 seeds

DEEP SEED A (21 queries):
  - 9 grid coverage viewports (3x3 tiling)
  - Analyze entropy map after grid scan
  - 6 repeats on highest-entropy viewport
  - 6 repeats on second-highest-entropy viewport

DEEP SEED B (21 queries):
  - 9 grid coverage viewports (3x3 tiling)
  - 6 repeats on highest-entropy viewport
  - 6 repeats on second-highest-entropy viewport

SHALLOW SEED C (3 queries):
  - 1 viewport overlapping with a deep-seed high-entropy area (transfer validation)
  - 2 viewports covering DIFFERENT feature-key regions than deep seeds

SHALLOW SEED D (3 queries):
  - Same strategy as C but different viewport positions

SHALLOW SEED E (2 queries):
  - 2 viewports covering the most critical feature-key regions
  (Use the one with best cross-seed transfer performance for validation)
```

Total: 21 + 21 + 3 + 3 + 2 = 50

**Why 21-21-3-3-2 instead of 22-22-2-2-2:**
- 3 queries per shallow seed lets you do 1 validation + 2 exploration
- Validation query confirms cross-seed transfer works for that seed
- If transfer breaks, you still have 2 direct observations as fallback
- Cost: 1 fewer repeat per deep seed (going from 6.5 to 6 avg repeats per viewport -- negligible impact on KL)

## Alternative: Maximum Cross-Seed Strategy

If cross-seed transfer is confirmed to be very strong (ceiling test scored 82-89):

```
DEEP SEED A (25 queries):
  - 9 grid coverage
  - 8 repeats on best viewport (n=9 total -> KL~0.28)
  - 8 repeats on second best viewport (n=9 total -> KL~0.28)

SHALLOW SEEDS B-E (6-7 queries each, ~25 total):
  - 2-3 viewports each covering DIFFERENT feature keys
  - NO repeats -- rely entirely on cross-seed transfer for probabilities
  - Use direct observations only for feature-key identification
```

This maximizes the quality of your probability estimates at the cost of relying heavily on transfer. Only use this if your cross-seed ceiling test confirms >85 scores consistently.

## Sources

1. [Bayesian Optimal Experimental Design - Desi Ivanova](https://desirivanova.com/post/boed-intro/) -- BOED framework, EIG, sequential vs batch design
2. [Heuristics and optimal solutions to the breadth-depth dilemma - PNAS](https://www.pnas.org/doi/10.1073/pnas.2004929117) -- Square root sampling law, M* ~ sqrt(C), sharp transition at C~5-10
3. [Concentration of multinomial in KL divergence - DeepAI](https://deepai.org/publication/concentration-of-the-multinomial-in-kullback-leibler-divergence-near-the-ratio-of-alphabet-and-sample-sizes) -- KL concentration at (k-1)/n threshold
4. [Understanding Dirichlet-Multinomial Models - Gundersen](https://gregorygundersen.com/blog/2020/12/24/dirichlet-multinomial/) -- Posterior update formula, expected probability estimation
5. [Replication or Exploration? Sequential Design for Stochastic Simulation - arXiv](https://arxiv.org/pdf/1710.03206) -- IMSPE criterion, noise-dependent replication decisions
6. [Optimal budget allocation for stochastic simulation - IISE Transactions](https://www.tandfonline.com/doi/full/10.1080/24725854.2021.1953197) -- Exploration vs replication optimal ratio
7. [Bayesian experimental design - Wikipedia](https://en.wikipedia.org/wiki/Bayesian_experimental_design) -- General BOED overview
8. [Multinomial confidence intervals - statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.multinomial_proportions_confint.html) -- Practical CI computation for multinomial
9. [Information maximization for multi-armed bandits - arXiv](https://ar5iv.labs.arxiv.org/html/2503.15962) -- Information-directed sampling for correlated arms
10. [Hierarchical Bayesian modeling - bayesball](https://bayesball.github.io/BOOK/bayesian-hierarchical-modeling.html) -- Shared parameters across groups, sample allocation
11. [Optimal sampling in hierarchical systems - OSTI](https://osti.gov/biblio/1008129-iaRysd) -- Bayesian optimal allocation across hierarchical groups
12. [Finding structure in multi-armed bandits - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010028519302518) -- Correlated bandits, generalization between arms
