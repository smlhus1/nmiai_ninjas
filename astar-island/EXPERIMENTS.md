# Astar Island Experiment Log

**Les ALLTID denne filen for du starter nye eksperimenter. Ikke gjenta ting som er testet.**

## Current Baseline
- **Solver:** `solver_v3.py` med Dirichlet pseudo-count + cross-seed + v2 features
- **Backtest scores (LOO, cross-seed fra GT):** R2=88.4, R4=91.8, R5=85.9, R6=85.5, R7=70.4, R8=92.4, AVG=85.7
- **Live scores:** R1=16.5, R2=71.3, R3=22.7, R4=75.6, R5=55.4, R6=48.8, R7=63.2, R8=87.5, R9=92.0, R10=91.6, R11=89.6, R12=71.0, R13=91.8, R14=pending
- **Ceiling (cross-seed perfect):** R4=89.3, R5=82.3, R6=82.5
- **Leaderboard:** Rank 19, weighted=173.0 (topp=177.1)

## What Works (implemented in solver_v3)
| Feature | Impact | Notes |
|---------|--------|-------|
| Cross-seed feature model | +25 pts | THE key insight. All seeds share hidden params. |
| Dirichlet pseudo-count blending | +7.3 pts (noisy obs) | Replaces hardcoded if/elif. concentration=3, cross_scale=1.0 |
| V2 features (food_bin, ocean_dist, frontier, density_bin) | +1-3.5 pts | Better than old (n_forest, near_ocean) |
| 5-deep (all seeds grid coverage) | +0.53 pts | Max cross-seed breadth: 8000 cells, 124 feature keys |
| Floor 0.001 (was 0.01) | +1.71 pts | Lower floor = less wasted probability mass. Monotonically better. |
| Cross_scale 2.0 (was 1.0) | +0.37 pts | Cross-seed should be weighted heavier with 8 historical rounds |
| Removed post-hoc settlement adjustment | +0.29 pts | Was harmful — hacky additive adjustments worsened predictions |
| Saved obs fallback on --submit | bugfix | Prevents empty predictions when budget=0 |

## Tested and REJECTED
| Experiment | Result | Why |
|-----------|--------|-----|
| Transition rate bins as feature | 0 effect | Too few rounds, each has unique bin combo. LOO = 100% fallback. |
| Gaussian smoothing (sigma 0.3-2.0) | 0 effect | Features already capture spatial correlation |
| Winter severity MLE from food | fails | Survivor bias — surviving settlements have good food regardless of winter |
| Temperature scaling (T=0.5-5.0) | T=1.0 optimal | Already at optimum. No free points. |
| Hierarchical feature fallback (7 levels, k=3-20) | +0.03 pts | Most keys have enough data. frontier/density rarely split uniquely. |
| MLP 64-32 (sklearn MLPRegressor) | 0 improvement | Feature space too simple/discrete for interpolation to help |
| MLP + fine-tune on cross-seed | +3.4 pts | But this IS cross-seed transfer — we already have it via lookup |
| Log-linear pooling (geometric mean) | -0.93 pts | Amplifies errors in noisy estimates. Dirichlet is better. |
| Confidence-weighted cross-seed (sqrt/log/cap/shrinkage) | 0 effect | Linear scaling is already optimal. Sublinear loses info. |
| k-NN feature fallback | ~0 effect | Only 1-7 cells (0.1%) need fallback. Not worth it. |
| Adaptive concentration per round type | ~0 effect | Cross-seed dominates, historical prior is irrelevant. |
| Entropy-based viewport selection | +0.10 pts | Settlement-density covers 80-90% of high-entropy cells already. |
| Jeffreys smoothing in cross-seed | -0.01 to -0.19 | Most keys have enough data. Smoothing dilutes correct predictions. |
| Direct observation boost (2-20x) | -0.01 to -2.41 | Cross-seed dominance is correct. Boosting direct obs overfits to noise. |
| Bayesian observation denoising | -33 to -59 pts | Catastrophic. Uniform prior destroys signal. One-hot is correct. |

## Untested Ideas (from research)
| Idea | Source | Expected | Notes |
|------|--------|----------|-------|
| Per-transition rate correction (other rounds) | -2.7 to -27.8 | Cross-seed rates = avg of all rounds, not current round. |
| Obs-corrected cross-seed (same round, feature-level) | +10.55 in experiment, -0.5 in backtest | Helps with noisy obs but hurts with perfect GT. Test LIVE. |
| Per-init-class separate models | 0 effect | init_class already in key — no dilution to prevent |
| Ensembling (3 model variants) | -0.06 to -0.53 | Weaker models drag down. Not independent errors. |
| Direct observation boost (2-20x) | -0.01 to -2.41 | Cross-seed dominance is correct feature not bug |
| Bayesian observation denoising | -33 to -59 | Catastrophic. One-hot IS correct, averaging provides smoothing. |
| Cross_scale 5-15 (extended sweep) | +0.52 in experiment, NOT reproduced in solver | Experiment used different baseline. cs=2 is best in full solver. |
| Per-seed cross-seed exclusion | -0.95 | Not data leak — all seeds sharing is correct. |
| Similarity-weighted cross-seed | -0.48 | Feature-key matching already acts as implicit similarity filter. |
| Finer distance granularity | ~0 | d<=8 with 9 values is already optimal. |
| Observation-informed historical weighting | ~0 | Cross-seed dominates, historical irrelevant. |
| Spatial cross-seed transfer (position+3x3 match) | -0.31 | Feature-key matching is already implicit similarity. |
| Local patch features (n_settle_4, n_forest_4, cluster_5x5) | -0.17 | Existing features already cover neighbor context. |
| Two-stage rate estimation (same-round rates) | +7 in experiment, -0.2 to -3.8 in solver | Redundant when cross-seed already has same-round data. |
| All-in-one-seed (50 queries on 1 seed) | -1.28 | Breadth > depth. 1 seed can't cover all feature keys. |
| Cross_scale 5-15 extended sweep | +0.52 in experiment, NOT reproduced in solver | Experiment used different baseline. cs=2 is best in full solver. |
| Per-seed cross-seed exclusion | -0.95 | Not data leak — all seeds sharing is correct. |
| Similarity-weighted cross-seed | -0.48 | Feature-key matching already acts as implicit similarity filter. |
| Finer distance granularity (various bins) | ~0 | d<=8 with 9 values is already optimal. |
| Settlement-enriched features for stable rounds | 0 | Settlements are only 200/8000 dynamic cells. Forest+Empty dominate. |
| Per-init-class Dirichlet params (settle cs=10) | +0.07 | Below threshold. Cross-seed counts dominate regardless. |
| Settlement-conditional mixture model | +0.04 at w=0.3, -1.39 at w=1.0 | Hypothesis correct but 4 seeds too few for conditional distributions. |
| Query reallocation back to 2-deep(25) | -2.35 | Breadth (5 seeds) still beats depth (2 seeds). |
| Cross-seed >=2 filter → >=1 | ~0 | More keys but marginal effect. |
| Proportional port redistribution | ~0 | Correct but negligible impact. |
| Adaptive viewport selection | viewport research | unknown | Choose repeat viewports AFTER seeing grid queries |
| Settlement survival logistic model | metadata research | unknown | P(survive) = sigmoid(w*food + ...) |
| Faction/owner clustering | metadata research | unknown | Voronoi territories from owner_id |
| Food potential as continuous feature | feature research | unknown | Unbin food_potential for MLP/continuous model |
| CNN on local patches | ML research | +1-5 pts | Overkill per ML agent, but could capture local patterns |

## Key Insight: Stable Round Bottleneck
- R12=71.0 vs R11=89.6 — both "stable" but R12 has 21% settlement death vs 5%
- Cross-seed model predicts 0% settlement death — doesn't distinguish dying vs surviving settlements
- Bottleneck is Forest (score 38-58) and Empty (48-66) cells, NOT settlements (only 200/8000)
- Settlement metadata (food/pop) could help but only available for observed viewport cells

## Observation Strategy
- **Current:** 5-deep (10 queries each: 9 grid + 1 settlement viewport)
- **Total:** 50 queries
- **Tested live:** R7 (2-deep), R8-R9 (2-deep+Dirichlet), R10+ (5-deep)

## Historical Rounds Reference
| Round | E->S% | S->S% | S->E% | F->S% | Character |
|-------|-------|-------|-------|-------|-----------|
| R1 | 0.5 | 57.2 | 41.3 | 1.2 | Stable, low expansion |
| R2 | 19.1 | 43.2 | 38.0 | 19.8 | High expansion |
| R4 | 7.8 | 27.0 | 43.8 | 8.5 | Moderate death |
| R5 | 0.6 | 29.3 | 70.7 | 2.8 | Brutal winter |
| R6 | 3.6 | 57.7 | 42.3 | 9.6 | Stable, like R1 |
| R7 | 6.3 | 60.5 | 38.7 | 14.9 | Stable, moderate expansion |
| R8 | 0.0 | 0.0 | 100.0 | 0.0 | Ultra-brutal, all die |
| R9 | 0.0 | 1.9 | 98.1 | 0.0 | Ultra-brutal |
| R10 | 0.0 | 0.0 | 100.0 | 0.0 | Ultra-brutal, all die |
| R11 | 5.7 | 95.2 | 4.8 | 16.6 | Very stable, high survival |
| R12 | 10.5 | 78.6 | 19.5 | 17.6 | Stable, moderate death |
| R13 | 0.0 | 0.8 | 99.2 | 0.0 | Ultra-brutal |
