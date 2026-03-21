# Astar Island Experiment Log

**Les ALLTID denne filen før du starter nye eksperimenter. Ikke gjenta ting som er testet.**

## Current Baseline
- **Solver:** `solver_v3.py` med Dirichlet pseudo-count + cross-seed + v2 features
- **Params:** concentration=3.0, cross_scale=2.0, floor=0.001, 5-deep(10) query strategy
- **Backtest scores (LOO, cross-seed fra GT):** R2=88.4, R4=91.8, R5=85.9, R6=85.5, R7=70.4, R8=92.4, AVG=85.7
- **Live scores:** R1=16.5, R2=71.3, R3=22.7, R4=75.6, R5=55.4, R6=48.8, R7=63.2, R8=87.5, R9=92.0, R10=91.6, R11=89.6, R12=71.0, R13=91.8, R14=85.8, R15=91.6
- **Leaderboard:** Rank 37→20→37, weighted=190.4 (topp=196.6)
- **Gap to top:** 3.0 raw pts (we score 91.5 where top scores 94.5 on brutal rounds)

## What Works (implemented in solver_v3)
| Feature | Impact | Notes |
|---------|--------|-------|
| Cross-seed feature model | +25 pts | THE key insight. All seeds share hidden params. |
| Dirichlet pseudo-count blending | +7.3 pts (noisy obs) | Replaces hardcoded if/elif. concentration=3, cross_scale=2.0 |
| V2 features (food_bin, ocean_dist, frontier, density_bin) | +1-3.5 pts | Better than old (n_forest, near_ocean) |
| 5-deep (all seeds grid coverage) | +0.53 pts | Max cross-seed breadth: 8000 cells, 124 feature keys |
| Floor 0.001 (was 0.01) | +1.71 pts | Lower floor = less wasted probability mass |
| Cross_scale 2.0 (was 1.0) | +0.37 pts | Cross-seed weighted heavier |
| Removed post-hoc settlement adjustment | +0.29 pts | Was harmful |
| Saved obs fallback on --submit | bugfix | Prevents empty predictions when budget=0 |
| Cross-seed >=1 filter (was >=2) | ~0 but correct | Don't throw away single observations |
| Proportional port redistribution | ~0 but correct | Better than flat redistribution |

## Tested and REJECTED (51 experiments)

### Parameter tuning (11 experiments)
| Experiment | Result | Why |
|-----------|--------|-----|
| Temperature scaling (T=0.5-5.0) | T=1.0 optimal | Already at optimum |
| Cross_scale 5-15 extended sweep | NOT reproduced in solver | Experiment used different baseline. cs=2 best in full solver |
| Lower concentration (0-2) | -0.05 to -0.33 | Historical prior HELPS. More is better |
| Dynamic concentration by round type | -0.11 | Lower conc worse regardless of round type |
| Adaptive concentration per round type | ~0 | Cross-seed dominates, historical prior irrelevant |
| Per-init-class Dirichlet params (settle cs=10) | +0.07 | Below threshold. Cross-seed counts dominate |
| Direct observation boost (2-20x) | -0.01 to -2.41 | Cross-seed dominance is correct |
| Distance-adaptive init_class boost (dist>=4-6, boost 1-20) | +0.01 | Entropy-weighting already ignores wilderness cells |
| Cross-seed decay for far cells (factor 0.5-0.95) | ~0 | Reducing cross-seed at distance hurts |
| Forest non-frontier boost (2-50) | +0.01 | Model already knows non-frontier forest is stable |
| Forest distance decay factor (0.5-5.0) | +0.01 | Same — cross-seed handles this |

### Feature engineering (8 experiments)
| Experiment | Result | Why |
|-----------|--------|-----|
| Transition rate bins as feature | 0 effect | Too few rounds, unique bin per round |
| Local patch features (n_settle_4, n_forest_4, cluster_5x5) | -0.17 | Existing features cover neighbor context |
| Finer distance granularity (various bins) | ~0 | d<=8 with 9 values is optimal |
| Coastal feature (od<=1 AND dist<=3) | 0.00 | ocean_dist already captures this |
| Settlement-enriched features for stable rounds | 0 | Settlements only 200/8000 dynamic cells |
| Hierarchical feature fallback (7 levels, k=3-20) | +0.03 | Most keys have enough data |
| Food potential as continuous (unbinned) | not tested | Feature research suggested it |
| Per-init-class separate models | 0 effect | init_class already in key |

### Model architecture (10 experiments)
| Experiment | Result | Why |
|-----------|--------|-----|
| MLP 64-32 (sklearn MLPRegressor) | 0 | Feature space too simple for interpolation |
| MLP + fine-tune on cross-seed | +3.4 pts | IS cross-seed transfer — already have via lookup |
| Log-linear pooling (geometric mean) | -0.93 | Amplifies errors in noisy estimates |
| Ensembling (3 model variants) | -0.06 to -0.53 | Weaker models drag down |
| Gaussian smoothing (sigma 0.3-2.0) | 0 | Features already capture spatial correlation |
| Bayesian observation denoising | -33 to -59 | Catastrophic. One-hot IS correct |
| Jeffreys smoothing in cross-seed | -0.01 to -0.19 | Dilutes correct predictions |
| Settlement-conditional mixture model | +0.04 at w=0.3 | 4 seeds too few for conditional distributions |
| Two-stage rate estimation | +7 experiment, -0.2 to -3.8 solver | Redundant with cross-seed |
| MC simulator blend (weight 0.5-20) | +0.01 | Heuristic params too coarse. Adds noise not signal |

### Cross-seed model (7 experiments)
| Experiment | Result | Why |
|-----------|--------|-----|
| Confidence-weighted (sqrt/log/cap/shrinkage) | 0 | Linear scaling already optimal |
| k-NN feature fallback | ~0 | Only 1-7 cells need fallback |
| Per-seed cross-seed exclusion | -0.95 | All seeds sharing is correct |
| Similarity-weighted cross-seed | -0.48 | Feature-key is implicit similarity |
| Obs-corrected cross-seed (feature-level) | +10.55 experiment, -0.5 solver | Helps noisy obs, hurts perfect GT |
| Observation-informed historical weighting | ~0 | Cross-seed dominates |
| Per-transition rate correction | -2.7 to -27.8 | Cross-seed rates = avg of all rounds |

### Query/observation strategy (6 experiments)
| Experiment | Result | Why |
|-----------|--------|-----|
| Entropy-based viewport selection | +0.10 | Settlement-density covers 80-90% already |
| All-in-one-seed (50 queries on 1 seed) | -1.28 | Breadth > depth |
| Query reallocation 2-deep(25) + 3-zero | -2.35 | 5 seeds breadth still wins |
| Spatial cross-seed (position+3x3 match) | -0.31 | Feature-key matching is better |
| Multi-viewport (1x13 vs 2x7 vs 4x3 vs 13x1) | 4x3 +0.59 | Led to current 5-deep strategy |
| Winter severity MLE from food | fails | Survivor bias |

## Meta-Analysis Findings (from 14-round deep analysis)
- **Feature-key ceiling:** ~75-77 for hard rounds, 90+ for brutal
- **Top-5 universal loss keys:** Same 5 feature keys dominate loss in ALL 14 rounds (inland cells near settlements with high food)
- **Loss split:** ~60% Empty, ~30% Forest, ~6% Settlement
- **Distance effect:** dist 6-8 has 3-10x worse KL ratio, but low entropy-weight = small score impact
- **Forest over-represented:** 30.6% of loss on 27.1% of entropy weight
- **Port blind:** 0% port accuracy across all rounds (ports share keys with non-port cells)
- **The 3-pt gap to top:** concentrated in Empty/Forest calibration at dist 2-5, NOT settlements

### Fresh eyes round (from Claude Web suggestions, 3 experiments)
| Experiment | Result | Why |
|-----------|--------|-----|
| Observed class in feature key | -1.12 | Fragments cross-seed into 6x more buckets. 4 seeds too sparse. |
| ABC posterior conditioned on observations | ~0 | Match rate variance too low. Static cells dominate, all runs look alike. |
| Adaptive viewport targeting (loss hotspots) | +0.15 | Consistent but below threshold. Full grid coverage > targeted. |

### Simulator-based (2 experiments)
| Experiment | Result | Why |
|-----------|--------|-----|
| MC simulator blend (heuristic params) | +0.01 | Heuristic params too coarse. Adds noise not signal. |
| ABC posterior sampling (rejection) | ~0 | Simulator doesn't match server well enough for conditioning. |

### Calibration & estimation (4 experiments)
| Experiment | Result | Why |
|-----------|--------|-----|
| Isotonic regression calibration | +0.08 | Biases are real but small. High-volume bins well-calibrated. |
| Per-cell gradient refinement (KL loss + L2 reg) | +0.23 avg but -5.3 on R4 | Too unstable. Overfits to cross-seed on some rounds. |
| Empirical Bayes per-group concentration | +0.06 | Right direction but tiny effect. Cross-seed dominates. |
| Combined EB + per-init-class cs | +0.13 in experiment, REGRESSED in solver | Experiment baseline differs from full solver. Does not reproduce. |

### Simulator-based (2 experiments)
| Experiment | Result | Why |
|-----------|--------|-----|
| MC simulator blend (heuristic params) | +0.01 | Heuristic params too coarse. Adds noise not signal. |
| ABC posterior sampling (rejection) | ~0 | Match rate variance too low. Static cells dominate. |

## Key Learning: Experiment-to-Solver Gap
Multiple experiments showed +0.1 to +0.5 improvement but REGRESSED when implemented in solver_v3.py. Root cause: experiments use a slightly different baseline (different feature extraction timing, different cross-seed building, etc.). **Always verify in full solver backtest before declaring victory.**

## Remaining Untested Ideas
| Idea | Source | Expected | Notes |
|------|--------|----------|-------|
| Faction/owner clustering | metadata research | unknown | Voronoi territories from owner_id |
| Exact server simulator reverse-engineering | fresh eyes | +3 pts if possible | Would need to match server sim exactly |

## Key Insight: Stable Round Bottleneck
- R12=71.0 vs R11=89.6 — both "stable" but R12 has 21% settlement death vs 5%
- Cross-seed model predicts 0% settlement death — doesn't distinguish dying vs surviving settlements
- Bottleneck is Forest (score 38-58) and Empty (48-66) cells, NOT settlements
- Settlement metadata (food/pop) could help but only available for observed viewport cells

## Key Insight: Feature-Key Barrier
- Two cells with identical feature keys have fundamentally different GT distributions
- Example: Empty at dist=2, same food/ocean/frontier → but one is near surviving settlement (P(S)=0.4), other near dying (P(S)=0.1)
- This cannot be solved with feature engineering — it requires per-cell information
- Cross-seed model averages these, landing at P(S)=0.25 for both → KL loss on both

## Observation Strategy
- **Current:** 5-deep (10 queries each: 9 grid + 1 settlement viewport)
- **Total:** 50 queries
- **Tested live:** R7 (2-deep), R8-R9 (2-deep+Dirichlet), R10+ (5-deep)

## Historical Rounds Reference
| Round | E->S% | S->S% | S->E% | F->S% | Character | Live Score |
|-------|-------|-------|-------|-------|-----------|------------|
| R1 | 0.5 | 57.2 | 41.3 | 1.2 | Stable | 16.5 |
| R2 | 19.1 | 43.2 | 38.0 | 19.8 | High expansion | 71.3 |
| R4 | 7.8 | 27.0 | 43.8 | 8.5 | Moderate death | 75.6 |
| R5 | 0.6 | 29.3 | 70.7 | 2.8 | Brutal winter | 55.4 |
| R6 | 3.6 | 57.7 | 42.3 | 9.6 | Stable | 48.8 |
| R7 | 6.3 | 60.5 | 38.7 | 14.9 | Stable+expansion | 63.2 |
| R8 | 0.0 | 0.0 | 100.0 | 0.0 | Ultra-brutal | 87.5 |
| R9 | 0.0 | 1.9 | 98.1 | 0.0 | Ultra-brutal | 92.0 |
| R10 | 0.0 | 0.0 | 100.0 | 0.0 | Ultra-brutal | 91.6 |
| R11 | 5.7 | 95.2 | 4.8 | 16.6 | Very stable | 89.6 |
| R12 | 10.5 | 78.6 | 19.5 | 17.6 | Stable+death | 71.0 |
| R13 | 0.0 | 0.8 | 99.2 | 0.0 | Ultra-brutal | 91.8 |
| R14 | 13.6 | 86.3 | 12.3 | 26.0 | Stable+high F->S | 85.8 |
| R15 | 0.0 | 10.0 | 90.0 | 0.0 | Ultra-brutal | 91.6 |
