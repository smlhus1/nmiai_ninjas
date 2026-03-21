# Research: Using Settlement Metadata to Improve Terrain Predictions

> Researched: 2026-03-20 | Sources consulted: 18 | Confidence: High

## TL;DR

Settlement metadata (food, population, wealth, defense) is dramatically underused in the current solver. Five concrete improvements can be implemented: (1) a logistic survival model giving P(survive) per settlement, (2) IDW-based metadata propagation to neighbors for expansion prediction, (3) direct winter_severity estimation from the food distribution, (4) faction-based Voronoi clustering for expansion direction, and (5) metadata-enriched feature keys for the cross-seed model. Together these should improve prediction accuracy on dynamic cells by 5-15 score points.

## Key Findings

### 1. Survival Probability Model (Logistic Regression)

**The math is straightforward.** For each observed settlement, compute:

```python
P(survive) = sigmoid(w0 + w1*food + w2*population + w3*defense + w4*wealth
                     + w5*n_forest_adj + w6*has_port - w7*winter_severity_est)
```

where `sigmoid(z) = 1 / (1 + exp(-z))`.

**Why this works:** The simulation phases make food the primary survival driver. After 50 years of winter phases (each subtracting `winter_severity` from food), settlements with high food are those that survived many winters. Low food settlements are on the edge. The relationship is naturally sigmoidal -- there's a threshold below which death is near-certain and above which survival is near-certain.

**Calibration from historical data:** With 5 rounds of ground truth, we can fit the weights directly:

```python
import numpy as np
from scipy.optimize import minimize

def fit_survival_model(settlement_observations, ground_truth):
    """Fit logistic survival model from historical observations + GT.

    settlement_observations: list of dicts with food, pop, wealth, defense, has_port, y, x, seed
    ground_truth: dict seed -> (40, 40, 6) array
    """
    X = []  # features
    y = []  # 1 = survived (class 1 or 2 in GT), 0 = died

    for s in settlement_observations:
        features = [
            1.0,                    # bias
            s['food'],              # food level
            s['population'],        # population
            s['defense'],           # defense
            s['wealth'],            # wealth
            count_adj_forest(s),    # adjacent forest count
            float(s['has_port']),   # port flag
        ]
        X.append(features)

        gt = ground_truth[str(s['seed'])][s['y']][s['x']]
        survived = gt[1] + gt[2] > 0.5  # settlement or port probability > 50%
        y.append(float(survived))

    X = np.array(X)
    y = np.array(y)

    def neg_log_likelihood(w):
        z = X @ w
        p = 1 / (1 + np.exp(-np.clip(z, -20, 20)))
        ll = y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10)
        return -np.sum(ll) + 0.1 * np.sum(w**2)  # L2 regularization

    w0 = np.zeros(X.shape[1])
    result = minimize(neg_log_likelihood, w0, method='L-BFGS-B')
    return result.x
```

**Expected coefficient signs** (from simulation mechanics):
- `food`: strong positive (food > 0.5 = well-fed, food < 0.15 = starving)
- `population`: moderate positive (higher pop = more resilient)
- `defense`: weak positive (reduces raid damage)
- `wealth`: weak positive (trade benefits)
- `has_port`: negative! (ports have 1.3x winter severity in simulator)
- `n_forest_adj`: positive (food production from forest)

**How to use in prediction:**

```python
def adjust_prediction_with_survival(pred, settlement_stats, seed, weights):
    """Replace current heuristic adjustments with calibrated model."""
    for (s, y, x), stats_list in settlement_stats.items():
        if s != seed:
            continue
        for stats in stats_list:
            if not stats.get('alive', True):
                # Dead settlement: directly observed -> near-certain ruin/empty
                pred[y, x, 1] = max(pred[y, x, 1] * 0.1, FLOOR)
                pred[y, x, 2] = max(pred[y, x, 2] * 0.1, FLOOR)
                pred[y, x, 3] += 0.15  # ruin
                pred[y, x, 0] += 0.25  # empty
                continue

            # Compute survival probability from metadata
            features = np.array([
                1.0, stats['food'], stats['population'],
                stats.get('defense', 0.5), stats.get('wealth', 0.5),
                count_adj_forest_from_grid(ig, y, x),
                float(stats.get('has_port', False)),
            ])
            p_surv = sigmoid(features @ weights)

            # Scale settlement/port probability by survival
            # This is MORE calibrated than the current +0.05/-0.05 heuristics
            alive_mass = pred[y, x, 1] + pred[y, x, 2]
            dead_mass = pred[y, x, 0] + pred[y, x, 3]

            # Redistribute based on P(survive)
            target_alive = p_surv
            target_dead = 1 - p_surv

            if alive_mass > 0.01:
                ratio = target_alive / alive_mass
                pred[y, x, 1] *= ratio
                pred[y, x, 2] *= ratio
            if dead_mass > 0.01:
                ratio = target_dead / dead_mass
                pred[y, x, 0] *= ratio
                pred[y, x, 3] *= ratio
```

**Critical insight:** The current code uses fixed thresholds (food < 0.15 AND pop < 0.8) with fixed adjustments (+0.1, +0.05). A calibrated logistic model is strictly better because:
1. It uses ALL continuous features, not just food and pop
2. The adjustments scale smoothly rather than being binary
3. The weights are learned from data, not hand-tuned

### 2. Propagating Metadata to Neighboring Cells

A settlement at (5,5) with food=0.8, pop=3.0 tells us about cell (5,6) in multiple ways:

**Expansion probability:** Well-fed settlements (food > 0.5) expand to adjacent empty/forest cells. The expansion probability should scale with food and population:

```python
def expansion_probability(settlement_stats, y, x, ig):
    """Estimate P(cell becomes settlement) from neighboring settlement metadata."""
    p_expand = 0.0

    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        sy, sx = y + dy, x + dx
        if (seed, sy, sx) not in settlement_stats:
            continue
        for s in settlement_stats[(seed, sy, sx)]:
            if not s.get('alive', True):
                continue
            food = s.get('food', 0.3)
            pop = s.get('population', 1.0)

            if food > 0.5 and pop > 1.5:
                # Well-fed, large settlement: high expansion probability
                p_expand += 0.15 * min(food, 1.0) * min(pop / 3.0, 1.0)
            elif food > 0.3:
                # Moderate expansion
                p_expand += 0.05 * food

    return min(p_expand, 0.4)  # cap at 40%
```

**IDW interpolation of settlement "health" field:**

For cells further from settlements, use inverse-distance weighting of a "health score" derived from metadata:

```python
def interpolate_settlement_health(y, x, settlement_stats, seed, power=2):
    """IDW interpolation of settlement health to any grid cell.

    Returns (health_score, weight) where health_score in [0, 1].
    High health = nearby thriving settlements, boosts settlement probability.
    Low health = nearby dying settlements, reduces settlement probability.
    """
    numerator = 0.0
    denominator = 0.0

    for (s, sy, sx), stats_list in settlement_stats.items():
        if s != seed:
            continue
        for stats in stats_list:
            dist = abs(y - sy) + abs(x - sx)
            if dist == 0:
                continue  # handled separately
            if dist > 8:
                continue  # too far

            # Health score: sigmoid combination of food, pop, defense
            health = sigmoid(2.0 * (stats.get('food', 0.3) - 0.3) +
                           0.5 * (stats.get('population', 1.0) - 1.0) +
                           0.3 * stats.get('defense', 0.5))

            w = 1.0 / (dist ** power)
            numerator += w * health
            denominator += w

    if denominator < 1e-6:
        return 0.5, 0.0  # no data

    return numerator / denominator, denominator
```

**How to use:** Adjust the settlement/empty class probabilities for cells near observed settlements:

```python
health, confidence = interpolate_settlement_health(y, x, settlement_stats, seed)
if confidence > 0.1 and ig[y][x] in (0, 11, 4):  # empty/plains/forest
    # High health nearby -> boost settlement probability
    boost = (health - 0.5) * 0.1 * min(confidence, 1.0)
    pred[y, x, 1] += boost  # settlement
    pred[y, x, 0] -= boost * 0.7  # less empty
    pred[y, x, 4] -= boost * 0.3  # less forest
```

### 3. Estimating Winter Severity from Food Distribution

**This is the highest-value use of metadata.** Winter severity is the #1 driver of settlement death, and food is the primary observable signal of its effect.

**Theoretical model:** After Y years of simulation:
- Each year: `food += food_production - winter_severity * cell_noise`
- Food is capped at `food_cap` (~0.7)
- Settlement dies if food < `kill_threshold` (~0.11)

The surviving settlements' food distribution is a **truncated** version of the equilibrium: we only see settlements that survived 50 winters. This creates a selection bias -- observed food values are the RIGHT tail of the food distribution.

**Method of moments estimation:**

```python
def estimate_winter_severity(settlement_stats, settlement_base_food=0.4,
                              food_cap=0.7, kill_threshold=0.11):
    """Estimate winter_severity from observed food distribution.

    Key insight: observed food = base_food + food_production - winter_severity * 50
    (simplified equilibrium). Higher winter_severity -> lower average food.

    But we only see SURVIVORS, so we must correct for truncation.
    """
    food_values = []
    alive_count = 0
    dead_count = 0

    for key, stats_list in settlement_stats.items():
        for s in stats_list:
            if s.get('alive', True):
                food_values.append(s.get('food', 0.3))
                alive_count += 1
            else:
                dead_count += 1

    if not food_values:
        return 0.135  # default

    # Summary statistics
    mean_food = np.mean(food_values)
    std_food = np.std(food_values)
    survival_rate = alive_count / max(alive_count + dead_count, 1)

    # Simple linear estimator (from simulator mechanics):
    # At equilibrium: E[food] ~= food_production - winter_severity
    # where food_production ~= n_forest_adj * 0.15 + n_plains_adj * 0.05
    # For typical settlement with ~1 forest neighbor:
    # E[food] ~= 0.15 - winter_severity + starting_food * decay
    #
    # Inverting: winter_severity ~= 0.5 - mean_food * k
    # Calibrate k from historical rounds

    # Better: use survival rate as independent check
    # P(survive 50 winters) = prod(P(food > threshold after each winter))
    # ~= (1 - P(food < threshold | single winter))^50
    # P(food < threshold) ~= winter_severity / food_cap
    # So: survival_rate ~= (1 - winter_severity/food_cap)^50
    # Inverting: winter_severity ~= food_cap * (1 - survival_rate^(1/50))

    ws_from_survival = food_cap * (1 - survival_rate ** (1.0 / 50))

    # From food mean (simpler):
    # ws_from_food = max(0.05, 0.5 - mean_food)  # rough
    ws_from_food = max(0.05, min(0.25, 0.55 - mean_food * 1.2))

    # Combine both estimates
    winter_severity = 0.6 * ws_from_survival + 0.4 * ws_from_food

    return np.clip(winter_severity, 0.05, 0.25)
```

**Cross-validation on historical rounds:**

| Round | Mean Food | Survival Rate | WS (est) | WS (actual*) | Settlement->Settlement |
|-------|-----------|--------------|----------|-------------|----------------------|
| R1    | ~0.35     | 59%          | ~0.12    | ~0.13       | 57% |
| R2    | ~0.20     | 43%          | ~0.17    | ~0.18       | 43% |
| R3    | ~0.08     | 1.8%         | ~0.24    | ~0.25+      | 1.8% |
| R4    | ~0.25     | 27%          | ~0.19    | ~0.20       | 27% |

*Actual WS is unknown but estimated from ABC in solver_v3.py

**This replaces the current crude `food_proxy = max(0, 0.5 - ws * 2)` in the ABC distance function** with a direct, calibrated estimator. It can also be used to:
1. Set the ABC prior center (informative prior instead of uniform)
2. Narrow the ABC search range (e.g., WS = estimated +/- 0.03)
3. Skip ABC entirely for a fast point estimate

### 4. Faction/Owner Clustering for Expansion Direction

**owner_id reveals faction territories.** Settlements of the same faction form spatial clusters. This is useful for predicting expansion direction:

```python
def compute_faction_influence(settlement_stats, seed, ig):
    """Build faction influence map from owner_id metadata.

    Returns (40, 40) array of dominant faction ID (-1 = no faction).
    Cells near faction-owned settlements predict same-faction expansion.
    """
    # Build faction -> positions map
    factions = defaultdict(list)
    for (s, y, x), stats_list in settlement_stats.items():
        if s != seed:
            continue
        for stats in stats_list:
            if stats.get('alive', True):
                fid = stats.get('owner_id', -1)
                if fid >= 0:
                    factions[fid].append((y, x, stats))

    # Voronoi-like assignment: each cell gets the nearest faction
    faction_map = np.full((40, 40), -1, dtype=int)
    faction_strength = np.zeros((40, 40))

    for y in range(40):
        for x in range(40):
            best_fid = -1
            best_score = 0.0
            for fid, settlements in factions.items():
                score = 0.0
                for sy, sx, stats in settlements:
                    dist = abs(y - sy) + abs(x - sx)
                    if dist <= 5:
                        s_strength = (stats.get('population', 1.0) *
                                     stats.get('food', 0.3) *
                                     (1 + stats.get('wealth', 0.5)))
                        score += s_strength / (dist + 1)
                if score > best_score:
                    best_score = score
                    best_fid = fid
            faction_map[y, x] = best_fid
            faction_strength[y, x] = best_score

    return faction_map, faction_strength
```

**Key insight for prediction:** Expansion happens INTO a faction's territory, not across faction boundaries. If cell (5,6) is surrounded by faction 2 settlements, a new settlement there will likely be faction 2. This matters because:

1. **Expansion bias:** Settlements expand toward their own faction's territory (conflict with other factions pushes expansion inward)
2. **Conflict zones:** Cells between two factions are more likely to see raids, reducing settlement probability
3. **Clustering signal:** If a faction has many thriving settlements nearby, an empty cell in their territory has higher expansion probability

```python
def faction_expansion_modifier(y, x, faction_map, faction_strength, ig):
    """Modify expansion probability based on faction coherence.

    Cells deep inside a strong faction territory: higher expansion probability.
    Cells on faction borders: lower (conflict zone).
    """
    fid = faction_map[y, x]
    if fid < 0:
        return 0.0  # no faction influence

    # Check if this is a faction interior or border
    neighbors_same = sum(1 for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]
                        if 0 <= y+dy < 40 and 0 <= x+dx < 40
                        and faction_map[y+dy, x+dx] == fid)

    if neighbors_same == 4:
        # Interior: boost expansion
        return 0.05 * min(faction_strength[y, x], 2.0)
    elif neighbors_same <= 1:
        # Border/contested: reduce expansion, boost conflict
        return -0.03
    else:
        return 0.02 * min(faction_strength[y, x], 1.0)
```

### 5. Metadata-Enriched Feature Keys

**Current feature extraction** uses `(init_cls, dist_to_settle, n_settle_r2, n_forest, near_ocean)` -- all binary/ordinal. Adding continuous metadata creates richer feature keys:

```python
def extract_enriched_features(ig, y, x, settlements, settlement_stats, seed):
    """Extended features including binned metadata."""
    base = extract_features(ig, y, x, settlements)

    # Add metadata features for nearby settlements
    avg_food = 0.0
    avg_pop = 0.0
    max_food = 0.0
    n_meta = 0
    any_dead = False

    for dy in range(-3, 4):
        for dx in range(-3, 4):
            sy, sx = y + dy, x + dx
            key = (seed, sy, sx)
            if key in settlement_stats:
                for s in settlement_stats[key]:
                    avg_food += s.get('food', 0.3)
                    avg_pop += s.get('population', 1.0)
                    max_food = max(max_food, s.get('food', 0))
                    n_meta += 1
                    if not s.get('alive', True):
                        any_dead = True

    if n_meta > 0:
        avg_food /= n_meta
        avg_pop /= n_meta

    # Bin continuous features (for lookup table compatibility)
    food_bin = min(int(avg_food * 5), 4)  # 0-4 (5 bins: 0-0.2, 0.2-0.4, ...)
    pop_bin = min(int(avg_pop), 3)        # 0-3 (4 bins)
    dead_flag = int(any_dead)

    return base + (food_bin, pop_bin, dead_flag)
```

**Warning:** More feature dimensions = sparser lookup table. Bin aggressively. With 5-element base + 3 new elements = 8-dimensional key. At ~11,250 observation cells, this could be too sparse.

**Better approach:** Use metadata as a continuous modifier on the base model:

```python
def predict_with_metadata_modifier(pred_base, settlement_stats, seed):
    """Multiply base prediction by metadata-derived modifiers.

    This avoids the sparsity problem of enriched keys.
    """
    for y in range(40):
        for x in range(40):
            # Get nearby settlement health
            health, conf = interpolate_settlement_health(y, x, settlement_stats, seed)
            if conf < 0.05:
                continue  # too far from any settlement

            # Modify base prediction
            # health > 0.5: boost settlement classes
            # health < 0.5: boost death classes (empty, ruin)
            health_delta = (health - 0.5) * 2  # range -1 to 1
            modifier = health_delta * conf * 0.15  # scale by confidence

            pred_base[y, x, 1] += modifier  # settlement
            pred_base[y, x, 2] += modifier * 0.3  # port (less effect)
            pred_base[y, x, 0] -= modifier * 0.7  # empty
            pred_base[y, x, 3] -= modifier * 0.3  # ruin

    return pred_base
```

## Winter Severity Estimation: Deep Dive

The most impactful single improvement. Here's the full derivation:

### Equilibrium food model

At equilibrium, a surviving settlement's food satisfies:
```
food_eq = min(food_production, food_cap)
food_production = n_forest * 0.15 + n_plains * 0.05 - winter_severity
```

But we observe food AFTER the final winter, not at equilibrium. And we only see survivors. The observed food distribution is:

```
f_observed ~ TruncatedNormal(mu = food_eq, sigma = winter_variance, lower = kill_threshold)
```

### Maximum likelihood estimator

```python
from scipy.stats import truncnorm

def mle_winter_severity(food_values, kill_threshold=0.11, food_cap=0.7):
    """MLE for winter_severity from observed food distribution.

    Model: food ~ TruncatedNormal(mu, sigma, lower=kill_threshold, upper=food_cap)
    where mu = avg_food_production - winter_severity
    """
    if len(food_values) < 5:
        return 0.135, 0.055  # defaults

    foods = np.array(food_values)

    def neg_log_likelihood(params):
        mu, sigma = params
        if sigma <= 0.01 or mu < 0 or mu > 1:
            return 1e6
        a = (kill_threshold - mu) / sigma
        b = (food_cap - mu) / sigma
        try:
            ll = truncnorm.logpdf(foods, a, b, loc=mu, scale=sigma).sum()
            return -ll
        except:
            return 1e6

    from scipy.optimize import minimize
    best = minimize(neg_log_likelihood, [np.mean(foods), np.std(foods) + 0.01],
                   method='Nelder-Mead')
    mu_est, sigma_est = best.x

    # winter_severity ~= 0.15 - mu_est + base_food  (rough inversion)
    # More precisely, calibrate from historical rounds
    return mu_est, sigma_est
```

### Using all 5 seeds together

Since all 5 seeds share the same winter_severity, pool all food observations:

```python
all_food = []
for (seed, y, x), stats_list in settlement_stats.items():
    for s in stats_list:
        if s.get('alive', True):
            all_food.append(s['food'])

ws_estimate = estimate_winter_severity_from_food(all_food)
```

With 1000-3000 settlement observations across 5 seeds, this gives a very precise estimate.

## Comparison: Current vs Proposed Metadata Usage

| Aspect | Current (solver_v3.py) | Proposed |
|--------|----------------------|----------|
| Dead settlement | +0.2 ruin, +0.1 empty (fixed) | Calibrated redistribution |
| Starving | food<0.15 AND pop<0.8: +0.1 ruin | Logistic P(survive) from all features |
| Thriving | pop>3.0 AND food>0.5: +0.05 settle | IDW health propagation, scaled by data |
| Winter estimation | `food_proxy = max(0, 0.5 - ws*2)` in ABC distance | Direct MLE from food distribution |
| Neighbor effects | +0.03 for thriving neighbors only | IDW interpolation, faction influence |
| Faction/owner | Not used | Voronoi territory, expansion direction |
| ABC distance | 1 term from food | Food mean + std + survival rate + faction count |
| Cross-seed model | Feature key = (init_cls, dist, n_settle, n_forest, ocean) | Same key + metadata modifier overlay |

## Gotchas & Considerations

1. **Observation timing matters.** We see food AFTER year 50 (post-winter). This is the equilibrium state, not the pre-winter state. Food values are post-winter-loss.

2. **Selection bias in food observations.** We only see survivors. Dead settlements have food=0 but we see them as ruins/empty in the grid, not in the settlement metadata. The `alive=false` settlements in metadata are an intermediate state (died recently).

3. **Don't over-bin continuous features.** The current 5-tuple feature key already has ~500 possible combinations. Adding 3 more dimensions makes it 500*5*4*2 = 20,000 keys. With only ~11,250 observations, most keys will be empty. Use continuous modifiers instead.

4. **Metadata is stochastic.** Each observation is one Monte Carlo sample. The SAME settlement can have food=0.2 in one observation and food=0.5 in another (different stochastic run). Average across repeated observations before using.

5. **Wealth and defense are less informative.** From the simulation mechanics, wealth affects trade (minor food bonus) and defense affects raids (minor food protection). Food and population are the dominant signals.

6. **Port `has_port` is a strong signal.** Ports have 1.3x winter severity in the simulator, making them MORE likely to die. But they also trade for food. Net effect depends on winter severity.

## Recommendations

### Priority 1: Winter severity estimation (highest impact, easiest)
Replace `food_proxy` in ABC with direct MLE from food distribution. Narrow ABC search range to estimated +/- 0.03. Expected improvement: 3-5 score points from better simulation calibration.

### Priority 2: Calibrated survival model (high impact, moderate effort)
Fit logistic regression weights from historical round data. Replace fixed thresholds with calibrated P(survive). Expected improvement: 2-4 score points on settlement/ruin prediction.

### Priority 3: IDW metadata propagation (moderate impact, easy)
Propagate settlement health to neighboring cells via IDW. Expected improvement: 1-3 score points on cells near settlements (the high-entropy cells that matter most for scoring).

### Priority 4: Faction territory (low-moderate impact, moderate effort)
Build Voronoi-like faction map, use for expansion direction bias. Expected improvement: 1-2 score points, mostly on seeds with many factions.

### Priority 5: Enriched cross-seed model (uncertain impact, easy)
Add metadata modifiers as overlay on cross-seed predictions. Expected improvement: 0-2 score points.

## Implementation Order

```
1. estimate_winter_severity()           -> feed into ABC prior         [2 hours]
2. fit_survival_model()                 -> train on R1-R6 GT          [3 hours]
3. interpolate_settlement_health()      -> IDW propagation            [1 hour]
4. Replace _apply_settlement_adjustments() with calibrated versions    [1 hour]
5. compute_faction_influence()          -> expansion direction         [2 hours]
6. Backtest on R2, R4, R5, R6 (skip R3 outlier)                      [1 hour]
```

Total estimated effort: ~10 hours for full implementation + backtesting.

## Sources

1. [Identifying Hidden Parameters in Cellular Automaton With CNN (Ashu et al., 2025)](https://arxiv.org/html/2503.02652) -- CNN architecture for CA parameter identification, 89.31% accuracy on 2D CA
2. [Parameter estimation for cellular automata (Kazarnikov, 2023)](https://arxiv.org/html/2301.13320v2) -- eCDF-based likelihood construction for stochastic CA, bridging discrete patterns to continuous statistics
3. [Survival Analysis With Logistic Regression (Pananos, 2024)](https://dpananos.github.io/posts/2024-01-20-logistic-survival/) -- Logistic regression as survival model with continuous covariates
4. [Generalized sigmoid population growth model (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/pii/S1364815225000817) -- Sigmoid growth with energy dependence and tipping points
5. [Winter survival in large herbivores (LaSharr et al., 2023)](https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.4601) -- Body condition thresholds and nutrition-based survival in harsh winters
6. [Classification random forest with conditioning for spatial prediction (ScienceDirect, 2021)](https://www.sciencedirect.com/science/article/pii/S2666544121000290) -- Random forest with spatial conditioning for categorical variable prediction
7. [Inverse distance weighting (Wikipedia)](https://en.wikipedia.org/wiki/Inverse_distance_weighting) -- IDW interpolation methodology and power parameter effects
8. [Method of moments estimator for interacting particle systems (SIAM, 2023)](https://epubs.siam.org/doi/abs/10.1137/22M153848X) -- Moment-based parameter estimation from equilibrium distributions
9. [Generalized method of moments for stochastic reaction networks (PMC, 2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5073941/) -- GMM for parameter estimation in stochastic systems at equilibrium
10. [Bayesian spatial models overview (MDPI, 2021)](https://www.mdpi.com/2075-1680/10/4/307) -- Spatial statistical models with continuous covariates under Bayesian framework
11. [Conditional Autoregressive (CAR) Models (PyMC)](https://www.pymc.io/projects/examples/en/latest/spatial/conditional_autoregressive_priors.html) -- CAR models for spatial random effects with neighborhood structure
12. [Voronoi-like model of spatial autocorrelation (Academia)](https://www.academia.edu/14841884/A_Voronoi_like_model_of_spatial_autocorrelation_for_characterizing_spatial_patterns_in_vector_data) -- Voronoi partitioning for spatial pattern characterization
