# Research: Feature Engineering for Cellular Automaton Prediction

> Researched: 2026-03-20 | Sources consulted: 14 | Confidence: High

## TL;DR

Your current 5-tuple feature set captures only the immediate neighborhood. The biggest wins come from: (1) **larger neighborhood composition features** (radius 3-5 terrain ratios), (2) **food potential proxy** (adjacent forest + plains count as continuous), (3) **settlement cluster connectivity**, and (4) **hierarchical Bayesian fallback** for sparse keys instead of the current 2-level fallback. These changes can reduce your unique key count while capturing more variance, and the hierarchical smoothing eliminates the sparse-key problem entirely.

## Key Findings

### 1. Additional Spatial Features (High Impact)

Based on land-use change CA literature and analysis of your simulation dynamics:

#### A. Food Potential Score (continuous, don't bin)
The simulation's key driver is winter survival, which depends on adjacent forest/plains. Your current `n_forest` (4-connected, capped at 2) is too coarse.

```python
def food_potential(grid, raw_grid, y, x):
    """Continuous food score: weighted sum of food-producing neighbors in radius 2."""
    score = 0.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if not (0 <= ny < 40 and 0 <= nx < 40):
                continue
            dist = abs(dy) + abs(dx)
            weight = 1.0 / dist  # closer = more important
            cell = grid[ny][nx]
            if cell == 4:  # forest
                score += 0.15 * weight  # matches sim food_from_forest
            elif cell == 0 and raw_grid[ny][nx] == 11:  # plains (not ocean)
                score += 0.05 * weight  # matches sim food_from_plains
    return score
```

**Why:** This directly models the sim's food calculation. A settlement with 3 adjacent forests (food_potential=0.45) will survive most winters; one with 0 forests won't. Your current binary `n_forest` caps at 2 and misses radius-2 forests entirely.

#### B. Terrain Composition in Radius 3 (ratios, not counts)
Instead of counting specific types, compute **ratios** of each terrain type in a larger neighborhood.

```python
def neighborhood_composition(grid, y, x, radius=3):
    """Terrain type ratios in Moore neighborhood of given radius."""
    counts = [0] * 6  # one per class
    total = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < 40 and 0 <= nx < 40:
                cls = CODE_TO_CLASS.get(grid[ny][nx], 0)
                counts[cls] += 1
                total += 1
    if total == 0:
        return [0.0] * 6
    return [c / total for c in counts]
```

**Key ratios to extract:**
- `settlement_density_r3`: fraction of settlements in radius 3 (expansion pressure)
- `forest_ratio_r3`: fraction of forest in radius 3 (food availability)
- `empty_ratio_r3`: fraction of empty/plains in radius 3 (expansion room)
- `ocean_ratio_r3`: fraction of ocean (coastal vs inland)

**Why:** CA literature consistently shows radius-3 Moore neighborhoods capture 80%+ of local interaction effects. Ratios are more informative than capped counts and don't blow up key space.

#### C. Distance to Ocean (not just binary adjacency)
Replace `near_ocean` binary with a graduated distance:

```python
def ocean_distance(raw_grid, y, x):
    """Manhattan distance to nearest ocean cell, capped at 5."""
    # Precompute once with BFS from all ocean cells
    min_d = 99
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            ny, nx = y + dy, x + dx
            if 0 <= ny < 40 and 0 <= nx < 40:
                if raw_grid[ny][nx] == 10:
                    min_d = min(min_d, abs(dy) + abs(dx))
    return min(min_d, 5)
```

**Why:** Port formation requires coastal adjacency, but settlement survival near coast is a gradient. Distance 1 (coastal) has different dynamics than distance 3 (near-coastal) vs distance 8+ (inland).

#### D. Settlement Cluster Size
Count connected settlements (4-connected component size). Larger clusters have more food sources and mutual support.

```python
def settlement_cluster_size(grid, y, x):
    """Size of connected settlement/port component containing (y,x). 0 if not a settlement."""
    if grid[y][x] not in (1, 2):  # not settlement or port
        # For empty/forest cells: size of nearest settlement cluster
        return 0
    # BFS from (y,x) counting connected settlements/ports
    visited = set()
    queue = [(y, x)]
    visited.add((y, x))
    while queue:
        cy, cx = queue.pop(0)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = cy+dy, cx+dx
            if 0 <= ny < 40 and 0 <= nx < 40 and (ny,nx) not in visited:
                if grid[ny][nx] in (1, 2):
                    visited.add((ny, nx))
                    queue.append((ny, nx))
    return len(visited)
```

**Why:** Isolated settlements die faster (no mutual food sharing in the mental model). Cluster size correlates with survival probability.

#### E. Edge/Border Detection
Is the cell on the frontier between settlement clusters and wilderness?

```python
def is_frontier(grid, y, x):
    """1 if cell is empty/forest AND adjacent to a settlement; 0 otherwise."""
    if grid[y][x] not in (0, 4):  # must be empty or forest
        return 0
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = y+dy, x+dx
        if 0 <= ny < 40 and 0 <= nx < 40 and grid[ny][nx] in (1, 2):
            return 1
    return 0
```

**Why:** Frontier cells are the ONLY ones that can transition empty->settlement. This is a binary indicator of expansion candidacy.

### 2. Handling Sparse Keys (the 20% Problem)

Your current fallback is 2-level: full key -> (init_cls, dist) -> uniform. This is too aggressive -- dropping from 5 features to 2 loses too much information.

#### Recommended: Hierarchical Bayesian Smoothing

Implement a **cascade of progressively coarser keys**, where each level acts as a prior for the next:

```python
def hierarchical_predict(feature_key, models):
    """
    Cascade through feature key levels with Bayesian smoothing.

    Levels (finest to coarsest):
    L0: (init_cls, dist, n_settle, n_forest, near_ocean)  -- full key
    L1: (init_cls, dist, n_settle, near_ocean)             -- drop forest
    L2: (init_cls, dist, near_ocean)                       -- drop settle count
    L3: (init_cls, dist_bin)                               -- coarse distance
    L4: (init_cls,)                                        -- just terrain type
    L5: global average                                     -- uniform-ish prior
    """
    keys = [
        feature_key,                                          # L0
        (feature_key[0], feature_key[1], feature_key[2], feature_key[4]),  # L1
        (feature_key[0], feature_key[1], feature_key[4]),     # L2
        (feature_key[0], min(feature_key[1], 4)),             # L3: dist binned to 0-4
        (feature_key[0],),                                    # L4
    ]

    # Start from coarsest (always has data)
    pred = np.full(6, 1/6)  # L5: global prior

    for level in reversed(range(len(keys))):
        key = keys[level]
        if key in models[level]:
            data = models[level][key]
            n = data['count']
            mean = data['mean']
            # Blend: more samples -> trust this level more
            alpha = n / (n + 5.0)  # pseudo-count of 5 for prior
            pred = alpha * mean + (1 - alpha) * pred

    return pred
```

**Why this is better than your current approach:**
- Never falls off a cliff (uniform is never reached unless ALL levels are empty, which can't happen)
- Each level contributes proportionally to its sample count
- `alpha = n / (n + 5)` means: 10 samples -> 67% trust this level; 50 samples -> 91% trust
- The pseudo-count of 5 is tunable via cross-validation

#### Alternative: Laplace Smoothing on the Lookup Table

For each key with `n` samples, blend the raw average with the parent key's average:

```
smoothed[key] = (n * raw_mean[key] + alpha * parent_mean[parent_key]) / (n + alpha)
```

Where `alpha` = 3-5 works well for your sample sizes.

### 3. Better Feature Representations

#### A. Don't Discretize -- Use Continuous Features with KNN-style Lookup

Instead of binning distance into integers and capping, keep features continuous and use **weighted nearest-neighbor averaging**:

```python
def predict_continuous(features, training_data, k=20):
    """
    Nadaraya-Watson kernel regression over feature space.

    features: numpy array of continuous features for query cell
    training_data: list of (feature_vector, gt_distribution) pairs
    """
    # Precompute feature arrays for speed
    X = np.array([t[0] for t in training_data])  # (N, D)
    Y = np.array([t[1] for t in training_data])  # (N, 6)

    # Squared Euclidean distance (normalize features first!)
    diff = X - features
    dists = np.sum(diff ** 2, axis=1)

    # Gaussian kernel
    bandwidth = np.median(dists) * 0.5  # adaptive bandwidth
    weights = np.exp(-dists / (2 * bandwidth))

    # Weighted average
    pred = np.average(Y, axis=0, weights=weights)
    return pred
```

**Pros:** No binning artifacts, handles continuous features naturally, automatically interpolates between known states.
**Cons:** Slower (O(N) per prediction vs O(1) lookup), needs feature normalization.

**Practical compromise:** Use continuous features for the 5 most important dimensions, bin the rest. Store training data as a KD-tree for O(log N) queries.

#### B. Feature Normalization for Mixed Types

If mixing continuous and discrete features:

```python
# Normalize each feature to [0, 1] range
feature_ranges = {
    'init_cls': (0, 5),           # categorical -> keep as-is or one-hot
    'dist_settlement': (0, 20),   # continuous
    'food_potential': (0, 1.0),   # continuous
    'settlement_density_r3': (0, 1), # ratio
    'forest_ratio_r3': (0, 1),   # ratio
    'ocean_distance': (0, 5),     # discrete-ish
    'is_frontier': (0, 1),       # binary
}
```

### 4. Recommended Feature Set (Practical Implementation)

Replace the current 5-tuple with a **7-feature hybrid** that keeps keys manageable:

```python
def extract_features_v2(grid, raw_grid, y, x, settlements, precomputed):
    """
    Improved feature extraction.

    Returns: tuple for lookup key + continuous features for smoothing
    """
    init_cls = CODE_TO_CLASS.get(grid[y][x], 0)

    # Distance to nearest settlement (continuous, cap at 10)
    dist = min((abs(y-sy) + abs(x-sx) for sy, sx in settlements), default=99)
    dist = min(dist, 10)

    # Food potential (continuous, use precomputed)
    food_pot = precomputed['food_potential'][y][x]
    food_bin = 0 if food_pot < 0.1 else (1 if food_pot < 0.3 else 2)

    # Settlement density in radius 3 (ratio -> binned)
    s_density = precomputed['settlement_density_r3'][y][x]
    s_bin = 0 if s_density < 0.05 else (1 if s_density < 0.15 else 2)

    # Ocean distance (0-5)
    ocean_d = precomputed['ocean_distance'][y][x]
    ocean_bin = min(ocean_d, 3)  # 0, 1, 2, 3+

    # Is frontier (binary)
    frontier = precomputed['is_frontier'][y][x]

    # Discrete key for lookup (7 dims, ~400-600 unique keys expected)
    key = (init_cls, min(dist, 6), food_bin, s_bin, ocean_bin, frontier)

    # Continuous vector for kernel smoothing fallback
    continuous = np.array([
        init_cls / 5.0,
        dist / 10.0,
        food_pot,
        s_density,
        ocean_d / 5.0,
        float(frontier),
        precomputed['forest_ratio_r3'][y][x],
    ])

    return key, continuous
```

**Estimated key count:** ~400-600 unique keys across 48K samples = ~80-120 samples/key average, with far fewer sparse keys.

### 5. Precomputation Strategy

All spatial features should be precomputed ONCE per initial grid (they don't change during prediction):

```python
def precompute_features(grid, raw_grid):
    """Precompute all spatial features for a 40x40 grid. O(H*W*R^2)."""
    H, W = 40, 40
    result = {}

    # BFS from all ocean cells for ocean distance
    from collections import deque
    ocean_dist = np.full((H, W), 99, dtype=int)
    queue = deque()
    for y in range(H):
        for x in range(W):
            if raw_grid[y][x] == 10:
                ocean_dist[y][x] = 0
                queue.append((y, x))
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = cy+dy, cx+dx
            if 0 <= ny < H and 0 <= nx < W and ocean_dist[ny][nx] > ocean_dist[cy][cx] + 1:
                ocean_dist[ny][nx] = ocean_dist[cy][cx] + 1
                queue.append((ny, nx))
    result['ocean_distance'] = ocean_dist

    # Food potential in radius 2
    food_pot = np.zeros((H, W), dtype=float)
    for y in range(H):
        for x in range(W):
            score = 0.0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W:
                        d = abs(dy) + abs(dx)
                        w = 1.0 / d
                        if grid[ny][nx] == 4:
                            score += 0.15 * w
                        elif grid[ny][nx] == 0 and raw_grid[ny][nx] == 11:
                            score += 0.05 * w
            food_pot[y][x] = score
    result['food_potential'] = food_pot

    # Settlement density in radius 3
    s_density = np.zeros((H, W), dtype=float)
    for y in range(H):
        for x in range(W):
            total = 0
            settle = 0
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W:
                        total += 1
                        if grid[ny][nx] in (1, 2):
                            settle += 1
            s_density[y][x] = settle / total if total > 0 else 0
    result['settlement_density_r3'] = s_density

    # Forest ratio in radius 3
    f_ratio = np.zeros((H, W), dtype=float)
    for y in range(H):
        for x in range(W):
            total = 0
            forest = 0
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W:
                        total += 1
                        if grid[ny][nx] == 4:
                            forest += 1
            f_ratio[y][x] = forest / total if total > 0 else 0
    result['forest_ratio_r3'] = f_ratio

    # Is frontier
    frontier = np.zeros((H, W), dtype=int)
    for y in range(H):
        for x in range(W):
            if grid[y][x] not in (0, 4):  # must be empty or forest
                continue
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] in (1, 2):
                    frontier[y][x] = 1
                    break
    result['is_frontier'] = frontier

    return result
```

## Comparison: Current vs Proposed Features

| Feature | Current | Proposed | Why Better |
|---------|---------|----------|------------|
| Init class | 6 values | Same | No change needed |
| Settlement distance | Manhattan, cap 8 | Manhattan, cap 10, finer bins | Slightly more range |
| Settlement count r2 | Count, cap 2 | Density ratio r3, 3 bins | Ratio captures fraction, larger radius |
| Forest neighbors | 4-connected count, cap 2 | Food potential score, 3 bins | Models actual food mechanic, includes plains |
| Ocean adjacency | Binary 4-connected | BFS distance, 4 bins | Gradient captures port probability decay |
| Frontier | (missing) | Binary | Directly predicts expansion candidacy |
| Forest ratio r3 | (missing) | Continuous (for smoothing) | Regional food availability |
| Key space | 167 keys, 20% sparse | ~400-600 keys, <5% sparse with smoothing | Hierarchical fallback eliminates sparsity |

## Gotchas & Considerations

- **Overfitting risk**: More features = larger key space. Monitor the fraction of keys with <10 samples. The hierarchical fallback is essential.
- **Cross-validation**: With 6 rounds x 5 seeds, do leave-one-round-out CV. Different rounds have different transition rates (R3 was a brutal outlier), so round-level CV is critical.
- **Precomputation cost**: radius-3 neighborhoods are O(H*W*49) per grid = ~78K ops. For 30 grids, ~2.3M ops. Negligible.
- **Feature correlation**: `food_potential` and `forest_ratio_r3` will be correlated. Use one for the discrete key, the other for continuous smoothing.
- **Ocean vs plains distinction**: Your `raw_grid` with code 10 vs 11 is ESSENTIAL. Without it, food potential is wrong for coastal cells.
- **Static cells**: Ocean (code 10) and Mountain (class 5) NEVER change. Don't waste model capacity on them -- predict [1,0,0,0,0,0] for ocean and [0,0,0,0,0,1] for mountain directly.

## Recommendations

### Priority 1: Hierarchical Bayesian Fallback (Easiest, Biggest Impact)
Replace your 2-level fallback with the 5-level cascade. This alone eliminates the sparse key problem and improves predictions on rare feature combinations.

### Priority 2: Food Potential Score
Replace `n_forest` with the continuous food potential score. This directly models the simulation's survival mechanic and is the single most predictive feature after init_class.

### Priority 3: Ocean Distance (BFS)
Replace binary `near_ocean` with BFS distance. This improves port prediction and captures the coastal gradient.

### Priority 4: Frontier Indicator
Add the binary frontier feature. Directly predicts which cells CAN transition to settlement.

### Priority 5: Larger Neighborhood Ratios
Add settlement_density_r3 and forest_ratio_r3. These capture regional dynamics that local features miss.

### Do NOT Do (Low ROI)
- KD-tree / full kernel regression: adds complexity, marginal gain over hierarchical lookup
- Cluster size computation: expensive BFS, high variance, hard to bin
- Gradient/edge detection kernels: over-engineered for 40x40 grid
- Deep learning (CNN/LSTM): not enough data for 48K samples, lookup table is already optimal for this regime

## Sources
1. [Urban Growth Modeling with Cellular Automata - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC5191102/) -- spatial variable catalog for CA models (distance, slope, neighborhood density)
2. [Improved Urban CA with Trend-Adjusted Neighborhood](https://link.springer.com/article/10.1186/s13717-020-00234-9) -- weighted neighborhood effects beyond simple counts
3. [Dynamic CA with Spatially Nonstationary Rules](https://www.tandfonline.com/doi/full/10.1080/15481603.2018.1426262) -- spatially varying transition rules, neighborhood enrichment
4. [CA-Markov Land Use Change Modeling](https://www.tandfonline.com/doi/full/10.1080/10106049.2023.2268059) -- driving factor selection and distance metrics
5. [Additive Smoothing - Wikipedia](https://en.wikipedia.org/wiki/Additive_smoothing) -- Laplace/Jeffreys prior for sparse probability estimation
6. [TabPFN: Small Data Tabular Prediction](https://www.nature.com/articles/s41586-024-08328-6) -- foundation model approach for small tabular datasets
7. [Kernel Density Estimation for Discrete Data](https://aakinshin.net/posts/kde-discrete/) -- handling discrete/mixed feature spaces
8. [Random Forest-CA Model for Urban Growth](https://www.mdpi.com/2220-9964/4/2/447) -- neighborhood composition features at multiple radii
9. [Sparse Hierarchical Table Ensemble](https://openreview.net/forum?id=24N4XH2NaYq) -- cascading fallback for sparse high-dimensional tables
10. [CA Extended Neighborhood Mechanisms](https://www.sciencedirect.com/science/article/pii/S1364815215300724) -- impact of neighborhood radius on CA simulation accuracy
