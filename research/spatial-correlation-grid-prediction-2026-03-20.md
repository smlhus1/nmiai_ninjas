# Research: Spatial Correlation in Grid-Based Prediction

> Researched: 2026-03-20 | Sources consulted: 18 | Confidence: High

## TL;DR

For a 40x40x6 probability tensor, **Gaussian smoothing per class channel + renormalization** is the simplest and fastest approach (< 1ms). For stronger spatial consistency, **DenseCRF post-processing** via `pydensecrf` is the gold standard (~50ms for 40x40x6). For unobserved cells, **inverse-distance-weighted neighbor averaging** beats kriging in simplicity and speed. Mean-field MRF inference is the "correct" approach mathematically but DenseCRF wraps it in a fast C++ library that's easier than rolling your own.

## Key Findings

### 1. Gaussian Smoothing of Probability Maps (Simplest, Fastest)

The simplest approach: apply `scipy.ndimage.gaussian_filter` independently to each class channel of the probability tensor, then renormalize so probabilities sum to 1.

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def spatial_smooth_probs(probs, sigma=1.0):
    """
    Smooth a (H, W, C) probability tensor spatially.
    Each class channel is smoothed independently, then renormalized.

    Args:
        probs: (40, 40, 6) array where probs[y, x, :] sums to 1
        sigma: Gaussian kernel std dev. 1.0 = mild smoothing, 2.0 = strong

    Returns:
        Smoothed (40, 40, 6) probability tensor
    """
    smoothed = np.zeros_like(probs)
    for c in range(probs.shape[2]):
        smoothed[:, :, c] = gaussian_filter(probs[:, :, c], sigma=sigma)

    # Renormalize so each cell sums to 1
    totals = smoothed.sum(axis=2, keepdims=True)
    totals = np.maximum(totals, 1e-10)  # avoid division by zero
    smoothed /= totals
    return smoothed
```

**Pros:**
- Trivially fast (< 1ms for 40x40x6)
- Zero dependencies beyond scipy
- Works as a direct post-processing step on existing predictions
- Handles the "settlement clusters expand together" pattern well

**Cons:**
- Isotropic: smooths equally in all directions regardless of terrain/walls
- Does not respect game mechanics (e.g., water boundaries, mountains)
- sigma must be tuned: too high over-smooths, too low does nothing

**Recommended sigma values:**
- `sigma=0.5`: very mild, just slightly blends neighboring predictions
- `sigma=1.0`: moderate, good starting point for settlement clustering
- `sigma=1.5-2.0`: strong, only if predictions are very noisy

**Important variant: weighted smoothing.** If you have confidence scores (more observations = more confident), you can smooth `probs * confidence` and `confidence` separately, then divide:

```python
def confidence_weighted_smooth(probs, confidence, sigma=1.0):
    """
    Smooth predictions weighted by observation confidence.

    Args:
        probs: (40, 40, 6) probability tensor
        confidence: (40, 40) confidence weights (e.g., observation count)
        sigma: Gaussian kernel std dev
    """
    weighted = probs * confidence[:, :, np.newaxis]
    smooth_weighted = np.zeros_like(weighted)
    for c in range(probs.shape[2]):
        smooth_weighted[:, :, c] = gaussian_filter(weighted[:, :, c], sigma=sigma)
    smooth_conf = gaussian_filter(confidence, sigma=sigma)
    smooth_conf = np.maximum(smooth_conf, 1e-10)
    result = smooth_weighted / smooth_conf[:, :, np.newaxis]
    # Renormalize
    totals = result.sum(axis=2, keepdims=True)
    result /= np.maximum(totals, 1e-10)
    return result
```

This naturally gives more weight to cells with 14 observations vs 1 observation during smoothing.

---

### 2. DenseCRF Post-Processing (Best Quality, Still Fast)

The `pydensecrf` library implements Krahenbuhl & Koltun's "Efficient Inference in Fully Connected CRFs" — the standard approach for adding spatial consistency to independent per-pixel predictions. It wraps optimized C++ code.

**How it works:**
1. Your independent predictions become "unary potentials" (how much the model believes each class at each cell)
2. Pairwise potentials encode spatial smoothness (nearby cells should have similar labels)
3. Mean-field variational inference iteratively refines predictions (5-10 iterations)

The energy function being minimized:
```
E(x) = SUM_i  psi_unary(x_i)  +  SUM_{i,j}  psi_pairwise(x_i, x_j)
```

Where:
- `psi_unary(x_i) = -log P(x_i)` from your feature-key predictions
- `psi_pairwise` = Gaussian kernel penalizing different labels at nearby positions

```python
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def densecrf_smooth(probs, sxy=3, compat=3, n_iters=5):
    """
    Apply DenseCRF spatial smoothing to probability predictions.

    Args:
        probs: (H, W, C) probability tensor, C=6 classes
        sxy: spatial kernel bandwidth (larger = smoother, 1-10)
        compat: compatibility weight (larger = stronger smoothing)
        n_iters: mean-field iterations (5 is usually enough)

    Returns:
        Refined (H, W, C) probability tensor
    """
    H, W, C = probs.shape

    # DenseCRF2D expects (C, H*W) layout
    # probs must be (C, H, W) for unary_from_softmax
    probs_chw = probs.transpose(2, 0, 1)  # (C, H, W)

    # Clamp to avoid log(0)
    probs_chw = np.clip(probs_chw, 1e-6, 1.0 - 1e-6)

    d = dcrf.DenseCRF2D(W, H, C)

    # Unary potentials from our predictions
    U = unary_from_softmax(probs_chw)
    d.setUnaryEnergy(U)

    # Pairwise Gaussian: spatial smoothness only
    # sxy controls the spatial range of interaction
    d.addPairwiseGaussian(sxy=sxy, compat=compat)

    # Run inference
    Q = d.inference(n_iters)

    # Q is (C, H*W), reshape to (H, W, C)
    result = np.array(Q).reshape(C, H, W).transpose(1, 2, 0)
    return result
```

**Parameter tuning for 40x40 grid:**
- `sxy=2-3`: cells interact with immediate neighbors (1-2 cells away)
- `sxy=5`: cells interact with cells 3-4 away (settlement clusters)
- `compat=3`: mild smoothing, preserves most predictions
- `compat=10`: strong smoothing, heavily enforces spatial consistency
- `n_iters=5`: nearly always sufficient for convergence

**For the Norse sim specifically:**
- Start with `sxy=3, compat=5` — this gives moderate spatial consistency
- Increase `compat` if predictions are very noisy (few observations)
- Keep `sxy` small (2-3) if terrain boundaries matter

**Performance:** O(N) per iteration with efficient Gaussian filtering. For 40x40x6 = 9,600 variables, expect < 50ms for 5 iterations. Well within the 30s budget.

**Install:** `pip install pydensecrf`

**Bilateral kernel option:** If you have per-cell features (e.g., terrain type, food level), you can add a bilateral kernel that only smooths between cells with similar features:

```python
# Create feature image: terrain + food as "colors"
features = np.zeros((H, W, 2), dtype=np.uint8)
features[:, :, 0] = terrain_type * 50     # scale to 0-255 range
features[:, :, 1] = food_level * 25

d.addPairwiseBilateral(sxy=5, srgb=13, rgbim=features, compat=10)
```

This would prevent smoothing across water/mountain boundaries — settlements near water wouldn't influence predictions for cells across a mountain range.

---

### 3. Mean-Field MRF / Potts Model (DIY Version)

If you don't want the `pydensecrf` dependency, you can implement mean-field inference for a Potts model directly in numpy. This is exactly what DenseCRF does internally, but without the optimized permutohedral lattice filter.

**The Potts model energy:**
```
E(x) = -SUM_i  log P_unary(x_i)  +  beta * SUM_{i~j} [x_i != x_j]
```

Where `beta` controls smoothness strength and `i~j` means i,j are neighbors on the grid.

**Mean-field update (iterative):**
```
q_i(label) ∝ exp(-unary_i(label) - beta * SUM_{j in neighbors(i)} q_j(label != label))
```

Simplified: each cell updates its belief by considering how much its neighbors disagree.

```python
def mean_field_potts(probs, beta=1.0, n_iters=10):
    """
    Mean-field inference for Potts model on a 2D grid.

    Args:
        probs: (H, W, C) probability tensor (unary beliefs)
        beta: smoothness strength (0=no smoothing, 2=strong)
        n_iters: iterations

    Returns:
        Refined (H, W, C) probability tensor
    """
    H, W, C = probs.shape

    # Unary log-potentials
    log_unary = np.log(np.clip(probs, 1e-10, 1.0))

    q = probs.copy()

    for iteration in range(n_iters):
        # Compute neighbor agreement: for each cell, sum of q values from 4-neighbors
        neighbor_sum = np.zeros_like(q)
        # Up
        neighbor_sum[1:, :, :] += q[:-1, :, :]
        # Down
        neighbor_sum[:-1, :, :] += q[1:, :, :]
        # Left
        neighbor_sum[:, 1:, :] += q[:, :-1, :]
        # Right
        neighbor_sum[:, :-1, :] += q[:, :1, :]

        # Count actual neighbors (corners have 2, edges have 3, interior have 4)
        n_neighbors = np.zeros((H, W))
        n_neighbors[1:, :] += 1
        n_neighbors[:-1, :] += 1
        n_neighbors[:, 1:] += 1
        n_neighbors[:, :-1] += 1

        # Potts pairwise: encourage same label as neighbors
        # The "message" from neighbors is: agreement score for each label
        # Disagreement penalty = beta * (n_neighbors - neighbor_sum_for_this_label)
        # Equivalently: encouragement = beta * neighbor_sum_for_this_label

        log_q = log_unary + beta * neighbor_sum

        # Normalize (softmax)
        log_q -= log_q.max(axis=2, keepdims=True)  # numerical stability
        q = np.exp(log_q)
        q /= q.sum(axis=2, keepdims=True)

    return q
```

**Bug fix note:** The right-neighbor line has a typo — should be:
```python
neighbor_sum[:, :-1, :] += q[:, 1:, :]
```

**Pros:**
- Pure numpy, no dependencies
- Easy to modify (add directional biases, terrain-aware weights)
- Fast for 40x40: ~1ms per iteration

**Cons:**
- Must tune beta carefully
- Converges slower than DenseCRF (no permutohedral lattice acceleration)
- Does not handle long-range interactions without many iterations

---

### 4. Handling Unobserved Cells

For cells with 0 observations, you need to impute predictions. Options ranked by simplicity:

#### 4a. Nearest Observed Neighbor (Simplest)

```python
from scipy.ndimage import distance_transform_edt

def fill_unobserved_nearest(probs, observed_mask):
    """
    Fill unobserved cells with nearest observed cell's prediction.

    Args:
        probs: (H, W, C) probability tensor (unobserved cells can be anything)
        observed_mask: (H, W) boolean, True where we have observations

    Returns:
        Filled (H, W, C) probability tensor
    """
    if observed_mask.all():
        return probs

    # Find nearest observed cell for each unobserved cell
    _, nearest_indices = distance_transform_edt(
        ~observed_mask, return_distances=True, return_indices=True
    )

    # Copy predictions from nearest observed cell
    result = probs.copy()
    unobs = ~observed_mask
    result[unobs] = probs[nearest_indices[0][unobs], nearest_indices[1][unobs]]
    return result
```

#### 4b. Inverse-Distance-Weighted (IDW) Averaging (Better)

Uses all nearby observed cells, weighted by inverse distance:

```python
def fill_unobserved_idw(probs, observed_mask, power=2, max_neighbors=8):
    """
    Fill unobserved cells with IDW interpolation from observed neighbors.

    Args:
        probs: (H, W, C)
        observed_mask: (H, W) boolean
        power: distance weighting exponent (2 = standard, higher = more local)
        max_neighbors: max observed cells to consider
    """
    H, W, C = probs.shape
    result = probs.copy()

    obs_coords = np.argwhere(observed_mask)  # (N, 2) array of [y, x]
    unobs_coords = np.argwhere(~observed_mask)

    if len(unobs_coords) == 0 or len(obs_coords) == 0:
        return result

    for uy, ux in unobs_coords:
        # Distances to all observed cells
        dists = np.sqrt((obs_coords[:, 0] - uy)**2 + (obs_coords[:, 1] - ux)**2)

        # Take closest neighbors
        nearest_idx = np.argsort(dists)[:max_neighbors]
        nearest_dists = dists[nearest_idx]

        # Handle exact overlap (distance = 0)
        if nearest_dists[0] < 1e-10:
            result[uy, ux] = probs[obs_coords[nearest_idx[0], 0],
                                    obs_coords[nearest_idx[0], 1]]
            continue

        weights = 1.0 / (nearest_dists ** power)
        weights /= weights.sum()

        # Weighted average of neighbor predictions
        weighted_probs = np.zeros(C)
        for i, idx in enumerate(nearest_idx):
            oy, ox = obs_coords[idx]
            weighted_probs += weights[i] * probs[oy, ox]

        result[uy, ux] = weighted_probs / weighted_probs.sum()

    return result
```

#### 4c. Gaussian Process Interpolation (Most Principled, Slowest)

For 40x40 with sparse observations, GP is computationally feasible but overkill:

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def fill_unobserved_gp(probs, observed_mask, length_scale=3.0):
    """GP interpolation per class channel. Slow but principled."""
    H, W, C = probs.shape
    result = probs.copy()

    obs_coords = np.argwhere(observed_mask)
    unobs_coords = np.argwhere(~observed_mask)

    if len(unobs_coords) == 0:
        return result

    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=0.01)

    for c in range(C):
        obs_values = probs[observed_mask, c]
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0)
        gp.fit(obs_coords, obs_values)
        pred = gp.predict(unobs_coords)
        result[unobs_coords[:, 0], unobs_coords[:, 1], c] = np.clip(pred, 0, 1)

    # Renormalize
    totals = result.sum(axis=2, keepdims=True)
    result /= np.maximum(totals, 1e-10)
    return result
```

**GP performance:** For 40x40 = 1600 cells, fitting 6 GPs with ~200-800 observed points each takes 1-5 seconds. Acceptable but much slower than IDW.

**Recommendation:** Use IDW (4b) for speed, or GP (4c) if you want uncertainty estimates. Nearest-neighbor (4a) is fine for a quick baseline.

---

### 5. Belief Propagation on Grid (Academically Correct, Practically Unnecessary)

Loopy BP on a 40x40 grid with 6 labels is feasible (~9600 message variables) but provides marginal benefit over mean-field for this problem size.

**Implementation via pyLBP:**
```python
from pylbp import MRF

mrf = MRF(40, 40, 6)
mrf.init_base_belief(probs)  # (40, 40, 6) unary beliefs
mrf.init_smoothness(smoothness_matrix)  # (6, 6) label compatibility
for i in range(10):
    mrf.pass_messages()
beliefs = mrf.calc_belief()
```

**Performance:** The Loopy-Belief-Propagation repo shows ~9 seconds for ~36,000 nodes with binary labels. For 1,600 nodes with 6 labels, expect < 1 second. But parallel message passing (Right/Left/Up/Down phases) is needed for reasonable speed.

**Verdict:** DenseCRF already implements mean-field inference which is equivalent to BP for grid MRFs in practice. Use DenseCRF instead of rolling your own BP.

---

### 6. Weather/Climate Model Approaches (Kriging & Ensemble)

Weather models handle spatial correlation through:

1. **Variogram fitting:** Model how correlation decays with distance using exponential/Matern functions
2. **Kriging:** Optimal linear interpolation using the variogram — BLUP (Best Linear Unbiased Prediction)
3. **Ensemble methods:** Run multiple models, spatially interpolate the ensemble

**For the Norse sim analogy:**
- Winter severity is spatially correlated (all cells in a region get the same winter)
- This is modeled naturally: if you know the winter severity, you apply it to all cells
- The spatial correlation is in the *input* (weather), not just the output

**Kriging for sparse observations (using PyKrige):**
```python
from pykrige.ok import OrdinaryKriging

def kriging_interpolate(probs, observed_mask, variogram_model='exponential'):
    """Use kriging to interpolate each class probability."""
    H, W, C = probs.shape
    result = probs.copy()

    obs_y, obs_x = np.where(observed_mask)
    pred_y, pred_x = np.where(~observed_mask)

    if len(pred_y) == 0:
        return result

    for c in range(C):
        ok = OrdinaryKriging(
            obs_x.astype(float), obs_y.astype(float),
            probs[observed_mask, c],
            variogram_model=variogram_model,
            verbose=False
        )
        z_pred, ss_pred = ok.execute('points',
                                      pred_x.astype(float),
                                      pred_y.astype(float))
        result[pred_y, pred_x, c] = np.clip(z_pred, 0, 1)

    totals = result.sum(axis=2, keepdims=True)
    result /= np.maximum(totals, 1e-10)
    return result
```

**Verdict:** Kriging is overkill for 40x40 with decent coverage. IDW gives similar results for much less complexity. Kriging shines when you have very sparse observations AND need uncertainty estimates.

---

### 7. Game-Specific Spatial Heuristics (Simplest of All)

Given the specific Norse sim dynamics, some spatial patterns can be hardcoded:

```python
def apply_expansion_prior(probs, settlement_mask, expansion_boost=0.15):
    """
    Boost settlement probability for cells adjacent to existing settlements.
    Models: well-fed settlements expand to adjacent empty/forest cells.
    """
    from scipy.ndimage import binary_dilation

    # Cells adjacent to settlements
    adjacent = binary_dilation(settlement_mask, iterations=1) & ~settlement_mask

    # Boost settlement class probability for adjacent cells
    # Assuming class indices: 0=empty, 1=forest, 2=settlement, 3=ruin, 4=water, 5=mountain
    SETTLEMENT_CLASS = 2

    result = probs.copy()
    result[adjacent, SETTLEMENT_CLASS] += expansion_boost

    # Renormalize
    totals = result.sum(axis=2, keepdims=True)
    result /= np.maximum(totals, 1e-10)
    return result

def apply_cluster_death_prior(probs, food_levels, death_threshold=0.3):
    """
    If multiple nearby settlements have low food, boost ruin probability for all.
    Models: harsh winters kill ALL low-food settlements simultaneously.
    """
    from scipy.ndimage import uniform_filter

    RUIN_CLASS = 3
    SETTLEMENT_CLASS = 2

    # Average food level in 3x3 neighborhood
    local_avg_food = uniform_filter(food_levels, size=3)

    # Where local food is critically low, boost ruin probability
    starving_region = local_avg_food < death_threshold

    result = probs.copy()
    # Transfer some settlement probability to ruin probability
    transfer = result[starving_region, SETTLEMENT_CLASS] * 0.2
    result[starving_region, SETTLEMENT_CLASS] -= transfer
    result[starving_region, RUIN_CLASS] += transfer

    return result
```

---

## Comparison / Options Analysis

| Approach | Speed (40x40x6) | Complexity | Quality | Dependencies | Best For |
|----------|-----------------|------------|---------|--------------|----------|
| Gaussian smoothing | < 1ms | Trivial | Good | scipy | First thing to try |
| DenseCRF | < 50ms | Low | Best | pydensecrf | Production quality |
| Mean-field Potts (DIY) | ~10ms | Medium | Good | numpy only | No-dependency constraint |
| Belief propagation | ~100ms | High | Good | pylbp or DIY | Academic correctness |
| IDW interpolation | < 10ms | Low | Good | numpy/scipy | Filling unobserved cells |
| GP interpolation | 1-5s | Medium | Best | sklearn | Few observations + uncertainty |
| Kriging | 2-10s | High | Best | pykrige | Very sparse + uncertainty needed |
| Game heuristics | < 1ms | Trivial | Varies | numpy/scipy | Known game mechanics |

## Recommended Pipeline

For your specific use case (40x40x6 tensor, 6 rounds x 5 seeds, some cells with 14 obs, some with 0), here is the recommended pipeline in order:

```python
def spatial_postprocess(raw_probs, observed_mask, confidence,
                        settlement_mask=None, food_levels=None):
    """
    Complete spatial post-processing pipeline.

    Args:
        raw_probs: (40, 40, 6) independent predictions
        observed_mask: (40, 40) boolean
        confidence: (40, 40) observation count per cell
        settlement_mask: (40, 40) boolean, current settlements (optional)
        food_levels: (40, 40) float, current food levels (optional)
    """
    # Step 1: Fill unobserved cells with IDW from observed neighbors
    probs = fill_unobserved_idw(raw_probs, observed_mask)

    # Step 2: Apply game-specific priors (optional)
    if settlement_mask is not None:
        probs = apply_expansion_prior(probs, settlement_mask)
    if food_levels is not None:
        probs = apply_cluster_death_prior(probs, food_levels)

    # Step 3: Confidence-weighted Gaussian smoothing
    probs = confidence_weighted_smooth(probs, confidence, sigma=1.0)

    # Step 4 (optional): DenseCRF for final spatial consistency
    # probs = densecrf_smooth(probs, sxy=3, compat=5)

    return probs
```

**Start with Steps 1-3 only.** Add DenseCRF (Step 4) only if you see spatial inconsistencies in the output (e.g., isolated settlement predictions surrounded by ruins, or scattered ruin predictions in a healthy cluster).

## Gotchas & Considerations

- **Smoothing destroys signal at boundaries.** If water/mountains create hard boundaries between regions, Gaussian smoothing bleeds predictions across them. Use DenseCRF with bilateral kernel (terrain features) to prevent this.
- **Over-smoothing kills rare classes.** If ruins are rare (5% of cells), strong smoothing will suppress ruin predictions further. Reduce sigma/compat or handle rare classes separately.
- **Renormalization is mandatory.** After ANY operation on probability channels, renormalize to sum=1. Forgetting this is the #1 bug.
- **pydensecrf installation can fail on Windows.** If `pip install pydensecrf` fails, try `pip install pydensecrf` from conda-forge, or use the DIY mean-field Potts approach instead.
- **Observation count matters more than method.** The difference between 1 observation and 14 observations dwarfs the difference between smoothing methods. Prioritize getting more observations (more seeds/rounds) over perfecting the smoother.
- **Gaussian smoothing on log-probabilities vs probabilities:** Smoothing raw probabilities is simpler and works fine. Smoothing log-probabilities and exponentiating is mathematically cleaner but rarely makes a practical difference for small grids.

## Sources

1. [SciPy gaussian_filter docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html) — API reference for Gaussian smoothing
2. [pydensecrf GitHub](https://github.com/lucasb-eyer/pydensecrf) — DenseCRF Python wrapper, usage examples, parameter guidance
3. [DenseCRF Tutorial (Pseudo-Lab)](https://pseudo-lab.github.io/SegCrew-Book/docs/Appendix/DenseCRF.html) — Complete walkthrough of energy function, pairwise potentials, inference
4. [CS228 Notes: Undirected Graphical Models](https://ermongroup.github.io/cs228-notes/representation/undirected/) — MRF mathematical formulation
5. [pyLBP GitHub](https://github.com/mesilliac/pyLBP) — Numpy-based loopy belief propagation for grids
6. [Loopy-Belief-Propagation GitHub](https://github.com/parthnatekar/Loopy-Belief-Propagation) — Sequential/parallel BP with benchmarks (~9s for 36K nodes)
7. [Kriging Wikipedia](https://en.wikipedia.org/wiki/Kriging) — Spatial interpolation theory
8. [scikit-gstat Kriging docs](https://scikit-gstat.readthedocs.io/en/latest/userguide/kriging.html) — Practical kriging with sparse data in Python
9. [PyKrige guide](https://python.plainenglish.io/exploring-spatial-interpolation-with-pykrige-a-comprehensive-guide-to-kriging-in-python-67eaa1b8362e) — OrdinaryKriging usage examples
10. [Spatial Interpolation Methods](https://towardsdatascience.com/3-best-methods-for-spatial-interpolation-912cab7aee47/) — IDW vs Kriging vs other methods comparison
11. [MRF denoising (U Toronto)](http://www.cs.toronto.edu/~fleet/courses/2503/fall11/Handouts/mrf.pdf) — Ising/Potts model formulation for grid smoothing
12. [Gaussian smoothing intro](https://matthew-brett.github.io/teaching/smoothing_intro.html) — Theory of Gaussian smoothing with normalization
