"""Astar Island Solver v3 — 5-phase pipeline with ABC param estimation + Dirichlet priors.

Pipeline:
1. OBSERVE  — 2 deep seeds (repeat viewports for stochastic samples) + 3 shallow seeds
2. ESTIMATE — ABC rejection sampling to estimate simulation parameters from settlement metadata
3. CALIBRATE — softmax-weighted historical GT + Dirichlet priors
4. PREDICT  — MC simulation + Bayesian update from observations
5. VALIDATE & SUBMIT

Usage:
    py astar-island/solver_v3.py                    # dry run on active round
    py astar-island/solver_v3.py --submit            # submit to active round
    py astar-island/solver_v3.py --backtest 4        # test against round 4 GT
    py astar-island/solver_v3.py --backtest 2        # test against round 2 GT
"""

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K = 6
FLOOR = 0.001
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
CODE_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2MDFkN2QwMi0yZTViLTQxNjgtODZiZC02OGFlMjk0M2QzNDEiLCJlbWFpbCI6InN0aWFuNDJAZ21haWwuY29tIiwiZXhwIjoxNzc0MjAzOTQ0fQ.fK5N9Q-thmwwCTj1uYsGLJhtFGq-S0nA0XU6QhqjiU8"
BASE = "https://api.ainm.no"

# 3x3 full-coverage grid (stride 13, viewport 15x15 → overlap 2)
FULL_GRID = [(0, 0), (13, 0), (25, 0),
             (0, 13), (13, 13), (25, 13),
             (0, 25), (13, 25), (25, 25)]

# Historical rounds with pre-computed transition vectors and GT paths
HISTORICAL = [
    {"name": "R1", "vec": np.array([0.5, 57.2, 41.3, 1.2]),
     "gt": "astar-island/data/ground_truth_r1.json",
     "rd": "data/astar_round1.json"},
    {"name": "R2", "vec": np.array([19.1, 43.2, 38.0, 19.8]),
     "gt": "astar-island/data/ground_truth_r2.json",
     "rd": "data/astar_round2.json"},
    # R3 skipped — brutal outlier (1.8% survival, 0 expansion)
    {"name": "R4", "vec": np.array([7.8, 27.0, 43.8, 8.5]),
     "gt": "astar-island/data/ground_truth_r4.json",
     "rd": "data/astar_round4.json"},
    {"name": "R5", "vec": np.array([0.6, 29.3, 70.7, 2.8]),
     "gt": "astar-island/data/ground_truth_r5.json",
     "rd": "data/astar_round5.json"},
    {"name": "R6", "vec": np.array([3.6, 57.7, 42.3, 9.6]),
     "gt": "astar-island/data/ground_truth_r6.json",
     "rd": "data/astar_round6.json"},
    {"name": "R7", "vec": np.array([6.3, 60.5, 38.7, 14.9]),
     "gt": "astar-island/data/ground_truth_r7.json",
     "rd": "data/astar_round7.json"},
    {"name": "R8", "vec": np.array([0.0, 0.0, 100.0, 0.0]),
     "gt": "astar-island/data/ground_truth_r8.json",
     "rd": "data/astar_round8.json"},
    {"name": "R9", "vec": np.array([0.0, 1.9, 98.1, 0.0]),
     "gt": "astar-island/data/ground_truth_r9.json",
     "rd": "data/astar_round9.json"},
    {"name": "R10", "vec": np.array([0.0, 0.0, 100.0, 0.0]),
     "gt": "astar-island/data/ground_truth_r10.json",
     "rd": "data/astar_round10.json"},
    {"name": "R11", "vec": np.array([5.7, 95.2, 4.8, 16.6]),
     "gt": "astar-island/data/ground_truth_r11.json",
     "rd": "data/astar_round11.json"},
    {"name": "R12", "vec": np.array([10.5, 78.6, 19.5, 17.6]),
     "gt": "astar-island/data/ground_truth_r12.json",
     "rd": "data/astar_round12.json"},
    {"name": "R13", "vec": np.array([0.0, 0.8, 99.2, 0.0]),
     "gt": "astar-island/data/ground_truth_r13.json",
     "rd": "data/astar_round13.json"},
    {"name": "R14", "vec": np.array([13.6, 86.3, 12.3, 26.0]),
     "gt": "astar-island/data/ground_truth_r14.json",
     "rd": "data/astar_round14.json"},
    {"name": "R15", "vec": np.array([0.0, 10.0, 90.0, 0.0]),
     "gt": "astar-island/data/ground_truth_r15.json",
     "rd": "data/astar_round15.json"},
]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def make_session():
    s = requests.Session()
    s.cookies.set("access_token", TOKEN)
    s.headers["Origin"] = "https://app.ainm.no"
    return s


def get_active_round(session):
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round!")
        sys.exit(1)
    print(f"Round {active['round_number']}: {active['id']}")
    print(f"Closes: {active['closes_at']}")
    return active


def get_round_detail(session, round_id):
    return session.get(f"{BASE}/astar-island/rounds/{round_id}").json()


def observe_with_retry(session, round_id, seed_index, vx, vy):
    for attempt in range(3):
        resp = session.post(f"{BASE}/astar-island/simulate", json={
            "round_id": round_id, "seed_index": seed_index,
            "viewport_x": vx, "viewport_y": vy,
            "viewport_w": 15, "viewport_h": 15,
        })
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Seed {seed_index} ({vx},{vy}): {data['queries_used']}/{data['queries_max']}")
            return data
        elif resp.status_code == 429:
            print(f"  Rate limited, waiting 2s...")
            time.sleep(2)
        else:
            print(f"  FAIL: {resp.status_code} {resp.text[:200]}")
            return None
    print(f"  FAIL: 3 retries exhausted")
    return None


def initial_state_to_classes(grid):
    h, w = len(grid), len(grid[0])
    classes = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            classes[y, x] = CODE_TO_CLASS.get(grid[y][x], 0)
    return classes


# ---------------------------------------------------------------------------
# Feature extraction v2 — improved features
# ---------------------------------------------------------------------------
def compute_ocean_distance(ig):
    """BFS from all ocean cells to compute distance map."""
    from collections import deque
    H, W = 40, 40
    dist = np.full((H, W), 99, dtype=int)
    q = deque()
    for y in range(H):
        for x in range(W):
            if ig[y][x] == 10:  # ocean
                dist[y, x] = 0
                q.append((y, x))
    while q:
        cy, cx = q.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < H and 0 <= nx < W and dist[ny, nx] > dist[cy, cx] + 1:
                dist[ny, nx] = dist[cy, cx] + 1
                q.append((ny, nx))
    return dist


def extract_features(ig, y, x, settlements, ocean_dist=None):
    """Extract features for a cell: (init_cls, dist, food_bin, ocean_dist, frontier, density_bin)."""
    ic = CODE_TO_CLASS.get(ig[y][x], 0)

    # Distance to nearest settlement (capped at 8)
    d = min((abs(y - sy) + abs(x - sx) for sy, sx in settlements), default=99)
    d = min(d, 8)

    # Food potential in radius 2 (weighted by distance)
    food = 0.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < 40 and 0 <= nx < 40:
                dist_w = 1.0 / (abs(dy) + abs(dx))
                code = ig[ny][nx]
                if code == 4:
                    food += 0.15 * dist_w  # forest
                elif code == 11:
                    food += 0.05 * dist_w  # plains
    food_bin = 0 if food < 0.15 else (1 if food < 0.4 else 2)

    # Ocean distance (graded 0-3+)
    od = min(ocean_dist[y, x], 3) if ocean_dist is not None else (
        0 if any(0 <= y + dy < 40 and 0 <= x + dx < 40 and ig[y + dy][x + dx] == 10
                 for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]) else 3)

    # Frontier: empty/forest AND adjacent to settlement
    frontier = 0
    if ic in (0, 4):
        frontier = int(any(0 <= y + dy < 40 and 0 <= x + dx < 40 and ig[y + dy][x + dx] in (1, 2)
                           for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]))

    # Settlement density in radius 3
    n_settle_r3 = sum(1 for sy, sx in settlements if abs(y - sy) + abs(x - sx) <= 3)
    density_bin = 0 if n_settle_r3 == 0 else (1 if n_settle_r3 <= 3 else 2)

    return (ic, d, food_bin, od, frontier, density_bin)


# ---------------------------------------------------------------------------
# Phase 1: OBSERVE
# ---------------------------------------------------------------------------
def find_settlement_dense_viewports(init_grid, n_viewports=13):
    """Find viewports covering the most settlement-adjacent cells (for repeats)."""
    settlements = [(y, x) for y in range(40) for x in range(40) if init_grid[y][x] in (1, 2)]
    if not settlements:
        return list(FULL_GRID)[:n_viewports]

    scored = []
    for vx in range(0, 26, 2):
        for vy in range(0, 26, 2):
            n = sum(1 for sy, sx in settlements if vy <= sy < vy + 15 and vx <= sx < vx + 15)
            scored.append((n, vx, vy))
    scored.sort(reverse=True)

    # Greedy cover
    covered = set()
    selected = []
    for _, vx, vy in scored:
        new_s = {(sy, sx) for sy, sx in settlements if vy <= sy < vy + 15 and vx <= sx < vx + 15}
        if len(new_s - covered) > 0 or len(selected) < 3:
            selected.append((vx, vy))
            covered |= new_s
            if len(selected) >= n_viewports:
                break

    # Pad with full grid if needed
    while len(selected) < n_viewports:
        for pos in FULL_GRID:
            if pos not in selected:
                selected.append(pos)
                if len(selected) >= n_viewports:
                    break
    return selected


def phase_observe(session, round_id, detail, budget):
    """Phase 1: Observe with 5-deep strategy.

    All 5 seeds: 9 grid + 1 settlement-dense viewport repeat = 10 queries each
    Total: 10*5 = 50 queries

    All seeds with full grid coverage maximizes cross-seed breadth (8000 cells).
    Experiment showed +0.53 pts over 3-deep+2-shallow strategy.

    Returns:
        observations: dict (seed, y, x) -> [class_idx, ...]
        settlement_stats: dict (seed, y, x) -> [settlement_dict, ...]
        raw_obs: list of raw observation dicts
    """
    n_seeds = len(detail["initial_states"])
    observations = defaultdict(list)
    settlement_stats = defaultdict(list)
    raw_obs = []

    if budget <= 0:
        print("Budget exhausted — no observations.")
        return observations, settlement_stats, raw_obs

    # Build query plan: all 5 seeds get 9 grid + 1 repeat = 10 each
    deep_seeds = list(range(n_seeds))
    shallow_seeds = []

    deep_queries = {}
    for seed in deep_seeds:
        ig = detail["initial_states"][seed]["grid"]
        best_vp = find_settlement_dense_viewports(ig, n_viewports=1)[0]
        vps = list(FULL_GRID) + [best_vp]  # 9 + 1 = 10
        deep_queries[seed] = vps

    shallow_queries = {}

    total_planned = sum(len(v) for v in deep_queries.values()) + sum(len(v) for v in shallow_queries.values())
    print(f"\nQuery plan: {total_planned} total (budget: {budget})")
    for seed in deep_seeds:
        print(f"  Deep seed {seed}: {len(deep_queries[seed])} queries")
    for seed in shallow_seeds:
        print(f"  Shallow seed {seed}: {len(shallow_queries[seed])} queries")

    # Execute deep seeds first (most valuable)
    queries_used = 0
    for seed in deep_seeds:
        print(f"\nObserving deep seed {seed}...")
        for vx, vy in deep_queries[seed]:
            if queries_used >= budget:
                print("  Budget exhausted!")
                break
            result = observe_with_retry(session, round_id, seed, vx, vy)
            if not result:
                continue
            queries_used = result["queries_used"]
            _parse_observation(result, seed, observations, settlement_stats, raw_obs)

    for seed in shallow_seeds:
        print(f"\nObserving shallow seed {seed}...")
        for vx, vy in shallow_queries[seed]:
            if queries_used >= budget:
                print("  Budget exhausted!")
                break
            result = observe_with_retry(session, round_id, seed, vx, vy)
            if not result:
                continue
            queries_used = result["queries_used"]
            _parse_observation(result, seed, observations, settlement_stats, raw_obs)

    print(f"\nTotal: {queries_used} queries used, {len(observations)} seed-cell observations")
    return observations, settlement_stats, raw_obs


def _parse_observation(result, seed, observations, settlement_stats, raw_obs):
    """Parse one observation result into our data structures."""
    vp = result["viewport"]
    grid = result["grid"]
    raw_obs.append({
        "seed": seed, "vx": vp["x"], "vy": vp["y"],
        "grid": grid, "settlements": result.get("settlements", []),
    })

    for dy in range(len(grid)):
        for dx in range(len(grid[0])):
            ay, ax = vp["y"] + dy, vp["x"] + dx
            if ay >= 40 or ax >= 40:
                continue
            cls = CODE_TO_CLASS.get(grid[dy][dx], 0)
            observations[(seed, ay, ax)].append(cls)

    for s in result.get("settlements", []):
        settlement_stats[(seed, s["y"], s["x"])].append(s)


# ---------------------------------------------------------------------------
# Phase 2: ESTIMATE PARAMS (ABC rejection)
# ---------------------------------------------------------------------------
def compute_summary_stats(observations, settlement_stats, detail):
    """Compute summary statistics from observations for ABC."""
    n_seeds = len(detail["initial_states"])

    # Transition counts
    T = np.zeros((6, 6))
    for (seed, y, x), obs_list in observations.items():
        if seed >= n_seeds:
            continue
        ig = detail["initial_states"][seed]["grid"]
        start = CODE_TO_CLASS.get(ig[y][x], 0)
        for end_cls in obs_list:
            T[start, end_cls] += 1

    # Summary stats
    stats = {}
    # Survival rate
    s_total = T[1].sum() + T[2].sum()
    if s_total > 0:
        s_alive = T[1, 1] + T[1, 2] + T[2, 1] + T[2, 2]
        stats["survival_rate"] = s_alive / s_total
    else:
        stats["survival_rate"] = 0.5

    # Expansion rate (empty -> settlement)
    if T[0].sum() > 0:
        stats["expansion_rate"] = (T[0, 1] + T[0, 2]) / T[0].sum()
    else:
        stats["expansion_rate"] = 0.005

    # Ruin fraction
    if s_total > 0:
        stats["ruin_fraction"] = (T[1, 3] + T[2, 3]) / s_total
    else:
        stats["ruin_fraction"] = 0.02

    # Avg food from settlement metadata
    food_values = []
    pop_values = []
    alive_count = 0
    dead_count = 0
    for key, stats_list in settlement_stats.items():
        for s in stats_list:
            if s.get("alive", True):
                alive_count += 1
                food_values.append(s.get("food", 0.3))
                pop_values.append(s.get("population", 1.0))
            else:
                dead_count += 1

    stats["avg_food"] = np.mean(food_values) if food_values else 0.3
    stats["avg_pop"] = np.mean(pop_values) if pop_values else 1.0
    stats["alive_count"] = alive_count
    stats["dead_count"] = dead_count

    print(f"\nSummary stats:")
    print(f"  Survival rate: {stats['survival_rate']:.3f}")
    print(f"  Expansion rate: {stats['expansion_rate']:.4f}")
    print(f"  Ruin fraction: {stats['ruin_fraction']:.4f}")
    print(f"  Avg food: {stats['avg_food']:.2f}, Avg pop: {stats['avg_pop']:.2f}")
    print(f"  Settlements: {alive_count} alive, {dead_count} dead")

    return stats


def phase_estimate_params(observations, settlement_stats, detail):
    """Phase 2: ABC rejection sampling to estimate simulation parameters.

    Returns list of accepted SimParams (top 1% of 2000 samples).
    """
    from simulator import SimParams, build_simulator

    obs_stats = compute_summary_stats(observations, settlement_stats, detail)

    # Prior ranges for 4 key parameters
    param_ranges = {
        "winter_severity": (0.05, 0.25),
        "expansion_rate": (0.0005, 0.01),
        "conflict_intensity": (0.02, 0.2),
        "death_to_ruin_rate": (0.02, 0.3),
    }

    n_samples = 500
    n_mc_per = 3  # MC sims per sample for speed
    rng = np.random.default_rng(42)

    results = []  # (distance, params_dict)

    print(f"\nABC rejection: {n_samples} samples x {n_mc_per} MC sims...")
    t0 = time.perf_counter()

    for i in range(n_samples):
        # Sample params from uniform priors
        ws = rng.uniform(*param_ranges["winter_severity"])
        er = rng.uniform(*param_ranges["expansion_rate"])
        ci = rng.uniform(*param_ranges["conflict_intensity"])
        dtr = rng.uniform(*param_ranges["death_to_ruin_rate"])

        p = SimParams(
            winter_severity=ws,
            expansion_rate=er,
            conflict_intensity=ci,
            death_to_ruin_rate=dtr,
            death_to_empty_rate=1.0 - dtr,
        )

        # Run MC sims on seed 0
        sim = build_simulator(detail, seed_index=0, params=p)
        initial = sim.initial_grid.copy()

        survival_rates = []
        expansion_rates = []
        ruin_fracs = []

        for mc in range(n_mc_per):
            final = sim.run(n_years=50, seed=i * 100 + mc)

            # Survival
            init_settle = (initial == 1) | (initial == 2)
            final_settle = (final == 1) | (final == 2)
            n_init = init_settle.sum()
            if n_init > 0:
                survival_rates.append((init_settle & final_settle).sum() / n_init)

            # Expansion
            init_empty = (initial == 0) & ~sim.is_ocean
            n_empty = init_empty.sum()
            if n_empty > 0:
                expansion_rates.append((init_empty & final_settle).sum() / n_empty)

            # Ruin fraction
            if n_init > 0:
                ruin_fracs.append((init_settle & (final == 3)).sum() / n_init)

        sim_stats = {
            "survival_rate": np.mean(survival_rates) if survival_rates else 0.5,
            "expansion_rate": np.mean(expansion_rates) if expansion_rates else 0.005,
            "ruin_fraction": np.mean(ruin_fracs) if ruin_fracs else 0.02,
        }

        # Distance in summary stat space
        dist = (
            (sim_stats["survival_rate"] - obs_stats["survival_rate"]) ** 2 * 4 +
            (sim_stats["expansion_rate"] - obs_stats["expansion_rate"]) ** 2 * 100 +
            (sim_stats["ruin_fraction"] - obs_stats["ruin_fraction"]) ** 2 * 10
        )

        # Food proxy: winter severity correlates inversely with avg food
        food_proxy = max(0, 0.5 - ws * 2)
        dist += (food_proxy - obs_stats["avg_food"]) ** 2

        results.append((dist, {
            "winter_severity": ws,
            "expansion_rate": er,
            "conflict_intensity": ci,
            "death_to_ruin_rate": dtr,
        }))

    elapsed = time.perf_counter() - t0
    print(f"  ABC done in {elapsed:.1f}s ({n_samples * n_mc_per} sims)")

    # Accept top 1% (20 samples)
    results.sort(key=lambda x: x[0])
    n_accept = max(10, n_samples // 100)
    accepted = results[:n_accept]

    # Summarize accepted params
    print(f"\n  Accepted {n_accept} parameter sets:")
    for param_name in param_ranges:
        values = [a[1][param_name] for a in accepted]
        print(f"    {param_name}: {np.mean(values):.4f} ± {np.std(values):.4f} "
              f"(range {np.min(values):.4f}-{np.max(values):.4f})")

    # Return list of SimParams
    accepted_params = []
    for _, pd in accepted:
        p = SimParams(
            winter_severity=pd["winter_severity"],
            expansion_rate=pd["expansion_rate"],
            conflict_intensity=pd["conflict_intensity"],
            death_to_ruin_rate=pd["death_to_ruin_rate"],
            death_to_empty_rate=1.0 - pd["death_to_ruin_rate"],
        )
        accepted_params.append(p)

    return accepted_params


# ---------------------------------------------------------------------------
# Phase 3: CALIBRATE MODEL (Dirichlet + softmax)
# ---------------------------------------------------------------------------
def compute_transition_vector(observations, detail):
    """Compute transition vector [E->S%, S->S%, S->E%, F->S%] from observations."""
    n_seeds = len(detail["initial_states"])
    T = np.zeros((6, 6))
    for (seed, y, x), obs_list in observations.items():
        if seed >= n_seeds:
            continue
        ig = detail["initial_states"][seed]["grid"]
        start = CODE_TO_CLASS.get(ig[y][x], 0)
        for end_cls in obs_list:
            T[start, end_cls] += 1

    e2s = 100 * (T[0, 1] + T[0, 2]) / T[0].sum() if T[0].sum() > 0 else 0
    s2s = 100 * T[1, 1] / T[1].sum() if T[1].sum() > 0 else 50
    s2e = 100 * T[1, 0] / T[1].sum() if T[1].sum() > 0 else 40
    f2s = 100 * T[4, 1] / T[4].sum() if T[4].sum() > 0 else 1
    return np.array([e2s, s2s, s2e, f2s])


def softmax_weights(current_vec, historical, temperature):
    """Compute softmax weights for historical rounds based on transition vector distance."""
    distances = []
    names = []
    for h in historical:
        d = np.sqrt(np.sum((current_vec - h["vec"]) ** 2))
        distances.append(d)
        names.append(h["name"])

    distances = np.array(distances)
    logits = -distances / temperature
    logits -= logits.max()  # numerical stability
    weights = np.exp(logits)
    weights /= weights.sum()

    for name, d, w in zip(names, distances, weights):
        print(f"  {name}: dist={d:.1f}, weight={w:.3f}")

    return weights


def cv_temperature(observations, detail, hist=None):
    """Cross-validate temperature on historical rounds using LOO.

    Tests T in [5, 10, 20, 50], returns best T.
    Only works if we have 3+ historical rounds.
    """
    hist = hist or HISTORICAL
    if len(hist) < 3:
        print("  Too few historical rounds for CV, using T=20")
        return 20.0

    temperatures = [5, 10, 20, 50]
    best_T = 20.0
    best_loss = float("inf")

    for T in temperatures:
        total_loss = 0.0
        for i, held_out in enumerate(hist):
            # Use other rounds to predict held-out round's transition vector
            others = [h for j, h in enumerate(hist) if j != i]
            # Avg transition vec of others as "current"
            avg_vec = np.mean([h["vec"] for h in others], axis=0)
            weights = []
            for h in others:
                d = np.sqrt(np.sum((avg_vec - h["vec"]) ** 2))
                weights.append(np.exp(-d / T))
            weights = np.array(weights)
            weights /= weights.sum()

            # Predicted vec = weighted average of others
            pred_vec = np.zeros(4)
            for w, h in zip(weights, others):
                pred_vec += w * h["vec"]

            loss = np.sum((pred_vec - held_out["vec"]) ** 2)
            total_loss += loss

        if total_loss < best_loss:
            best_loss = total_loss
            best_T = T

    print(f"  CV temperature: T={best_T} (loss={best_loss:.1f})")
    return best_T


def build_dirichlet_model(detail, observations, concentration=5.0, historical_override=None):
    """Build Dirichlet prior from weighted historical ground truths.

    Returns dict: feature_key -> alpha array (Dirichlet parameters).
    """
    hist = historical_override or HISTORICAL

    # Get current round transition vector
    current_vec = compute_transition_vector(observations, detail)
    print(f"\nCurrent transition vec: E->S={current_vec[0]:.1f}% S->S={current_vec[1]:.1f}% "
          f"S->E={current_vec[2]:.1f}% F->S={current_vec[3]:.1f}%")

    # CV temperature
    T = cv_temperature(observations, detail, hist=hist)

    # Softmax weights
    print(f"\nHistorical weights (T={T}):")
    weights = softmax_weights(current_vec, hist, T)

    # Build weighted feature model from historical GTs
    weighted_sums = defaultdict(lambda: np.zeros(K))
    weight_totals = defaultdict(float)

    for h, w in zip(hist, weights):
        if w < 0.01:
            continue
        try:
            with open(h["gt"]) as f:
                gt_data = json.load(f)
            with open(h["rd"]) as f:
                rd = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  {h['name']}: skip ({e})")
            continue

        for seed_str, gt_array in gt_data.items():
            seed = int(seed_str)
            if seed >= len(rd["initial_states"]):
                continue
            ig = rd["initial_states"][seed]["grid"]
            gt = np.array(gt_array)
            od = compute_ocean_distance(ig)
            settlements = [(y, x) for y in range(40) for x in range(40) if ig[y][x] in (1, 2)]
            for y in range(40):
                for x in range(40):
                    key = extract_features(ig, y, x, settlements, od)
                    weighted_sums[key] += w * gt[y, x]
                    weight_totals[key] += w

    # Convert to Dirichlet alphas: alpha_k = mean_k * concentration
    dirichlet_model = {}
    for key in weighted_sums:
        if weight_totals[key] > 0:
            mean = weighted_sums[key] / weight_totals[key]
            # Ensure no zero means
            mean = np.maximum(mean, FLOOR)
            mean /= mean.sum()
            dirichlet_model[key] = mean * concentration

    # Simple fallback (init_cls, dist only)
    simple_sums = defaultdict(lambda: np.zeros(K))
    simple_totals = defaultdict(float)
    for key in weighted_sums:
        sk = (key[0], key[1])
        simple_sums[sk] += weighted_sums[key]
        simple_totals[sk] += weight_totals[key]

    simple_model = {}
    for key in simple_sums:
        if simple_totals[key] > 0:
            mean = simple_sums[key] / simple_totals[key]
            mean = np.maximum(mean, FLOOR)
            mean /= mean.sum()
            simple_model[key] = mean * concentration

    print(f"  Dirichlet model: {len(dirichlet_model)} keys, {len(simple_model)} simple keys")
    return dirichlet_model, simple_model


# ---------------------------------------------------------------------------
# Phase 4: PREDICT (MC + Bayesian update)
# ---------------------------------------------------------------------------
def _build_cross_seed_model(detail, observations):
    """Build feature_key -> distribution model from ALL observations across ALL seeds.

    This is the key insight: all seeds share the same hidden simulation parameters.
    Observations from seed 0 directly inform predictions for seed 2-4.
    """
    n_seeds = len(detail["initial_states"])
    cross_model = defaultdict(list)  # feature_key -> [one_hot, ...]
    cross_simple = defaultdict(list)  # (init_cls, dist) -> [one_hot, ...]

    # Precompute per seed
    seed_cache = {}
    for seed_idx in range(n_seeds):
        ig = detail["initial_states"][seed_idx]["grid"]
        settlements = [(sy, sx) for sy in range(40) for sx in range(40) if ig[sy][sx] in (1, 2)]
        od = compute_ocean_distance(ig)
        seed_cache[seed_idx] = (ig, settlements, od)

    for (seed, y, x), obs_list in observations.items():
        if seed >= n_seeds:
            continue
        ig, settlements, od = seed_cache[seed]
        key = extract_features(ig, y, x, settlements, od)

        for cls in obs_list:
            one_hot = np.zeros(K)
            one_hot[cls] = 1.0
            cross_model[key].append(one_hot)
            cross_simple[(key[0], key[1])].append(one_hot)

    # Return (mean_distribution, n_samples) per key
    avg_cross = {k: (np.mean(v, axis=0), len(v)) for k, v in cross_model.items() if len(v) >= 1}
    avg_simple = {k: (np.mean(v, axis=0), len(v)) for k, v in cross_simple.items() if len(v) >= 1}

    n_samples = sum(len(v) for v in cross_model.values())
    print(f"  Cross-seed model: {len(avg_cross)} keys from {n_samples} observations")
    return avg_cross, avg_simple


def phase_predict(detail, observations, settlement_stats,
                  accepted_params, dirichlet_model, simple_model):
    """Phase 4: Dirichlet pseudo-count combination.

    All prediction sources contribute pseudo-counts to a single Dirichlet posterior:
    - Jeffreys prior: 0.5 per class (always)
    - Historical prior: concentration * mean_distribution
    - Cross-seed model: n_samples * cross_scale * mean_distribution
    - Direct observations: real counts

    The formula naturally adapts: sparse cross-seed → more weight to historical;
    rich cross-seed → dominates. No hardcoded blend weights needed.
    """
    n_seeds = len(detail["initial_states"])
    predictions = {}

    CONCENTRATION = 3.0   # historical prior strength
    CROSS_SCALE = 2.0     # cross-seed weight multiplier (tuned: 1.0→2.0)

    # Build cross-seed model from ALL observations
    print("\n  Building cross-seed model...")
    cross_model, cross_simple = _build_cross_seed_model(detail, observations)

    # Count observations per cell per seed
    obs_counts = defaultdict(lambda: np.zeros(K))
    obs_n = defaultdict(int)
    for (seed, y, x), obs_list in observations.items():
        for cls in obs_list:
            obs_counts[(seed, y, x)][cls] += 1
            obs_n[(seed, y, x)] += 1

    for seed in range(n_seeds):
        ig = detail["initial_states"][seed]["grid"]
        od = compute_ocean_distance(ig)
        settlements = [(y, x) for y in range(40) for x in range(40) if ig[y][x] in (1, 2)]

        # --- Build predictions per cell ---
        pred = np.zeros((40, 40, K))

        for y in range(40):
            for x in range(40):
                key = extract_features(ig, y, x, settlements, od)

                # Start with Jeffreys prior
                alpha = np.ones(K) * 0.5

                # Add historical prior
                hist_alpha = dirichlet_model.get(key,
                             simple_model.get((key[0], key[1]), None))
                if hist_alpha is not None:
                    hist_mean = hist_alpha / hist_alpha.sum()
                    alpha += hist_mean * CONCENTRATION
                else:
                    alpha += np.ones(K) * (CONCENTRATION / K)

                # Add cross-seed (sample-count adaptive)
                cross_data = cross_model.get(key,
                             cross_simple.get((key[0], key[1]), None))
                if cross_data is not None:
                    c_mean, c_n = cross_data
                    alpha += c_mean * c_n * CROSS_SCALE

                # Add direct observations (real counts)
                n_obs = obs_n.get((seed, y, x), 0)
                if n_obs > 0:
                    alpha += obs_counts[(seed, y, x)]

                pred[y, x] = alpha / alpha.sum()

        # Per-class bias correction multipliers (tuned on 9 rounds LOO)
        # Compensates: Settlement/Forest underestimated, Ruin overestimated
        CLASS_MULT = np.array([1.05, 1.15, 1.05, 0.95, 1.10, 1.0])
        for y in range(40):
            for x in range(40):
                pred[y, x] *= CLASS_MULT
                pred[y, x] = np.maximum(pred[y, x], FLOOR)
                pred[y, x] /= pred[y, x].sum()

        predictions[seed] = pred

        # Stats
        init_cls = initial_state_to_classes(ig)
        n_changed = np.sum(np.argmax(pred, axis=-1) != init_cls)
        n_obs_cells = sum(1 for key in obs_n if key[0] == seed)
        confidence = np.max(pred, axis=-1)
        print(f"Seed {seed}: {n_changed} changed, {n_obs_cells} obs, "
              f"conf min={confidence.min():.2f} mean={confidence.mean():.2f}")

    return predictions


def _apply_settlement_adjustments(pred, settlement_stats, seed, ig):
    """Adjust predictions based on settlement metadata (food, pop, alive)."""
    n_adjusted = 0
    for (s, y, x), stats_list in settlement_stats.items():
        if s != seed or y >= 40 or x >= 40:
            continue
        for stats in stats_list:
            if not stats.get("alive", True):
                # Dead settlement → boost ruin/empty
                pred[y, x, 3] += 0.2
                pred[y, x, 0] += 0.1
                pred[y, x, 1] = max(pred[y, x, 1] - 0.15, FLOOR)
                pred[y, x, 2] = max(pred[y, x, 2] - 0.1, FLOOR)
                n_adjusted += 1
            elif stats.get("food", 0.5) < 0.15 and stats.get("population", 1.0) < 0.8:
                # Starving + small → likely collapse
                pred[y, x, 3] += 0.1
                pred[y, x, 0] += 0.05
                pred[y, x, 1] = max(pred[y, x, 1] - 0.05, FLOOR)
                n_adjusted += 1
            elif stats.get("population", 0) > 3.0 and stats.get("food", 0) > 0.5:
                # Thriving → boost settlement confidence + neighbors
                pred[y, x, 1] += 0.05
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < 40 and 0 <= nx < 40 and ig[ny][nx] in (11, 0, 4):
                        pred[ny, nx, 1] += 0.03
                n_adjusted += 1
    if n_adjusted > 0:
        print(f"  {n_adjusted} cells adjusted from settlement metadata")


# ---------------------------------------------------------------------------
# Phase 5: VALIDATE & SUBMIT
# ---------------------------------------------------------------------------
def apply_hard_constraints(pred, ig):
    """Apply hard constraints: ocean->Empty, mountain->Mountain, port only near ocean."""
    h, w = 40, 40
    for y in range(h):
        for x in range(w):
            code = ig[y][x]

            # Ocean never changes
            if code == 10:
                pred[y, x] = np.full(K, FLOOR)
                pred[y, x, 0] = 1.0 - (K - 1) * FLOOR

            # Mountain never changes
            elif code == 5:
                pred[y, x] = np.full(K, FLOOR)
                pred[y, x, 5] = 1.0 - (K - 1) * FLOOR

            # Port only adjacent to ocean
            else:
                has_ocean = False
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and ig[ny][nx] == 10:
                        has_ocean = True
                        break
                if not has_ocean:
                    excess = pred[y, x, 2] - FLOOR
                    pred[y, x, 2] = FLOOR
                    if excess > 0:
                        # Redistribute proportionally to existing probs
                        remaining = pred[y, x].copy()
                        remaining[2] = 0  # exclude port
                        remaining[5] = 0  # exclude mountain
                        total_r = remaining.sum()
                        if total_r > 0:
                            pred[y, x] += excess * (remaining / total_r)
                        else:
                            for cls in [0, 1, 3, 4]:
                                pred[y, x, cls] += excess / 4

    # Floor and renormalize
    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=-1, keepdims=True)
    return pred


def validate_predictions(detail, predictions):
    """Validate predictions before submit. Returns True if OK."""
    h, w = detail["map_height"], detail["map_width"]
    n_seeds = len(detail["initial_states"])
    all_ok = True

    for seed in range(n_seeds):
        if seed not in predictions:
            print(f"  WARN: Seed {seed} missing!")
            all_ok = False
            continue

        pred = predictions[seed]
        if pred.shape != (h, w, K):
            print(f"  FAIL: Seed {seed} shape {pred.shape} != ({h},{w},{K})")
            all_ok = False
            continue

        sums = pred.sum(axis=-1)
        bad_sums = np.abs(sums - 1.0) > 0.02
        if bad_sums.any():
            print(f"  FAIL: Seed {seed} has {bad_sums.sum()} cells not summing to 1.0")
            all_ok = False

        if (pred <= 0).any():
            print(f"  FAIL: Seed {seed} has {(pred <= 0).sum()} zero/negative probs!")
            all_ok = False

        ig = detail["initial_states"][seed]["grid"]
        init_cls = initial_state_to_classes(ig)
        pred_cls = np.argmax(pred, axis=-1)
        n_changed = np.sum(pred_cls != init_cls)
        confidence = np.max(pred, axis=-1)

        # Per-class counts
        cc = {CLASS_NAMES[c]: int(np.sum(pred_cls == c)) for c in range(K)}
        print(f"  Seed {seed}: OK | {n_changed} changed | conf min={confidence.min():.2f} "
              f"mean={confidence.mean():.2f} | {cc}")

    if all_ok:
        print("\n  All validations passed.")
    else:
        print("\n  VALIDATION FAILURES!")
    return all_ok


# ---------------------------------------------------------------------------
# Scoring (for backtest)
# ---------------------------------------------------------------------------
def compute_score(gt, pred):
    """Compute entropy-weighted KL score (0-100) matching the challenge formula."""
    gt = np.array(gt, dtype=np.float64)
    pred = np.array(pred, dtype=np.float64)

    # Clamp
    gt = np.clip(gt, 1e-10, 1.0)
    pred = np.clip(pred, 1e-10, 1.0)

    entropy = -np.sum(gt * np.log(gt), axis=-1)
    kl = np.sum(gt * np.log(gt / pred), axis=-1)

    mask = entropy > 0.01
    if mask.sum() == 0:
        return 100.0

    weighted_kl = np.sum(entropy[mask] * kl[mask]) / np.sum(entropy[mask])
    score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))
    return score


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------
def run_backtest(round_number):
    """Run backtest against a historical round's ground truth.

    Uses saved observations if available, otherwise uses only historical model.
    """
    gt_path = f"astar-island/data/ground_truth_r{round_number}.json"
    rd_path = f"data/astar_round{round_number}.json"

    if not os.path.exists(gt_path):
        print(f"No ground truth for round {round_number}!")
        sys.exit(1)
    if not os.path.exists(rd_path):
        print(f"No round data for round {round_number}!")
        sys.exit(1)

    with open(gt_path) as f:
        gt_data = json.load(f)
    with open(rd_path) as f:
        detail = json.load(f)

    print(f"=== Backtest Round {round_number} ===")
    print(f"Seeds: {len(detail['initial_states'])}")

    # Load observations if available
    import glob as globmod
    obs_files = sorted(globmod.glob(f"astar-island/data/obs_r{round_number}_*.json"))

    observations = defaultdict(list)
    settlement_stats = defaultdict(list)
    raw_obs = []

    if obs_files:
        latest_obs = obs_files[-1]
        print(f"Using observations: {latest_obs}")
        with open(latest_obs) as f:
            raw_obs = json.load(f)
        for obs in raw_obs:
            seed = obs["seed"]
            grid = obs["grid"]
            vy, vx = obs.get("vy", 0), obs.get("vx", 0)
            for dy in range(len(grid)):
                for dx in range(len(grid[0])):
                    ay, ax = vy + dy, vx + dx
                    if ay >= 40 or ax >= 40:
                        continue
                    cls = CODE_TO_CLASS.get(grid[dy][dx], 0)
                    observations[(seed, ay, ax)].append(cls)
            for s in obs.get("settlements", []):
                settlement_stats[(seed, s["y"], s["x"])].append(s)
        print(f"  {len(observations)} observation entries loaded")
    else:
        print("  No saved observations — using model-only predictions")

    # Exclude the backtest round from historical data
    historical_filtered = [h for h in HISTORICAL if h["name"] != f"R{round_number}"]

    # Phase 2: Estimate params (use observations if available)
    if observations:
        accepted_params = phase_estimate_params(observations, settlement_stats, detail)
    else:
        # Use default params
        from simulator import SimParams
        accepted_params = [SimParams()]

    # Phase 3: Calibrate model (using only OTHER historical rounds)
    print("\n=== Phase 3: Calibrate ===")
    dirichlet_model, simple_model = build_dirichlet_model(
        detail, observations, concentration=5.0,
        historical_override=historical_filtered)

    # Phase 4: Predict
    print("\n=== Phase 4: Predict ===")
    predictions = phase_predict(detail, observations, settlement_stats,
                                accepted_params, dirichlet_model, simple_model)

    # Phase 5: Validate + hard constraints
    print("\n=== Phase 5: Validate ===")
    n_seeds = len(detail["initial_states"])
    for seed in range(n_seeds):
        ig = detail["initial_states"][seed]["grid"]
        predictions[seed] = apply_hard_constraints(predictions[seed], ig)

    validate_predictions(detail, predictions)

    # Score against GT
    print("\n=== Backtest Scores ===")
    scores = []
    for seed_str, gt_array in gt_data.items():
        seed = int(seed_str)
        if seed not in predictions:
            print(f"  Seed {seed}: no prediction!")
            continue
        gt = np.array(gt_array)
        score = compute_score(gt, predictions[seed])
        scores.append(score)

        # Per-class analysis
        gt_argmax = np.argmax(gt, axis=-1)
        pred_argmax = np.argmax(predictions[seed], axis=-1)
        mismatches = np.sum(gt_argmax != pred_argmax)
        print(f"  Seed {seed}: score={score:.1f}, mismatches={mismatches}")

        for cls in range(K):
            gt_mask = gt_argmax == cls
            if gt_mask.sum() == 0:
                continue
            correct = np.sum(pred_argmax[gt_mask] == cls)
            print(f"    {CLASS_NAMES[cls]}: {correct}/{gt_mask.sum()} ({100*correct/gt_mask.sum():.0f}%)")

    avg = np.mean(scores) if scores else 0
    print(f"\n  Average score: {avg:.1f}")
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Astar Island Solver v3 ===\n")

    # Backtest mode
    if "--backtest" in sys.argv:
        idx = sys.argv.index("--backtest")
        if idx + 1 >= len(sys.argv):
            print("Usage: --backtest <round_number>")
            sys.exit(1)
        round_number = int(sys.argv[idx + 1])
        run_backtest(round_number)
        return

    # Live mode
    session = make_session()
    active = get_active_round(session)
    round_id = active["id"]
    detail = get_round_detail(session, round_id)
    n_seeds = len(detail["initial_states"])
    print(f"Map: {detail['map_width']}x{detail['map_height']}, {n_seeds} seeds")

    # Check budget
    budget_info = session.get(f"{BASE}/astar-island/budget").json()
    remaining = budget_info["queries_max"] - budget_info["queries_used"]
    print(f"Budget: {budget_info['queries_used']}/{budget_info['queries_max']} ({remaining} remaining)")

    # Save round data
    os.makedirs("astar-island/data", exist_ok=True)
    rd_path = f"data/astar_round{active['round_number']}.json"
    with open(rd_path, 'w') as f:
        json.dump(detail, f)

    # Phase 1: Observe
    print("\n=== Phase 1: OBSERVE ===")
    observations, settlement_stats, raw_obs = phase_observe(session, round_id, detail, remaining)

    # Save observations
    if raw_obs:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        obs_path = f"astar-island/data/obs_r{active['round_number']}_{ts}.json"
        with open(obs_path, 'w') as f:
            json.dump(raw_obs, f)
        print(f"Observations saved to {obs_path}")

    # If no new observations, load latest saved observations for this round
    if not observations:
        import glob as globmod
        obs_files = sorted(globmod.glob(f"astar-island/data/obs_r{active['round_number']}_*.json"))
        if obs_files:
            latest = obs_files[-1]
            print(f"Loading saved observations: {latest}")
            with open(latest) as f:
                raw_obs = json.load(f)
            for obs in raw_obs:
                seed = obs["seed"]
                grid = obs["grid"]
                vy, vx = obs.get("vy", 0), obs.get("vx", 0)
                for dy in range(len(grid)):
                    for dx in range(len(grid[0])):
                        ay, ax = vy + dy, vx + dx
                        if ay >= 40 or ax >= 40:
                            continue
                        cls = CODE_TO_CLASS.get(grid[dy][dx], 0)
                        observations[(seed, ay, ax)].append(cls)
                for s in obs.get("settlements", []):
                    settlement_stats[(seed, s["y"], s["x"])].append(s)
            print(f"  {len(observations)} observation entries loaded")

    # Phase 2: Estimate params
    print("\n=== Phase 2: ESTIMATE PARAMS ===")
    if observations:
        accepted_params = phase_estimate_params(observations, settlement_stats, detail)
    else:
        from simulator import SimParams
        accepted_params = [SimParams()]
        print("No observations — using default params")

    # Phase 3: Calibrate model
    print("\n=== Phase 3: CALIBRATE MODEL ===")
    dirichlet_model, simple_model = build_dirichlet_model(detail, observations, concentration=5.0)

    # Phase 4: Predict
    print("\n=== Phase 4: PREDICT ===")
    predictions = phase_predict(detail, observations, settlement_stats,
                                accepted_params, dirichlet_model, simple_model)

    # Phase 5: Validate & Submit
    print("\n=== Phase 5: VALIDATE & SUBMIT ===")
    for seed in range(n_seeds):
        ig = detail["initial_states"][seed]["grid"]
        predictions[seed] = apply_hard_constraints(predictions[seed], ig)

    valid = validate_predictions(detail, predictions)

    if "--submit" in sys.argv:
        if not valid:
            print("REFUSING to submit — fix validation errors first!")
            return
        print("\nSubmitting...")
        for seed, pred in predictions.items():
            resp = session.post(f"{BASE}/astar-island/submit", json={
                "round_id": round_id,
                "seed_index": seed,
                "prediction": pred.tolist(),
            })
            status = "OK" if resp.status_code == 200 else f"FAIL {resp.status_code}"
            print(f"  Seed {seed}: {status}")

        # Check scores
        my_rounds = session.get(f"{BASE}/astar-island/my-rounds").json()
        for r in my_rounds:
            if r["id"] == round_id:
                print(f"\nRound score: {r.get('round_score', 'pending')}")
                print(f"Rank: {r.get('rank', '?')}/{r.get('total_teams', '?')}")
    else:
        print("\nDry run — add --submit to submit.")

    print("\nDone!")


if __name__ == "__main__":
    main()
