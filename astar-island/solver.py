"""Astar Island solver v2 — observe, analyze, predict, submit.

Improvements over v1:
- Saves all observations to file for analysis
- Cross-seed aggregation (same initial state → similar outcome)
- Smarter query distribution (more queries on fewer seeds for better stats)
- Post-round analysis (fetches ground truth, measures accuracy)
"""

import requests
import numpy as np
import sys
import json
import os
from collections import defaultdict
from datetime import datetime
from scipy.ndimage import gaussian_filter

# Auth
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2MDFkN2QwMi0yZTViLTQxNjgtODZiZC02OGFlMjk0M2QzNDEiLCJlbWFpbCI6InN0aWFuNDJAZ21haWwuY29tIiwiZXhwIjoxNzc0MjAzOTQ0fQ.fK5N9Q-thmwwCTj1uYsGLJhtFGq-S0nA0XU6QhqjiU8"
BASE = "https://api.ainm.no"

session = requests.Session()
session.cookies.set("access_token", TOKEN)
session.headers["Origin"] = "https://app.ainm.no"

CODE_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# 3x3 grid stride 13
BASE_VIEWPORTS = [
    (0, 0), (13, 0), (25, 0),
    (0, 13), (13, 13), (25, 13),
    (0, 25), (13, 25), (25, 25),
]

ALPHA = 0.5  # Jeffreys prior
FLOOR = 0.01
K = 6


def get_active_round():
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round!")
        # Check for completed rounds to analyze
        completed = [r for r in rounds if r["status"] == "completed"]
        if completed:
            print(f"Found {len(completed)} completed round(s) — use --analyze <round_id>")
        sys.exit(1)
    print(f"Round {active['round_number']}: {active['id']}")
    print(f"Closes: {active['closes_at']}")
    return active


def get_round_detail(round_id):
    detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()
    print(f"Map: {detail['map_width']}x{detail['map_height']}, {len(detail.get('initial_states', []))} seeds")
    return detail


def initial_state_to_classes(grid):
    h, w = len(grid), len(grid[0])
    classes = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            classes[y, x] = CODE_TO_CLASS.get(grid[y][x], 0)
    return classes


def observe(round_id, seed_index, vx, vy):
    resp = session.post(f"{BASE}/astar-island/simulate", json={
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": vx, "viewport_y": vy,
        "viewport_w": 15, "viewport_h": 15,
    })
    if resp.status_code != 200:
        print(f"  FAIL: {resp.status_code} {resp.text[:200]}")
        return None
    data = resp.json()
    print(f"  Seed {seed_index} ({vx},{vy}): {data['queries_used']}/{data['queries_max']}")
    return data


def plan_queries(n_seeds, budget):
    """Plan query distribution across seeds.

    Strategy: heavier observation on seeds 0-1 for learning,
    lighter on 2-4 for coverage.
    """
    if budget >= n_seeds * 10:
        # Enough for 10 per seed (full coverage)
        # Give extras to seed 0 for deeper stats
        base = 9  # 3x3 grid
        remaining = budget - base * n_seeds
        queries = {}
        for seed in range(n_seeds):
            vps = list(BASE_VIEWPORTS)
            # Add bonus viewports with remaining budget
            if remaining > 0:
                vps.append((6, 6))
                remaining -= 1
            if remaining > 0 and seed == 0:
                # Extra queries on seed 0 for learning
                for extra in [(3, 3), (10, 10), (20, 3), (3, 20), (20, 20)]:
                    if remaining > 0:
                        vps.append(extra)
                        remaining -= 1
            queries[seed] = vps
        return queries
    else:
        # Limited budget — prioritize breadth
        per_seed = budget // n_seeds
        queries = {}
        for seed in range(n_seeds):
            queries[seed] = BASE_VIEWPORTS[:per_seed]
        return queries


def observe_all(round_id, n_seeds, budget):
    """Run queries and save raw observations."""
    query_plan = plan_queries(n_seeds, budget)
    raw_observations = []  # list of {seed, viewport, grid, settlements}
    observations = defaultdict(list)  # (seed, y, x) -> [class, ...]
    cross_seed = defaultdict(list)  # (y, x) -> [class, ...] across ALL seeds
    settlement_stats = defaultdict(list)  # (seed, y, x) -> [{pop, food, wealth, defense, alive}, ...]

    for seed in range(n_seeds):
        vps = query_plan[seed]
        print(f"Seed {seed}: {len(vps)} queries planned")
        for vx, vy in vps:
            result = observe(round_id, seed, vx, vy)
            if not result:
                continue
            # Save raw
            raw_observations.append({
                "seed": seed, "vx": vx, "vy": vy,
                "grid": result["grid"],
                "settlements": result.get("settlements", []),
                "viewport": result["viewport"],
                "queries_used": result["queries_used"],
            })
            # Parse grid observations
            grid = result["grid"]
            vp = result["viewport"]
            for dy in range(len(grid)):
                for dx in range(len(grid[0])):
                    abs_y = vp["y"] + dy
                    abs_x = vp["x"] + dx
                    cls = CODE_TO_CLASS.get(grid[dy][dx], 0)
                    observations[(seed, abs_y, abs_x)].append(cls)
                    cross_seed[(abs_y, abs_x)].append(cls)
            # Parse settlement metadata
            for s in result.get("settlements", []):
                settlement_stats[(seed, s["y"], s["x"])].append({
                    "population": s.get("population", 0),
                    "food": s.get("food", 0),
                    "wealth": s.get("wealth", 0),
                    "defense": s.get("defense", 0),
                    "alive": s.get("alive", True),
                    "has_port": s.get("has_port", False),
                })

    n_settlements = len(settlement_stats)
    if n_settlements > 0:
        # Summarize settlement health
        alive_count = sum(1 for stats in settlement_stats.values()
                         if any(s["alive"] for s in stats))
        dead_count = sum(1 for stats in settlement_stats.values()
                         if any(not s["alive"] for s in stats))
        avg_pop = np.mean([s["population"] for stats in settlement_stats.values()
                          for s in stats if s["alive"]])
        print(f"Settlement stats: {n_settlements} observed, {alive_count} alive, "
              f"{dead_count} with deaths, avg pop={avg_pop:.1f}")

    print(f"Total: {len(observations)} seed-cell pairs, {len(cross_seed)} unique cells observed")
    return observations, cross_seed, raw_observations, settlement_stats


def build_transition_matrix(detail, observations):
    """Estimate P(end_class | start_class) from all observations across all seeds.

    Pools data from all seeds since they share hidden simulation parameters.
    Returns K x K matrix where T[start][end] = probability.
    """
    n_seeds = len(detail["initial_states"])
    # Count transitions: T_counts[start_class][end_class]
    T_counts = np.zeros((K, K))

    for seed in range(n_seeds):
        initial_grid = detail["initial_states"][seed]["grid"]
        initial_classes = initial_state_to_classes(initial_grid)

        for (s, y, x), obs_list in observations.items():
            if s != seed:
                continue
            start_cls = initial_classes[y, x]
            for end_cls in obs_list:
                T_counts[start_cls, end_cls] += 1

    # Jeffreys smoothing per row
    T = np.zeros((K, K))
    for start_cls in range(K):
        row_total = T_counts[start_cls].sum()
        if row_total > 0:
            T[start_cls] = (T_counts[start_cls] + ALPHA) / (row_total + K * ALPHA)
        else:
            # No observations for this start class — assume stays same
            T[start_cls] = np.full(K, FLOOR)
            T[start_cls, start_cls] = 1.0 - (K - 1) * FLOOR

    # Print transition matrix
    print("\nTransition matrix P(end | start):")
    print(f"{'':>12s}", end="")
    for cls in range(K):
        print(f"{CLASS_NAMES[cls]:>12s}", end="")
    print()
    for start in range(K):
        print(f"{CLASS_NAMES[start]:>12s}", end="")
        for end in range(K):
            print(f"{T[start, end]:>12.3f}", end="")
        print(f"  (n={int(T_counts[start].sum())})")

    return T


def build_predictions(detail, observations, cross_seed, settlement_stats=None):
    """Build predictions using per-seed obs + cross-seed aggregation + transition matrix + settlement metadata."""
    if settlement_stats is None:
        settlement_stats = {}
    h = detail["map_height"]
    w = detail["map_width"]
    n_seeds = len(detail["initial_states"])
    predictions = {}

    # Build transition matrix from all observations
    T = build_transition_matrix(detail, observations)

    for seed in range(n_seeds):
        initial_grid = detail["initial_states"][seed]["grid"]
        initial_classes = initial_state_to_classes(initial_grid)
        pred = np.full((h, w, K), FLOOR)

        for y in range(h):
            for x in range(w):
                seed_obs = observations.get((seed, y, x), [])
                cross_obs = cross_seed.get((y, x), [])

                if len(seed_obs) > 0:
                    # Best: direct observation for this seed
                    counts = np.zeros(K)
                    for cls in seed_obs:
                        counts[cls] += 1
                    n = len(seed_obs)

                    # Adaptive alpha: if cross-seed observations agree,
                    # use lower alpha (sharper prediction)
                    if len(cross_obs) > 2:
                        cross_counts = np.zeros(K)
                        for cls in cross_obs:
                            cross_counts[cls] += 1
                        cross_entropy = -np.sum((cross_counts / len(cross_obs))
                                                * np.log((cross_counts + 1e-10) / len(cross_obs)))
                        # Low entropy = seeds agree = sharper alpha
                        alpha = max(0.1, ALPHA * (cross_entropy / np.log(K)))
                    else:
                        alpha = ALPHA

                    pred[y, x] = (counts + alpha) / (n + K * alpha)

                elif len(cross_obs) > 0:
                    # Good: observations from other seeds (same hidden params)
                    counts = np.zeros(K)
                    for cls in cross_obs:
                        counts[cls] += 1
                    n = len(cross_obs)
                    # Use slightly higher alpha for cross-seed (less confident)
                    pred[y, x] = (counts + ALPHA * 2) / (n + K * ALPHA * 2)

                else:
                    # Fallback: use transition matrix as prior
                    init_cls = initial_classes[y, x]
                    pred[y, x] = T[init_cls].copy()

        # Settlement metadata adjustment
        # Low food/population settlements are more likely to become ruins
        n_adjusted = 0
        for (s, y, x), stats_list in settlement_stats.items():
            if s != seed:
                continue
            if y >= h or x >= w:
                continue
            for stats in stats_list:
                if not stats["alive"]:
                    # Dead settlement → boost ruin probability
                    pred[y, x, 3] += 0.3  # ruin
                    pred[y, x, 1] -= 0.1  # less settlement
                    pred[y, x, 2] -= 0.1  # less port
                    n_adjusted += 1
                elif stats["food"] < 0.2 and stats["population"] < 1.0:
                    # Starving + small → likely to collapse
                    pred[y, x, 3] += 0.15  # boost ruin
                    pred[y, x, 1] -= 0.05  # less settlement
                    n_adjusted += 1
                elif stats["population"] > 3.0 and stats["food"] > 0.5:
                    # Thriving → likely to expand to neighbors
                    pred[y, x, 1] += 0.1  # more confident settlement
                    # Boost settlement probability for adjacent empty/forest cells
                    for dy2, dx2 in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy2, x + dx2
                        if 0 <= ny < h and 0 <= nx < w:
                            init_code_n = initial_grid[ny][nx]
                            if init_code_n in (11, 0, 4):  # plains, empty, forest
                                pred[ny, nx, 1] += 0.05  # small settlement boost
                    n_adjusted += 1
        if n_adjusted > 0:
            print(f"Seed {seed}: {n_adjusted} cells adjusted from settlement metadata")

        # Hard constraints
        n_constrained = 0
        for y in range(h):
            for x in range(w):
                init_code = initial_grid[y][x]

                # Ocean never changes
                if init_code == 10:
                    pred[y, x] = np.full(K, FLOOR)
                    pred[y, x, 0] = 1.0 - (K - 1) * FLOOR
                    n_constrained += 1

                # Mountain never changes
                elif init_code == 5:
                    pred[y, x] = np.full(K, FLOOR)
                    pred[y, x, 5] = 1.0 - (K - 1) * FLOOR
                    n_constrained += 1

                # Port only possible adjacent to ocean
                elif init_code != 10:
                    has_ocean_neighbor = False
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and initial_grid[ny][nx] == 10:
                            has_ocean_neighbor = True
                            break
                    if not has_ocean_neighbor:
                        # Redistribute port probability to other classes
                        port_prob = pred[y, x, 2]
                        pred[y, x, 2] = FLOOR
                        excess = port_prob - FLOOR
                        if excess > 0:
                            for cls in [0, 1, 3, 4, 5]:
                                pred[y, x, cls] += excess / 5

        print(f"Seed {seed}: {n_constrained} cells hard-constrained (ocean/mountain)")

        # Gaussian smoothing — smooth each class channel independently
        # Only smooth cells that were directly observed (not prior-only cells)
        # This spreads settlement/ruin probability to nearby cells
        sigma = 0.5  # very mild — preserve signal from sparse features

        # Build mask of cells we should NOT smooth (static + unobserved)
        static_mask = np.zeros((h, w), dtype=bool)
        for y in range(h):
            for x in range(w):
                if initial_grid[y][x] in (10, 5):  # ocean or mountain
                    static_mask[y, x] = True

        # Save static values, smooth, restore
        static_backup = pred[static_mask].copy()
        for cls in range(K):
            pred[:, :, cls] = gaussian_filter(pred[:, :, cls], sigma=sigma)
        pred[static_mask] = static_backup

        # Floor and renormalize
        pred = np.maximum(pred, FLOOR)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        predictions[seed] = pred

        # Stats
        n_changed = 0
        for y in range(h):
            for x in range(w):
                if np.argmax(pred[y, x]) != initial_classes[y, x]:
                    n_changed += 1
        print(f"Seed {seed}: {n_changed} cells predicted to change from initial")

    return predictions


def validate_predictions(detail, predictions):
    """Validate predictions before submit. Returns True if OK."""
    h = detail["map_height"]
    w = detail["map_width"]
    n_seeds = len(detail["initial_states"])
    all_ok = True

    for seed in range(n_seeds):
        if seed not in predictions:
            print(f"  WARN: Seed {seed} missing!")
            all_ok = False
            continue

        pred = predictions[seed]

        # Shape check
        if pred.shape != (h, w, K):
            print(f"  FAIL: Seed {seed} shape {pred.shape} != ({h},{w},{K})")
            all_ok = False
            continue

        # Sum to 1.0 check
        sums = pred.sum(axis=-1)
        bad_sums = np.abs(sums - 1.0) > 0.02
        if bad_sums.any():
            n_bad = bad_sums.sum()
            worst = np.max(np.abs(sums - 1.0))
            print(f"  FAIL: Seed {seed} has {n_bad} cells not summing to 1.0 (worst: {worst:.4f})")
            all_ok = False

        # Floor check
        has_zero = (pred <= 0).any()
        if has_zero:
            n_zero = (pred <= 0).sum()
            print(f"  FAIL: Seed {seed} has {n_zero} zero/negative probabilities!")
            all_ok = False

        min_prob = pred.min()
        if min_prob < FLOOR:
            print(f"  WARN: Seed {seed} min probability {min_prob:.4f} < floor {FLOOR}")

        # Static cell check — ocean and mountain should be ~1.0 for class 0/5
        initial_grid = detail["initial_states"][seed]["grid"]
        initial_classes = initial_state_to_classes(initial_grid)
        n_static_wrong = 0
        for y in range(h):
            for x in range(w):
                init_code = initial_grid[y][x]
                init_cls = initial_classes[y, x]
                pred_cls = np.argmax(pred[y, x])
                # Ocean and mountain never change
                if init_code == 10 and pred_cls != 0:
                    n_static_wrong += 1
                elif init_code == 5 and pred_cls != 5:
                    n_static_wrong += 1
        if n_static_wrong > 0:
            print(f"  WARN: Seed {seed} has {n_static_wrong} static cells (ocean/mountain) predicted wrong")

        # Change summary
        pred_argmax = np.argmax(pred, axis=-1)
        n_changed = np.sum(pred_argmax != initial_classes)
        n_total = h * w

        # Per-class counts
        class_counts = {}
        for cls in range(K):
            class_counts[CLASS_NAMES[cls]] = int(np.sum(pred_argmax == cls))

        print(f"  Seed {seed}: OK | {n_changed}/{n_total} cells changed | {class_counts}")

        # Confidence stats
        confidence = np.max(pred, axis=-1)
        print(f"    Confidence: min={confidence.min():.2f} mean={confidence.mean():.2f} max={confidence.max():.2f}")

    if all_ok:
        print("\n  All validations passed.")
    else:
        print("\n  VALIDATION FAILURES — fix before submitting!")
    return all_ok


def submit_all(round_id, predictions):
    for seed, pred in predictions.items():
        resp = session.post(f"{BASE}/astar-island/submit", json={
            "round_id": round_id,
            "seed_index": seed,
            "prediction": pred.tolist(),
        })
        status = "OK" if resp.status_code == 200 else f"FAIL {resp.status_code}"
        print(f"Seed {seed}: {status}")


def check_scores(round_id):
    resp = session.get(f"{BASE}/astar-island/my-rounds").json()
    for r in resp:
        if r["id"] == round_id:
            print(f"\nRound score: {r.get('round_score', 'pending')}")
            print(f"Seeds submitted: {r.get('seeds_submitted', 0)}")
            print(f"Seed scores: {r.get('seed_scores', 'pending')}")
            print(f"Rank: {r.get('rank', 'pending')}/{r.get('total_teams', '?')}")
            print(f"Queries: {r.get('queries_used', 0)}/{r.get('queries_max', 50)}")
            return r
    print("Round not found")
    return None


def save_observations(round_id, raw_observations):
    """Save raw observation data to file."""
    os.makedirs("astar-island/data", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"astar-island/data/obs_{round_id[:8]}_{ts}.json"
    with open(path, "w") as f:
        json.dump(raw_observations, f)
    print(f"Observations saved to {path}")
    return path


def analyze_round(round_id):
    """Post-round: fetch ground truth and compare with our predictions."""
    print(f"\n=== Analyzing round {round_id[:8]} ===")
    detail = get_round_detail(round_id)
    n_seeds = len(detail.get("initial_states", []))

    for seed in range(n_seeds):
        resp = session.get(f"{BASE}/astar-island/analysis/{round_id}/{seed}")
        if resp.status_code != 200:
            print(f"Seed {seed}: analysis not available ({resp.status_code})")
            continue
        data = resp.json()
        gt = np.array(data["ground_truth"])
        pred = np.array(data.get("prediction", []))
        score = data.get("score", "?")

        # Analyze where we were wrong
        if len(pred) > 0:
            gt_argmax = np.argmax(gt, axis=-1)
            pred_argmax = np.argmax(pred, axis=-1)
            mismatches = np.sum(gt_argmax != pred_argmax)
            total = gt_argmax.size

            # Per-class accuracy
            print(f"\nSeed {seed}: score={score}, mismatches={mismatches}/{total}")
            for cls in range(K):
                gt_mask = gt_argmax == cls
                if gt_mask.sum() == 0:
                    continue
                correct = np.sum(pred_argmax[gt_mask] == cls)
                print(f"  {CLASS_NAMES[cls]}: {correct}/{gt_mask.sum()} ({100*correct/gt_mask.sum():.0f}%)")

            # Entropy analysis — where did we lose most points?
            gt_entropy = -np.sum(gt * np.log(gt + 1e-10), axis=-1)
            kl = np.sum(gt * np.log((gt + 1e-10) / (pred + 1e-10)), axis=-1)
            weighted_kl = gt_entropy * kl

            # Top 10 worst cells
            flat_idx = np.argsort(weighted_kl.flatten())[-10:]
            print(f"  Top 10 worst cells (highest weighted KL):")
            for idx in reversed(flat_idx):
                y, x = divmod(idx, gt.shape[1])
                print(f"    ({x},{y}): gt={CLASS_NAMES[gt_argmax[y,x]]} pred={CLASS_NAMES[pred_argmax[y,x]]} "
                      f"entropy={gt_entropy[y,x]:.2f} kl={kl[y,x]:.2f}")

    # Save analysis
    os.makedirs("astar-island/data", exist_ok=True)
    path = f"astar-island/data/analysis_{round_id[:8]}.json"
    with open(path, "w") as f:
        json.dump({"round_id": round_id, "analyzed_at": datetime.now().isoformat()}, f)
    print(f"\nAnalysis saved to {path}")


def main():
    print("=== Astar Island Solver v2 ===\n")

    # Handle --analyze flag
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        round_id = sys.argv[2] if len(sys.argv) > 2 else None
        if not round_id:
            # Find most recent completed round
            rounds = session.get(f"{BASE}/astar-island/rounds").json()
            completed = [r for r in rounds if r["status"] in ("completed", "scoring")]
            if completed:
                round_id = completed[-1]["id"]
            else:
                print("No completed rounds found")
                sys.exit(1)
        analyze_round(round_id)
        return

    # Handle --scores flag
    if len(sys.argv) > 1 and sys.argv[1] == "--scores":
        rounds = session.get(f"{BASE}/astar-island/rounds").json()
        for r in rounds:
            print(f"Round {r['round_number']}: {r['status']}")
            check_scores(r["id"])
        return

    # Normal flow: observe, predict, submit
    active = get_active_round()
    round_id = active["id"]
    detail = get_round_detail(round_id)

    budget = session.get(f"{BASE}/astar-island/budget").json()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']} ({remaining} remaining)")

    if remaining == 0:
        print("Budget exhausted! Building from initial state only.")
        observations, cross_seed, raw, settlement_stats = {}, {}, [], {}
    else:
        n_seeds = len(detail["initial_states"])
        observations, cross_seed, raw, settlement_stats = observe_all(round_id, n_seeds, remaining)
        save_observations(round_id, raw)

    predictions = build_predictions(detail, observations, cross_seed, settlement_stats)

    # Validate before submit
    print("\n=== Validation ===")
    valid = validate_predictions(detail, predictions)

    if "--submit" in sys.argv:
        if valid:
            submit_all(round_id, predictions)
            check_scores(round_id)
        else:
            print("REFUSING to submit — fix validation errors first!")
    else:
        print("\nDry run — add --submit to actually submit.")
        print("Review the validation output above first!")

    print("\nDone!")


if __name__ == "__main__":
    main()
