"""Astar Island R2 solver — analytical model + targeted observations.

Strategy:
- Seeds 0,1: full 3x3 grid (9 queries each = 18)
- Seeds 2,3,4: settlement-focused viewports (10-11 queries each = 32)
- Total: 50 queries
- Combine analytical R1 prior + observations via Bayesian update

DOES NOT SUBMIT without --submit flag.
"""

import requests
import numpy as np
import json
import sys
import os
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, 'astar-island')
from model import AnalyticalModel, extract_features, CODE_TO_CLASS, K, FLOOR

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2MDFkN2QwMi0yZTViLTQxNjgtODZiZC02OGFlMjk0M2QzNDEiLCJlbWFpbCI6InN0aWFuNDJAZ21haWwuY29tIiwiZXhwIjoxNzc0MjAzOTQ0fQ.fK5N9Q-thmwwCTj1uYsGLJhtFGq-S0nA0XU6QhqjiU8"
BASE = "https://api.ainm.no"

session = requests.Session()
session.cookies.set("access_token", TOKEN)
session.headers["Origin"] = "https://app.ainm.no"

# 3x3 full coverage grid
FULL_GRID = [(0,0),(13,0),(25,0),(0,13),(13,13),(25,13),(0,25),(13,25),(25,25)]


def get_active_round():
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round!")
        sys.exit(1)
    print(f"Round {active['round_number']}: {active['id']}")
    print(f"Closes: {active['closes_at']}")
    return active


def find_settlement_viewports(init_grid, n_viewports=11):
    """Find viewports that cover most settlement-adjacent cells."""
    settlements = [(y, x) for y in range(40) for x in range(40) if init_grid[y][x] in (1, 2)]

    scored = []
    for vx in range(0, 26, 2):  # step 2 for speed
        for vy in range(0, 26, 2):
            n_settle = sum(1 for sy, sx in settlements if vy <= sy < vy+15 and vx <= sx < vx+15)
            scored.append((n_settle, vx, vy))

    scored.sort(reverse=True)

    # Greedy: pick viewports that cover new settlements
    covered = set()
    selected = []
    for _, vx, vy in scored:
        new_settle = set()
        for sy, sx in settlements:
            if vy <= sy < vy+15 and vx <= sx < vx+15:
                new_settle.add((sy, sx))
        new_count = len(new_settle - covered)
        if new_count > 0 or len(selected) < 3:  # always take at least 3
            selected.append((vx, vy))
            covered |= new_settle
            if len(selected) >= n_viewports:
                break

    # Fill remaining with full grid positions not already covered
    while len(selected) < n_viewports:
        for pos in FULL_GRID:
            if pos not in selected:
                selected.append(pos)
                if len(selected) >= n_viewports:
                    break

    return selected


def observe(round_id, seed_index, vx, vy):
    import time
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
        elif resp.status_code == 429 and "Rate limit" in resp.text:
            print(f"  Rate limited, waiting 2s...")
            time.sleep(2)
            continue
        else:
            print(f"  FAIL: {resp.status_code} {resp.text[:200]}")
            return None
    print(f"  FAIL: 3 retries exhausted")
    return None


def main():
    print("=== Astar Island R2 Solver ===\n")

    # Load analytical model trained on R1
    model = AnalyticalModel()
    # Prefer latest combined model
    for mp in ['astar-island/data/model_r1r2r3.json', 'astar-island/data/model_r1r2.json', 'astar-island/data/model_r1.json']:
        if os.path.exists(mp):
            model_path = mp
            break
    if os.path.exists(model_path):
        model.load(model_path)
        print(f"Loaded R1 model: {len(model.model)} keys")
    else:
        print("No R1 model found! Run model.py first.")
        sys.exit(1)

    # Get active round
    active = get_active_round()
    round_id = active["id"]

    # Get round details
    detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()
    n_seeds = len(detail["initial_states"])
    print(f"Map: {detail['map_width']}x{detail['map_height']}, {n_seeds} seeds")

    # Save round data
    with open(f'data/astar_round{active["round_number"]}.json', 'w') as f:
        json.dump(detail, f)

    # Check budget
    budget = session.get(f"{BASE}/astar-island/budget").json()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"Budget: {budget['queries_used']}/{budget['queries_max']} ({remaining} remaining)")

    if remaining == 0:
        print("Budget exhausted! Using model-only predictions.")
        observations = {}
        settlement_stats = {}
    else:
        # Plan queries — ALWAYS full grid coverage (R3 lesson: partial coverage = bad scores)
        # 50 queries / 5 seeds = 10 per seed = 9 (3x3 grid) + 1 bonus
        query_plan = {}
        for seed in range(n_seeds):
            vps = list(FULL_GRID)  # 9 viewports, full coverage
            vps.append((6, 6))     # bonus center
            query_plan[seed] = vps

        print(f"\nQuery plan: {sum(len(v) for v in query_plan.values())} total")
        for seed, vps in query_plan.items():
            print(f"  Seed {seed}: {len(vps)} queries")

        # Execute queries
        observations = defaultdict(list)  # (seed, y, x) -> [class, ...]
        settlement_stats = defaultdict(list)
        raw_obs = []

        for seed in range(n_seeds):
            print(f"\nObserving seed {seed}...")
            for vx, vy in query_plan[seed]:
                result = observe(round_id, seed, vx, vy)
                if not result:
                    continue
                raw_obs.append({"seed": seed, "vx": vx, "vy": vy,
                                "grid": result["grid"], "settlements": result.get("settlements", [])})

                grid = result["grid"]
                vp = result["viewport"]
                for dy in range(len(grid)):
                    for dx in range(len(grid[0])):
                        ay, ax = vp["y"] + dy, vp["x"] + dx
                        cls = CODE_TO_CLASS.get(grid[dy][dx], 0)
                        observations[(seed, ay, ax)].append(cls)

                for s in result.get("settlements", []):
                    settlement_stats[(seed, s["y"], s["x"])].append(s)

        # Save observations
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        obs_path = f"astar-island/data/obs_r{active['round_number']}_{ts}.json"
        with open(obs_path, 'w') as f:
            json.dump(raw_obs, f)
        print(f"\nObservations saved to {obs_path}")

    # Build predictions
    print("\n=== Building predictions ===")
    predictions = {}
    for seed in range(n_seeds):
        ig = detail["initial_states"][seed]["grid"]

        # Convert observations to per-cell format for this seed
        seed_obs = {}
        for (s, y, x), obs_list in observations.items():
            if s == seed:
                seed_obs[(y, x)] = [int(c) for c in obs_list]

        pred = model.predict_grid(ig, observations=seed_obs, prior_weight=3.0)
        predictions[seed] = pred

        # Stats
        init_cls = np.array([[CODE_TO_CLASS.get(ig[y][x], 0) for x in range(40)] for y in range(40)])
        pred_argmax = np.argmax(pred, axis=-1)
        n_changed = np.sum(pred_argmax != init_cls)
        n_observed = len(seed_obs)
        confidence = np.max(pred, axis=-1)
        print(f"Seed {seed}: {n_changed} changed, {n_observed} cells observed, "
              f"confidence min={confidence.min():.2f} mean={confidence.mean():.2f}")

    # Validate
    print("\n=== Validation ===")
    from solver import validate_predictions, initial_state_to_classes
    validate_predictions(detail, predictions)

    # Submit or dry run
    if "--submit" in sys.argv:
        print("\n=== Submitting ===")
        for seed, pred in predictions.items():
            resp = session.post(f"{BASE}/astar-island/submit", json={
                "round_id": round_id,
                "seed_index": seed,
                "prediction": pred.tolist(),
            })
            status = "OK" if resp.status_code == 200 else f"FAIL {resp.status_code}"
            print(f"Seed {seed}: {status}")

        # Check scores
        my_rounds = session.get(f"{BASE}/astar-island/my-rounds").json()
        for r in my_rounds:
            if r["id"] == round_id:
                print(f"\nRound score: {r.get('round_score', 'pending')}")
                print(f"Rank: {r.get('rank', '?')}/{r.get('total_teams', '?')}")
    else:
        print("\nDry run — add --submit to actually submit.")

    print("\nDone!")


if __name__ == "__main__":
    main()
