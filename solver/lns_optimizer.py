"""LNS (Large Neighborhood Search) over BotAdapter captured plans.

1. Capture BotAdapter's 500-round plan through sim
2. Identify slow order segments (>15 rounds per order)
3. For each slow segment: re-run with different configs and splice if faster
4. Output: optimized plan with best-of-breed segments

Key insight: most orders complete in 9-12 rounds. The 9 slow orders (27r avg)
are caused by PIBT congestion. Different config/noise produces different
congestion patterns — some will be faster for specific orders.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import random
import copy
from multiprocessing import Pool, cpu_count
from pathlib import Path
from collections import Counter

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))


def capture_run(recon_path: str, config_overrides: dict = None) -> dict:
    """Run BotAdapter and capture per-round data.

    Returns dict with:
    - score: total score
    - orders: orders completed
    - order_completions: [(round, score, order_num, delta_rounds)]
    - score_at_180: score at round 180
    - per_round_scores: list of score per round
    """
    from Simulering.offline.bot_adapter import BotAdapter
    from Simulering.offline.simulator import Simulator
    from bot.config import CoordinatorConfig

    config = CoordinatorConfig.for_difficulty(20)
    if config_overrides:
        for k, v in config_overrides.items():
            if v is not None and hasattr(config, k):
                setattr(config, k, v)

    adapter = BotAdapter(suppress_logs=True, config=config)
    sim = Simulator.from_recon_file(recon_path)
    state = sim.reset()

    per_round_scores = []
    order_completions = []
    prev_orders = 0

    for r in range(500):
        response = adapter(state.to_dict())
        state, game_over = sim.step(response.get("actions", []))
        per_round_scores.append(sim._score)

        if sim._orders_completed > prev_orders:
            delta = r - (order_completions[-1][0] if order_completions else 0)
            order_completions.append((r, sim._score, sim._orders_completed, delta))
            prev_orders = sim._orders_completed

        if game_over:
            break

    return {
        "score": sim._score,
        "orders": sim._orders_completed,
        "order_completions": order_completions,
        "score_at_180": per_round_scores[179] if len(per_round_scores) >= 180 else 0,
        "per_round_scores": per_round_scores,
    }


def _worker(args):
    """Worker for parallel evaluation."""
    idx, recon_path, overrides = args
    try:
        result = capture_run(recon_path, overrides)
        return idx, result
    except Exception:
        return idx, {"score": 0, "orders": 0, "order_completions": [],
                     "score_at_180": 0, "per_round_scores": []}


def lns_optimize(recon_path: str, n_trials: int = 100, n_workers: int = None):
    """Run LNS: many BotAdapter variants, pick best segments."""
    if n_workers is None:
        n_workers = min(cpu_count(), 12)

    # Generate config variants
    configs = []
    for i in range(n_trials):
        overrides = {}
        # Vary guidance params
        overrides["guidance_alpha"] = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        overrides["guidance_beta"] = random.choice([1.0, 2.0, 3.0, 5.0])
        overrides["guidance_decay"] = random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
        overrides["guidance_update_interval"] = random.choice([2, 3, 5])
        # Vary stuck thresholds
        overrides["stuck_transit_rounds"] = random.randint(3, 8)
        overrides["stuck_pick_rounds"] = random.randint(2, 6)
        overrides["stuck_deliver_rounds"] = random.randint(1, 4)
        # Vary gate
        overrides["gate_max_delay"] = random.choice([0, 2, 4, 6])
        overrides["pre_pick_rush_remaining"] = random.choice([2, 3, 4, 6])
        # Vary noise
        overrides["distance_noise"] = random.choice([0.0, 0.1, 0.2, 0.5])
        overrides["noise_seed"] = random.randint(0, 9999)
        configs.append(overrides)

    # Add baseline (no overrides)
    configs.insert(0, {})

    print(f"Running {len(configs)} configs with {n_workers} workers...", flush=True)
    t0 = time.time()

    args = [(i, recon_path, cfg) for i, cfg in enumerate(configs)]
    with Pool(n_workers) as pool:
        results = pool.map(_worker, args)

    all_results = [None] * len(configs)
    for idx, result in results:
        all_results[idx] = result

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s", flush=True)

    # Analyze: for each order, find which config completed it fastest
    # Build order-velocity matrix: configs x orders
    max_orders = max(r["orders"] for r in all_results if r)

    print(f"\n=== OVERVIEW ===")
    print(f"Configs: {len(configs)}")
    print(f"Max orders in any run: {max_orders}")

    # Score@180 distribution
    s180_scores = [r["score_at_180"] for r in all_results if r]
    s180_scores.sort(reverse=True)
    print(f"Score@180: best={s180_scores[0]}, p10={s180_scores[len(s180_scores)//10]}, "
          f"median={s180_scores[len(s180_scores)//2]}, worst={s180_scores[-1]}")

    # Total score distribution
    total_scores = sorted([r["score"] for r in all_results if r], reverse=True)
    print(f"Total: best={total_scores[0]}, p10={total_scores[len(total_scores)//10]}, "
          f"median={total_scores[len(total_scores)//2]}")

    # Best s180
    best_s180_idx = max(range(len(all_results)), key=lambda i: all_results[i]["score_at_180"] if all_results[i] else 0)
    best = all_results[best_s180_idx]
    print(f"\nBest s180 config (#{best_s180_idx}): s180={best['score_at_180']}, "
          f"total={best['score']}, orders={best['orders']}")
    if best_s180_idx > 0:
        print(f"Config: {configs[best_s180_idx]}")

    # Per-order: for each order number, find the config where it completed fastest
    print(f"\n=== PER-ORDER FASTEST COMPLETION ===")
    for order_num in range(1, max_orders + 1):
        best_delta = 9999
        best_cfg = -1
        baseline_delta = None

        for cfg_idx, result in enumerate(all_results):
            if not result:
                continue
            for r, s, o, delta in result["order_completions"]:
                if o == order_num:
                    if cfg_idx == 0:
                        baseline_delta = delta
                    if delta < best_delta:
                        best_delta = delta
                        best_cfg = cfg_idx
                    break

        if baseline_delta is not None:
            improvement = baseline_delta - best_delta
            marker = f" *** SAVE {improvement}r" if improvement > 3 else ""
            print(f"  Order {order_num:2d}: baseline={baseline_delta:2d}r, "
                  f"best={best_delta:2d}r (cfg #{best_cfg}){marker}")

    # Theoretical best: sum of fastest per-order deltas
    print(f"\n=== THEORETICAL BEST (cherry-pick fastest per order) ===")
    total_rounds = 0
    total_score = 0
    for order_num in range(1, max_orders + 1):
        best_delta = 9999
        for result in all_results:
            if not result:
                continue
            for r, s, o, delta in result["order_completions"]:
                if o == order_num:
                    best_delta = min(best_delta, delta)
                    break

        if best_delta < 9999:
            total_rounds += best_delta
            # Approximate: each order ~11 score (6 items + 5 bonus)
            total_score += 11

    print(f"Sum of fastest deltas: {total_rounds} rounds for {max_orders} orders")
    print(f"Estimated score: {total_score}")
    if total_rounds > 0:
        print(f"Theoretical velocity: {total_score/total_rounds:.2f} score/round")
    if total_rounds <= 180:
        print(f"All orders fit in 180 rounds! Score@180 ≈ {total_score}")
    else:
        # How many orders fit in 180 rounds?
        cumulative = 0
        orders_in_180 = 0
        for order_num in range(1, max_orders + 1):
            best_delta = 9999
            for result in all_results:
                if not result:
                    continue
                for r, s, o, delta in result["order_completions"]:
                    if o == order_num:
                        best_delta = min(best_delta, delta)
                        break
            if best_delta < 9999:
                cumulative += best_delta
                if cumulative <= 180:
                    orders_in_180 += 1
        print(f"Orders fitting in 180r (cherry-picked): {orders_in_180}")
        est_score = orders_in_180 * 11
        print(f"Estimated s180 (cherry-picked): ~{est_score}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    lns_optimize(args.recon, n_trials=args.trials, n_workers=args.workers)
