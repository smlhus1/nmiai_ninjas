"""ControlledBotAdapter: BotAdapter with overridden item assignments.

Uses the REAL Coordinator+PIBT for pathfinding and collision avoidance,
but injects a specific item-to-bot assignment plan instead of letting
the planner choose. This gives us the best of both worlds:
- Optimal GLOBAL assignments (which bot picks which item)
- Optimal LOCAL decisions (PIBT collision avoidance)

The assignment plan is a list of (order_idx, item_idx, bot_id, shelf_idx)
tuples that specify which bot picks which item from which shelf.
"""

from __future__ import annotations

import json
import logging
import sys
import os
import copy
import random
import time
from pathlib import Path
from collections import Counter
from dataclasses import dataclass

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import CoordinatorConfig
from Simulering.offline.simulator import Simulator
from Simulering.offline.bot_adapter import BotAdapter


def run_sim(recon: str, cfg: CoordinatorConfig) -> tuple[int, int]:
    """Run one sim, return (score, orders)."""
    sim = Simulator.from_recon_file(recon)
    a = BotAdapter(suppress_logs=True, config=cfg)
    r = sim.run(a)
    return r["score"], r["orders_completed"]


def exhaustive_search(recon: str, n_seeds: int = 5000):
    """Search over noise seeds + key params using BotAdapter.

    BotAdapter has PIBT — it handles pathfinding perfectly.
    noise_seed changes WHICH shelf each bot targets.
    We search thousands of seeds to find optimal routing.
    """
    base = CoordinatorConfig.from_dict(
        json.loads(Path("logs/best_nightmare_config.json").read_text()))

    best_score = 0
    best_cfg = None
    t0 = time.time()

    # Phase 1: Pure seed search across noise levels
    for noise in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        for seed in range(n_seeds // 8):
            cfg = copy.copy(base)
            cfg.noise_seed = seed
            cfg.distance_noise = noise

            score, orders = run_sim(recon, cfg)
            if score > best_score:
                best_score = score
                best_cfg = copy.copy(cfg)
                elapsed = time.time() - t0
                print(f"[{elapsed:.0f}s] noise={noise:.2f} seed={seed}: "
                      f"{score} (orders={orders}) ***", flush=True)

        elapsed = time.time() - t0
        print(f"[{elapsed:.0f}s] noise={noise:.2f} done, best={best_score}", flush=True)

    # Phase 2: Hill climb from best
    print(f"\nPhase 2: Hill climb from {best_score}", flush=True)
    current = copy.copy(best_cfg)
    current_score = best_score

    for i in range(500):
        candidate = copy.copy(current)
        # Small perturbation
        candidate.noise_seed = current.noise_seed + random.randint(-20, 20)
        candidate.distance_noise = max(0.01, current.distance_noise + random.uniform(-0.05, 0.05))

        if random.random() < 0.2:
            candidate.guidance_alpha = random.choice([0.5, 1.0, 2.0, 3.0, 4.0])
        if random.random() < 0.2:
            candidate.guidance_beta = random.choice([1.0, 2.0, 3.0, 5.0, 8.0])
        if random.random() < 0.15:
            candidate.guidance_update_interval = random.choice([2, 3, 5, 8])
        if random.random() < 0.15:
            candidate.shelf_randomness = random.uniform(0.0, 0.3)
        if random.random() < 0.1:
            candidate.switching_penalty = random.uniform(1.0, 6.0)

        score, orders = run_sim(recon, candidate)
        if score > current_score:
            current = candidate
            current_score = score
            if score > best_score:
                best_score = score
                best_cfg = copy.copy(current)
                elapsed = time.time() - t0
                print(f"[{elapsed:.0f}s] hill {i}: {score} ***", flush=True)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"[{elapsed:.0f}s] hill {i}: best={best_score}", flush=True)

    # Save
    Path("logs/best_nightmare_config.json").write_text(
        json.dumps(best_cfg.to_dict(), indent=2))

    elapsed = time.time() - t0
    print(f"\n=== BEST: {best_score} ({elapsed:.0f}s, {elapsed/60:.1f}m) ===", flush=True)
    return best_cfg, best_score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--seeds", type=int, default=5000)
    args = parser.parse_args()

    exhaustive_search(args.recon, n_seeds=args.seeds)
