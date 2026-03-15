"""Assignment evolver: evolve WHICH bot picks WHICH item from WHICH shelf.

Uses BotAdapter (with real PIBT) for movement — we only control assignments.
The key insight: BotAdapter makes good LOCAL decisions (pathfinding, collision).
We optimize GLOBAL decisions (which bot picks which item, shelf selection).

Genome = list of (item_type, shelf_choice_idx, zone_preference) per order.
BotAdapter uses these via noise_seed and shelf_randomness config params.

Actually simpler: we just evolve config params that deterministically
control routing decisions. noise_seed IS the genome.
But we can go deeper — control per-ORDER item-to-shelf assignments.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import os
import copy
import random
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import CoordinatorConfig
from Simulering.offline.simulator import Simulator
from Simulering.offline.bot_adapter import BotAdapter


def run_with_config(recon: str, cfg: CoordinatorConfig) -> tuple[int, int]:
    """Run sim with config, return (score, orders)."""
    sim = Simulator.from_recon_file(recon)
    a = BotAdapter(suppress_logs=True, config=cfg)
    r = sim.run(a)
    return r["score"], r["orders_completed"]


def massive_search(recon: str, base_cfg: CoordinatorConfig, n: int = 2000):
    """Brute-force search over noise seeds with the REAL sim.

    Each seed creates different routing decisions in the reactive bot.
    Some seeds happen to avoid congestion hotspots better than others.
    With 2000 seeds, we explore a huge space of routing variants.
    """
    best_score = 0
    best_seed = -1
    best_noise = 0.0
    best_cfg = None

    # Phase 1: Broad search — many seeds, fixed noise
    print("Phase 1: Broad seed search (1000 seeds × 2 noise levels)", flush=True)
    for noise in [0.1, 0.2]:
        for seed in range(1000):
            cfg = copy.copy(base_cfg)
            cfg.noise_seed = seed
            cfg.distance_noise = noise
            score, orders = run_with_config(recon, cfg)
            if score > best_score:
                best_score = score
                best_seed = seed
                best_noise = noise
                best_cfg = copy.copy(cfg)
                print(f"  seed={seed}, noise={noise}: {score} (orders={orders}) ***", flush=True)

            if (seed + 1) % 200 == 0:
                print(f"  Progress: {seed+1}/1000, noise={noise}, best={best_score}", flush=True)

    # Phase 2: Fine-tune around best seed
    print(f"\nPhase 2: Fine-tune around seed={best_seed}", flush=True)
    for delta in range(-50, 51):
        for noise in [best_noise - 0.05, best_noise, best_noise + 0.05]:
            if noise <= 0:
                continue
            cfg = copy.copy(base_cfg)
            cfg.noise_seed = best_seed + delta
            cfg.distance_noise = noise
            score, orders = run_with_config(recon, cfg)
            if score > best_score:
                best_score = score
                best_cfg = copy.copy(cfg)
                print(f"  seed={best_seed+delta}, noise={noise:.2f}: {score} ***", flush=True)

    # Phase 3: Also mutate guidance params around best
    print(f"\nPhase 3: Mutate guidance around best (score={best_score})", flush=True)
    for _ in range(500):
        cfg = copy.copy(best_cfg)
        cfg.noise_seed = best_cfg.noise_seed + random.randint(-10, 10)
        cfg.distance_noise = best_cfg.distance_noise + random.uniform(-0.05, 0.05)
        if random.random() < 0.3:
            cfg.guidance_alpha = random.choice([0.5, 1.0, 2.0, 3.0, 4.0])
        if random.random() < 0.3:
            cfg.guidance_beta = random.choice([1.0, 2.0, 3.0, 5.0, 8.0])
        if random.random() < 0.2:
            cfg.guidance_update_interval = random.choice([2, 3, 5, 8])
        if random.random() < 0.2:
            cfg.guidance_decay = random.choice([0.3, 0.5, 0.7, 0.9])

        score, orders = run_with_config(recon, cfg)
        if score > best_score:
            best_score = score
            best_cfg = copy.copy(cfg)
            print(f"  Guidance mutate: {score} ***", flush=True)

    return best_cfg, best_score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    base = CoordinatorConfig.nightmare()
    # Start from best known
    try:
        base = CoordinatorConfig.from_dict(
            json.loads(Path("logs/best_nightmare_config.json").read_text()))
    except Exception:
        pass

    t0 = time.time()
    best_cfg, best_score = massive_search(args.recon, base, n=2000)
    elapsed = time.time() - t0

    print(f"\n=== BEST: {best_score} ===", flush=True)
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    Path("logs/best_nightmare_config.json").write_text(
        json.dumps(best_cfg.to_dict(), indent=2))
    print(f"Config saved to logs/best_nightmare_config.json", flush=True)
