"""Hill-climbing optimizer for nightmare using validated sim.

Runs hundreds of BotAdapter simulations with mutated configs
to find optimal parameter combination for a specific recon.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import os
import copy
import random
from dataclasses import asdict
from pathlib import Path

# Unbuffered output for background runs
os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bot.config import CoordinatorConfig
from Simulering.offline.bot_adapter import BotAdapter
from Simulering.offline.simulator import Simulator


logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)


def run_sim(recon_path: str, config: CoordinatorConfig) -> tuple[int, int, int, int]:
    """Run one simulation, return (score, items, orders, rounds_used)."""
    sim = Simulator.from_recon_file(recon_path)
    adapter = BotAdapter(suppress_logs=True, config=config)
    result = sim.run(adapter)
    if isinstance(result, dict):
        return result["score"], result.get("items_delivered", 0), result.get("orders_completed", 0), result.get("rounds_used", 500)
    return result.score, result.items_delivered, result.orders_completed, getattr(result, 'rounds_used', 500)


def mutate_nightmare(cfg: CoordinatorConfig, temperature: float = 1.0) -> CoordinatorConfig:
    """Mutate with nightmare-specific params (guidance, etc)."""
    p = cfg.mutate(temperature)
    r = random.random

    # Guidance params — these are the big levers for nightmare
    if r() < 0.3 * temperature:
        p.guidance_enabled = random.choice([True, False])
    if r() < 0.3 * temperature:
        p.guidance_alpha = random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    if r() < 0.3 * temperature:
        p.guidance_beta = random.choice([1.0, 2.0, 3.0, 5.0, 8.0])
    if r() < 0.3 * temperature:
        p.guidance_update_interval = random.choice([2, 3, 5, 8, 10, 15])
    if r() < 0.3 * temperature:
        p.guidance_decay = random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
    if r() < 0.2 * temperature:
        p.endgame_threshold = random.choice([40, 50, 60, 70, 80, 100])
    if r() < 0.2 * temperature:
        p.distance_noise = random.choice([0.0, 0.05, 0.1, 0.15, 0.2])
    if r() < 0.15 * temperature:
        p.noise_seed = random.randint(0, 999)
    if r() < 0.15 * temperature:
        p.shelf_randomness = random.choice([0.0, 0.1, 0.2, 0.3])

    return p


def hill_climb(
    recon_path: str,
    iterations: int = 200,
    restarts: int = 5,
    seed: int = 42,
) -> tuple[CoordinatorConfig, int]:
    """Hill-climbing with random restarts."""
    random.seed(seed)

    # Start from best known configs
    starts = [
        CoordinatorConfig.nightmare(),  # baseline 297
    ]
    # gui_off variant
    c = CoordinatorConfig.nightmare(); c.guidance_enabled = False
    starts.append(c)
    # alpha3 beta5 int8
    c = CoordinatorConfig.nightmare(); c.guidance_alpha = 3.0; c.guidance_beta = 5.0; c.guidance_update_interval = 8
    starts.append(c)

    best_config = starts[0]
    best_score, _, _, _ = run_sim(recon_path, best_config)
    print(f"Baseline: {best_score}")

    for restart in range(restarts):
        if restart < len(starts):
            current = copy.copy(starts[restart])
        else:
            current = copy.copy(starts[0])
            for _ in range(6):
                current = mutate_nightmare(current, 1.0)

        current_score, _, _, _ = run_sim(recon_path, current)
        print(f"\nRestart {restart}: start={current_score}")

        stale = 0
        for i in range(iterations):
            if stale > 30:
                candidate = copy.copy(current)
                for _ in range(4):
                    candidate = mutate_nightmare(candidate, 1.0)
                stale = 0
            else:
                candidate = mutate_nightmare(current, max(0.3, 1.0 - i/iterations))

            score, items, orders, rounds = run_sim(recon_path, candidate)

            if score > current_score:
                current = candidate
                current_score = score
                stale = 0
                marker = " ***" if score > best_score else " *"
                print(f"  [{restart}.{i:3d}] {score} (items={items}, orders={orders}, r={rounds}){marker}")

                if score > best_score:
                    best_score = score
                    best_config = copy.copy(current)
                    # Save immediately
                    Path("logs/best_nightmare_config.json").write_text(
                        json.dumps(best_config.to_dict(), indent=2))
            else:
                stale += 1

            if (i + 1) % 50 == 0:
                print(f"  [{restart}.{i:3d}] progress: current={current_score}, best={best_score}, stale={stale}")

    return best_config, best_score


def param_sweep(recon_path: str) -> None:
    """Sweep key parameters one at a time."""
    base = CoordinatorConfig.nightmare()
    base_score, _, _, _ = run_sim(recon_path, base)
    print(f"Baseline: {base_score}\n")

    sweeps = {
        "endgame_threshold": [30, 40, 50, 60, 70, 80],
        "max_route_items": [1, 2, 3],
        "pre_pick_max_inventory": [1, 2, 3],
        "guidance_enabled": [True, False],
        "guidance_alpha": [0.5, 1.0, 2.0, 3.0],
        "guidance_update_interval": [2, 3, 5, 8],
        "switching_penalty": [0.0, 1.0, 2.0, 3.0, 5.0],
        "order_completion_bonus": [3, 5, 7, 10],
        "gate_max_delay": [0, 1, 3, 5, 8],
        "stuck_transit_rounds": [4, 6, 8, 10],
        "pre_pick_rush_remaining": [2, 3, 4, 6],
        "pre_pick_rush_max_rounds": [15, 20, 25, 30],
        "distance_noise": [0.0, 0.1, 0.2, 0.5],
        "parallelism_penalty": [0.0, 1.0, 2.0, 3.0],
    }

    improvements = []

    for param, values in sweeps.items():
        print(f"--- {param} ---")
        for val in values:
            cfg = copy.copy(base)
            setattr(cfg, param, val)
            score, items, orders, _ = run_sim(recon_path, cfg)
            current_val = getattr(base, param)
            marker = " <-- current" if val == current_val else ""
            delta = score - base_score
            if delta > 0:
                marker += f" +{delta}"
                improvements.append((param, val, delta))
            elif delta < 0:
                marker += f" {delta}"
            print(f"  {param}={val}: {score}{marker}")
        print()

    if improvements:
        print("=== Improvements found ===")
        improvements.sort(key=lambda x: -x[2])
        for param, val, delta in improvements:
            print(f"  {param}={val}: +{delta}")


def noise_search(recon_path: str, n: int = 100) -> None:
    """Try different noise seeds to find lucky routing decisions."""
    base = CoordinatorConfig.nightmare()
    base_score, _, _, _ = run_sim(recon_path, base)
    print(f"Baseline (no noise): {base_score}\n")

    best_seed = -1
    best_score = base_score

    for noise in [0.05, 0.1, 0.2, 0.3]:
        print(f"--- noise={noise} ---")
        for seed in range(n):
            cfg = copy.copy(base)
            cfg.distance_noise = noise
            cfg.noise_seed = seed
            score, items, orders, _ = run_sim(recon_path, cfg)
            if score > best_score:
                best_score = score
                best_seed = seed
                print(f"  seed={seed}: {score} (items={items}, orders={orders}) ***")
        print(f"  Best at noise={noise}: seed={best_seed}, score={best_score}")

    print(f"\nOverall best: noise seed={best_seed}, score={best_score}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--mode", choices=["sweep", "climb", "noise"], default="sweep")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--restarts", type=int, default=5)
    args = parser.parse_args()

    t0 = time.time()

    if args.mode == "sweep":
        param_sweep(args.recon)
    elif args.mode == "climb":
        best_cfg, best_score = hill_climb(
            args.recon, iterations=args.iterations, restarts=args.restarts,
        )
        print(f"\n=== Best config (score={best_score}) ===")
        print(json.dumps(best_cfg.to_dict(), indent=2))
        # Save
        out = Path("logs/best_nightmare_config.json")
        out.write_text(json.dumps(best_cfg.to_dict(), indent=2))
        print(f"Saved to {out}")
    elif args.mode == "noise":
        noise_search(args.recon)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
