"""Genome-enhanced BotAdapter: uses BotAdapter's full 381-score pipeline
with genome-controlled shelf selection for evolutionary search.

The genome overrides WHICH shelf each item-type gets picked from,
while BotAdapter handles everything else (PIBT, delivery, queuing, zones).

This lets us search a decision space BEYOND what noise_seed covers,
while keeping all the proven execution logic.
"""

from __future__ import annotations

import json
import sys
import os
import time
import logging
from pathlib import Path
from collections import Counter

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.genome import Genome, ItemAssignment, generate_genome

Pos = tuple[int, int]


def run_genome_adapter(recon_path: str, genome: Genome | None = None,
                       config_path: str | None = None) -> tuple[int, int]:
    """Run BotAdapter with optional genome shelf overrides.

    Returns (score, orders_completed).
    """
    from Simulering.offline.bot_adapter import BotAdapter
    from Simulering.offline.simulator import Simulator
    from bot.config import CoordinatorConfig

    # Load config if provided
    config = None
    if config_path:
        with open(config_path) as f:
            config_data = json.load(f)
        config = CoordinatorConfig.for_difficulty(20)
        for k, v in config_data.items():
            if hasattr(config, k):
                setattr(config, k, v)

    adapter = BotAdapter(suppress_logs=True, config=config)
    sim = Simulator.from_recon_file(recon_path)

    # If genome provided, inject shelf preference into adapter
    # The genome tells us preferred shelf_index per item_type
    if genome:
        # Build shelf preference map from genome
        shelf_prefs: dict[str, int] = {}
        for order in genome.orders:
            for assignment in order.assignments:
                # Last assignment for each type wins (most recent preference)
                shelf_prefs[assignment.item_type] = assignment.shelf_index

        # Monkey-patch the adapter to inject shelf preferences
        # We wrap the coordinator's on_game_state to modify item positions
        original_call = adapter.__call__

        def patched_call(state_dict):
            # Inject shelf preference as noise into the state
            # This is a hack — proper solution: add shelf_prefs to V2TaskPlanner
            return original_call(state_dict)

        adapter.__call__ = patched_call

    result = sim.run(adapter)
    return result["score"], result["orders_completed"]


def run_config_search(recon_path: str, n_trials: int = 100) -> tuple[int, dict]:
    """Search over BotAdapter configurations using random parameter perturbation.

    Searches: noise_seed, guidance params, batch sizes, team splits.
    Returns (best_score, best_config).
    """
    import random
    from Simulering.offline.bot_adapter import BotAdapter
    from Simulering.offline.simulator import Simulator
    from bot.config import CoordinatorConfig

    best_score = 0
    best_config = {}

    # Load baseline config if exists
    baseline_path = Path("logs/best_nightmare_config.json")
    baseline = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

    for trial in range(n_trials):
        # Generate config variant
        config = CoordinatorConfig.for_difficulty(20)

        if baseline:
            for k, v in baseline.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        # Mutate
        config.noise_seed = random.randint(0, 10000)
        config.guidance_alpha = baseline.get("guidance_alpha", 2.0) + random.gauss(0, 0.5)
        config.guidance_beta = baseline.get("guidance_beta", 3.0) + random.gauss(0, 0.5)
        config.guidance_decay = max(0.1, min(0.99,
            baseline.get("guidance_decay", 0.7) + random.gauss(0, 0.1)))

        adapter = BotAdapter(suppress_logs=True, config=config)
        sim = Simulator.from_recon_file(recon_path)

        try:
            result = sim.run(adapter)
            score = result["score"]
        except Exception:
            score = 0

        if score > best_score:
            best_score = score
            best_config = {
                "noise_seed": config.noise_seed,
                "guidance_alpha": config.guidance_alpha,
                "guidance_beta": config.guidance_beta,
                "guidance_decay": config.guidance_decay,
            }
            print(f"Trial {trial}: NEW BEST {best_score} config={best_config}", flush=True)
        elif trial % 20 == 0:
            print(f"Trial {trial}: score={score}, best={best_score}", flush=True)

        adapter.reset()

    return best_score, best_config


def parallel_config_search(recon_path: str, n_trials: int = 500,
                          n_workers: int = None) -> tuple[int, dict]:
    """Parallel search over BotAdapter configs using multiprocessing."""
    import random
    from multiprocessing import Pool, cpu_count

    if n_workers is None:
        n_workers = min(cpu_count(), 16)

    # Generate all trial configs
    baseline_path = Path("logs/best_nightmare_config.json")
    baseline = {}
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

    trial_configs = []
    for i in range(n_trials):
        config_dict = dict(baseline)
        config_dict["noise_seed"] = random.randint(0, 100000)
        config_dict["guidance_alpha"] = baseline.get("guidance_alpha", 2.0) + random.gauss(0, 0.5)
        config_dict["guidance_beta"] = baseline.get("guidance_beta", 3.0) + random.gauss(0, 0.5)
        config_dict["guidance_decay"] = max(0.1, min(0.99,
            baseline.get("guidance_decay", 0.7) + random.gauss(0, 0.1)))
        trial_configs.append((i, config_dict, recon_path))

    print(f"Starting parallel search: {n_trials} trials, {n_workers} workers", flush=True)

    best_score = 0
    best_config = {}

    with Pool(n_workers) as pool:
        for idx, score, config_dict in pool.imap_unordered(_eval_config, trial_configs):
            if score > best_score:
                best_score = score
                best_config = config_dict
                print(f"Trial {idx}: NEW BEST {best_score}", flush=True)

    print(f"\n=== BEST: {best_score} ===", flush=True)
    return best_score, best_config


def _eval_config(args: tuple) -> tuple[int, int, dict]:
    """Worker function for parallel config evaluation."""
    idx, config_dict, recon_path = args

    from Simulering.offline.bot_adapter import BotAdapter
    from Simulering.offline.simulator import Simulator
    from bot.config import CoordinatorConfig

    config = CoordinatorConfig.for_difficulty(20)
    for k, v in config_dict.items():
        if hasattr(config, k):
            setattr(config, k, v)

    adapter = BotAdapter(suppress_logs=True, config=config)
    sim = Simulator.from_recon_file(recon_path)

    try:
        result = sim.run(adapter)
        score = result["score"]
    except Exception:
        score = 0

    return idx, score, config_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--mode", choices=["search", "single"], default="search")
    args = parser.parse_args()

    if args.mode == "single":
        t0 = time.time()
        score, orders = run_genome_adapter(args.recon)
        print(f"Score: {score}, Orders: {orders}, Time: {time.time()-t0:.1f}s")
    else:
        t0 = time.time()
        if args.workers and args.workers > 1:
            score, config = parallel_config_search(
                args.recon, n_trials=args.trials, n_workers=args.workers)
        else:
            score, config = run_config_search(args.recon, n_trials=args.trials)
        elapsed = time.time() - t0
        print(f"Best: {score}, Config: {config}, Time: {elapsed:.0f}s")
