"""Evolutionary search over BotAdapter with shelf preferences + config params.

Searches: shelf_preference, guidance alpha/beta/decay, stuck thresholds,
endgame_threshold, gate_max_delay, pre_pick_rush_remaining.

Usage:
    py -m solver.adapter_search --recon <recon> [--pop 20] [--gens 30] [--workers 12]
"""

from __future__ import annotations

import json
import os
import sys
import time
import copy
import random
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Individual = shelf preferences + config overrides --

def _load_shelf_map(recon_path: str) -> dict[str, list]:
    with open(recon_path) as f:
        recon = json.load(f)
    return recon.get("shelf_map", {})


def _random_individual(shelf_map: dict[str, list]) -> dict:
    """Random individual: shelf prefs + config params."""
    ind = {"shelf_pref": {}, "config": {}}
    # Random shelf preferences
    for item_type, shelves in shelf_map.items():
        if shelves:
            ind["shelf_pref"][item_type] = random.randint(0, len(shelves) - 1)
    # Random config params
    ind["config"]["guidance_alpha"] = random.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    ind["config"]["guidance_beta"] = random.choice([1.0, 2.0, 3.0, 5.0, 8.0])
    ind["config"]["guidance_decay"] = random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
    ind["config"]["stuck_transit_rounds"] = random.randint(3, 8)
    ind["config"]["stuck_pick_rounds"] = random.randint(3, 7)
    ind["config"]["stuck_deliver_rounds"] = random.randint(1, 4)
    ind["config"]["endgame_threshold"] = random.choice([40, 50, 60, 70, 80])
    ind["config"]["gate_max_delay"] = random.randint(0, 6)
    ind["config"]["pre_pick_rush_remaining"] = random.choice([2, 3, 4, 6])
    return ind


def _mutate_individual(ind: dict, shelf_map: dict[str, list]) -> dict:
    """Mutate individual: change 1-2 shelf prefs OR 1 config param."""
    new = {"shelf_pref": dict(ind["shelf_pref"]), "config": dict(ind["config"])}
    r = random.random()

    if r < 0.5:
        # Mutate 1-2 shelf preferences
        types = list(shelf_map.keys())
        for _ in range(random.randint(1, 2)):
            if types:
                t = random.choice(types)
                shelves = shelf_map.get(t, [])
                if shelves:
                    new["shelf_pref"][t] = random.randint(0, len(shelves) - 1)
    else:
        # Mutate 1 config param
        param = random.choice([
            "guidance_alpha", "guidance_beta", "guidance_decay",
            "stuck_transit_rounds", "stuck_pick_rounds", "stuck_deliver_rounds",
            "endgame_threshold", "gate_max_delay", "pre_pick_rush_remaining",
        ])
        if param == "guidance_alpha":
            new["config"][param] = max(0.5, new["config"].get(param, 2.0) + random.gauss(0, 0.5))
        elif param == "guidance_beta":
            new["config"][param] = max(0.5, new["config"].get(param, 3.0) + random.gauss(0, 0.5))
        elif param == "guidance_decay":
            new["config"][param] = max(0.1, min(0.99, new["config"].get(param, 0.7) + random.gauss(0, 0.1)))
        elif param == "stuck_transit_rounds":
            new["config"][param] = max(3, min(10, new["config"].get(param, 6) + random.choice([-1, 1])))
        elif param == "stuck_pick_rounds":
            new["config"][param] = max(2, min(8, new["config"].get(param, 5) + random.choice([-1, 1])))
        elif param == "stuck_deliver_rounds":
            new["config"][param] = max(1, min(5, new["config"].get(param, 2) + random.choice([-1, 1])))
        elif param == "endgame_threshold":
            new["config"][param] = max(20, min(100, new["config"].get(param, 60) + random.choice([-10, -5, 5, 10])))
        elif param == "gate_max_delay":
            new["config"][param] = max(0, min(8, new["config"].get(param, 3) + random.choice([-1, 0, 1, 2])))
        elif param == "pre_pick_rush_remaining":
            new["config"][param] = random.choice([2, 3, 4, 6])

    return new


def _crossover_individual(p1: dict, p2: dict) -> dict:
    """Crossover: shelf prefs from one, config from the other (or mixed)."""
    new = {"shelf_pref": {}, "config": {}}
    # Shelf prefs: mix
    for t in set(list(p1["shelf_pref"].keys()) + list(p2["shelf_pref"].keys())):
        if random.random() < 0.5:
            new["shelf_pref"][t] = p1["shelf_pref"].get(t, 0)
        else:
            new["shelf_pref"][t] = p2["shelf_pref"].get(t, 0)
    # Config: mix (skip None values)
    for k in set(list(p1["config"].keys()) + list(p2["config"].keys())):
        v1 = p1["config"].get(k)
        v2 = p2["config"].get(k)
        if random.random() < 0.5 and v1 is not None:
            new["config"][k] = v1
        elif v2 is not None:
            new["config"][k] = v2
        elif v1 is not None:
            new["config"][k] = v1
    return new


def _eval_worker(args: tuple) -> tuple[int, int, int, int, dict]:
    """Worker: run BotAdapter with individual. Returns (idx, score, orders, score_180, individual)."""
    idx, individual, recon_path = args

    from Simulering.offline.bot_adapter import BotAdapter
    from Simulering.offline.simulator import Simulator
    from bot.config import CoordinatorConfig

    config = CoordinatorConfig.for_difficulty(20)
    # Apply config overrides
    for k, v in individual.get("config", {}).items():
        if v is not None and hasattr(config, k):
            setattr(config, k, v)
    # Apply shelf preference
    config.shelf_preference = individual.get("shelf_pref", {})

    adapter = BotAdapter(suppress_logs=True, config=config)
    sim = Simulator.from_recon_file(recon_path)

    try:
        # Run manually to capture score@180
        state = sim.reset()
        score_180 = 0
        for r in range(sim.max_rounds):
            response = adapter(state.to_dict())
            state, game_over = sim.step(response.get("actions", []))
            if r == 179:
                score_180 = sim._score
            if game_over:
                break
        score = sim._score
        orders = sim._orders_completed
    except Exception:
        score, orders, score_180 = 0, 0, 0

    return idx, score, orders, score_180, individual


def evolve_adapter(
    recon_path: str,
    pop_size: int = 20,
    generations: int = 30,
    n_workers: int = None,
):
    """Evolutionary search over BotAdapter shelf prefs + config params."""
    if n_workers is None:
        n_workers = min(cpu_count(), 14)

    shelf_map = _load_shelf_map(recon_path)
    print(f"Shelf map: {len(shelf_map)} item types", flush=True)

    # Initial population
    population: list[dict] = []

    # Seed with best known
    best_path = Path("logs/best_individual.json")
    if best_path.exists():
        seed = json.loads(best_path.read_text())
        population.append(seed)
        for _ in range(min(5, pop_size // 3)):
            population.append(_mutate_individual(seed, shelf_map))
        print(f"Seeded from best_individual.json + {len(population)-1} variants", flush=True)
    else:
        # Try old shelf_pref format
        old_path = Path("logs/best_shelf_pref.json")
        if old_path.exists():
            old_pref = json.loads(old_path.read_text())
            seed = {"shelf_pref": old_pref, "config": {}}
            population.append(seed)
            for _ in range(min(5, pop_size // 3)):
                population.append(_mutate_individual(seed, shelf_map))
            print(f"Seeded from best_shelf_pref.json + {len(population)-1} variants", flush=True)

    # Baseline (no overrides)
    population.append({"shelf_pref": {}, "config": {}})

    # Fill with random
    while len(population) < pop_size:
        population.append(_random_individual(shelf_map))

    best_score = 0
    best_fitness = 0.0
    best_ind = {}

    print(f"Starting evolution: pop={pop_size}, gens={generations}, workers={n_workers}", flush=True)

    for gen in range(generations):
        args = [(i, ind, recon_path) for i, ind in enumerate(population)]

        with Pool(n_workers) as pool:
            results = pool.map(_eval_worker, args)

        scores = [0] * pop_size
        scores_180 = [0] * pop_size
        fitness = [0.0] * pop_size
        inds = [{}] * pop_size
        for idx, score, orders, score_180, ind in results:
            scores[idx] = score
            scores_180[idx] = score_180
            # VELOCITY FITNESS: score@180 is king, total score breaks ties
            fitness[idx] = score_180 * 10 + score
            inds[idx] = ind

        ranked = sorted(range(pop_size), key=lambda i: -fitness[i])
        gen_best_score = scores[ranked[0]]
        gen_best_s180 = scores_180[ranked[0]]
        gen_best_fitness = fitness[ranked[0]]
        gen_avg = sum(scores) / len(scores)
        gen_avg_180 = sum(scores_180) / len(scores_180)

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_score = gen_best_score
            best_ind = inds[ranked[0]]
            Path("logs/best_individual.json").write_text(json.dumps(best_ind, indent=2))
            cfg = best_ind.get("config", {})
            print(f"Gen {gen:3d}: NEW BEST s180={gen_best_s180} total={best_score} (avg180={gen_avg_180:.0f} avg={gen_avg:.0f}) cfg={cfg} ***", flush=True)
        elif gen % 3 == 0:
            print(f"Gen {gen:3d}: best_s180={gen_best_s180} total={gen_best_score}, avg180={gen_avg_180:.0f}", flush=True)

        # Selection: top 20%
        elite_count = max(2, pop_size // 5)
        elite = [inds[ranked[i]] for i in range(elite_count)]

        # Breed
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            r = random.random()
            if r < 0.4:
                child = _mutate_individual(random.choice(elite), shelf_map)
                new_pop.append(child)
            elif r < 0.7:
                p1, p2 = random.sample(elite, min(2, len(elite)))
                child = _crossover_individual(p1, p2)
                child = _mutate_individual(child, shelf_map)
                new_pop.append(child)
            else:
                new_pop.append(_random_individual(shelf_map))

        population = new_pop[:pop_size]

    print(f"\n=== BEST: score={best_score}, fitness={best_fitness:.0f} ===", flush=True)
    return best_ind, best_score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=30)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    ind, score = evolve_adapter(args.recon, pop_size=args.pop,
                                generations=args.gens, n_workers=args.workers)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)
