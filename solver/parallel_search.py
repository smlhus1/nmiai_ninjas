"""Parallel evolutionary search over genomes using multiprocessing.

Runs 16 simulations in parallel. Evolves genomes to maximize score.
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

from solver.genome import Genome, generate_genome
from solver.genome_strategy import run_genome


def evaluate_genome(args: tuple) -> tuple[int, int, int, int, int]:
    """Worker function. Returns (genome_idx, score, orders, rounds, mid_score)."""
    idx, genome_dict, recon_path = args
    genome = _genome_from_dict(genome_dict)
    try:
        score, orders, rounds, mid_score = run_genome(recon_path, genome)
    except Exception:
        score, orders, rounds, mid_score = 0, 0, 500, 0
    return idx, score, orders, rounds, mid_score


def _genome_to_dict(g: Genome) -> dict:
    """Serialize genome for multiprocessing."""
    return {
        "orders": [
            [{"item_type": a.item_type, "shelf_index": a.shelf_index, "bot_id": a.bot_id}
             for a in o.assignments]
            for o in g.orders
        ],
        "guidance_alpha": g.guidance_alpha,
        "guidance_beta": g.guidance_beta,
        "guidance_decay": g.guidance_decay,
        "dropoff_load_factor": g.dropoff_load_factor,
        "max_deliverers": g.max_deliverers,
        "sprint_team_size": g.sprint_team_size,
        "max_deliverers_per_zone": g.max_deliverers_per_zone,
        "preposition_rounds": g.preposition_rounds,
    }


def _genome_from_dict(d: dict) -> Genome:
    """Deserialize genome."""
    from solver.genome import Genome, OrderAssignment, ItemAssignment
    genome = Genome()
    for order_data in d["orders"]:
        oa = OrderAssignment()
        for a in order_data:
            oa.assignments.append(ItemAssignment(
                item_type=a["item_type"],
                shelf_index=a["shelf_index"],
                bot_id=a["bot_id"],
            ))
        genome.orders.append(oa)
    # Restore routing params
    genome.guidance_alpha = d.get("guidance_alpha", 2.0)
    genome.guidance_beta = d.get("guidance_beta", 3.0)
    genome.guidance_decay = d.get("guidance_decay", 0.7)
    genome.dropoff_load_factor = d.get("dropoff_load_factor", 5)
    genome.max_deliverers = d.get("max_deliverers", 0)
    genome.sprint_team_size = d.get("sprint_team_size", 0)
    genome.max_deliverers_per_zone = d.get("max_deliverers_per_zone", 2)
    genome.preposition_rounds = d.get("preposition_rounds", 5)
    return genome


def evolve(
    recon_path: str,
    pop_size: int = 20,
    generations: int = 50,
    n_workers: int = None,
):
    """Evolutionary search with parallel evaluation."""
    if n_workers is None:
        n_workers = min(cpu_count(), 16)

    with open(recon_path) as f:
        recon = json.load(f)

    order_seq = recon["order_sequence"]
    shelf_map = recon["shelf_map"]
    n_bots = recon.get("bot_count", 20)
    drop_off_zones = [tuple(z) for z in recon["drop_off_zones"]]

    # Generate initial population
    population: list[Genome] = []
    for i in range(pop_size):
        if i == 0:
            g = generate_genome(order_seq, shelf_map, n_bots, strategy="greedy_nearest",
                              drop_off_zones=drop_off_zones)
        elif i < pop_size // 3:
            g = generate_genome(order_seq, shelf_map, n_bots, strategy="zone_affinity",
                              drop_off_zones=drop_off_zones)
        else:
            g = generate_genome(order_seq, shelf_map, n_bots, strategy="random",
                              drop_off_zones=drop_off_zones)
        population.append(g)

    best_score = 0
    best_fitness = 0.0
    best_genome = None

    print(f"Starting evolution: pop={pop_size}, gens={generations}, workers={n_workers}", flush=True)
    print(f"Fitness: velocity mode (score * 500 / rounds_used)", flush=True)

    for gen in range(generations):
        # Evaluate all genomes in parallel
        args = [(i, _genome_to_dict(g), recon_path) for i, g in enumerate(population)]

        with Pool(n_workers) as pool:
            results = pool.map(evaluate_genome, args)

        # Collect scores and fitness
        # Velocity fitness: mid_score (at round 250) + total score
        # This rewards genomes that complete orders FAST (high mid_score)
        # while still maximizing total output
        scores = [0] * pop_size
        fitness = [0.0] * pop_size
        for idx, score, orders, rounds, mid_score in results:
            scores[idx] = score
            # Weight mid-game score heavily: fast completion = more orders in live
            fitness[idx] = mid_score * 2 + score

        # Sort by FITNESS (velocity), not raw score
        ranked = sorted(range(pop_size), key=lambda i: -fitness[i])

        gen_best_score = scores[ranked[0]]
        gen_best_fitness = fitness[ranked[0]]
        gen_avg = sum(scores) / len(scores)

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_score = gen_best_score
            best_genome = copy.deepcopy(population[ranked[0]])
            # Save
            Path("logs/best_genome.json").write_text(
                json.dumps(_genome_to_dict(best_genome), indent=None))
            print(f"Gen {gen:3d}: NEW BEST score={best_score} fitness={best_fitness:.0f} (avg={gen_avg:.0f}) ***", flush=True)
        elif gen % 5 == 0:
            print(f"Gen {gen:3d}: best={best_score}, gen_best={gen_best_score}, avg={gen_avg:.0f}", flush=True)

        # Selection: keep top 20%
        elite_count = max(2, pop_size // 5)
        elite = [population[ranked[i]] for i in range(elite_count)]

        # Breed new generation
        new_pop = list(elite)  # keep elites

        while len(new_pop) < pop_size:
            r = random.random()
            if r < 0.5:
                # Mutate an elite
                parent = random.choice(elite)
                child = parent.mutate(n_bots=n_bots)
                new_pop.append(child)
            elif r < 0.8:
                # Crossover two elites
                p1, p2 = random.sample(elite, min(2, len(elite)))
                child = p1.crossover(p2).mutate(n_bots=n_bots)
                new_pop.append(child)
            else:
                # Fresh random genome (maintain diversity)
                g = generate_genome(order_seq, shelf_map, n_bots, strategy="random",
                                  drop_off_zones=drop_off_zones)
                new_pop.append(g)

        population = new_pop[:pop_size]

    print(f"\n=== BEST: score={best_score}, fitness={best_fitness:.0f} ===", flush=True)
    return best_genome, best_score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=50)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    best, score = evolve(args.recon, pop_size=args.pop, generations=args.gens,
                         n_workers=args.workers)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)
