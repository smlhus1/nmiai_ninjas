"""Evolutionary route optimizer.

Mutates bot routing decisions (shelf choice, zone assignment, item priority)
and runs validated sim to find optimal throughput. Each "genome" is a set of
routing overrides that deterministically control which bot picks which item.

Key insight: distance_noise + noise_seed in CoordinatorConfig changes which
shelf a bot picks. Different seeds = different routing = different score.
But we can go DEEPER by controlling shelf_randomness and zone assignments.

This optimizer searches the space of routing decisions, not just params.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import copy
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import CoordinatorConfig
from Simulering.offline.simulator import Simulator
from Simulering.offline.bot_adapter import BotAdapter


@dataclass
class Genome:
    """A routing decision genome — controls how bots make choices."""
    config: CoordinatorConfig
    score: int = 0
    orders: int = 0

    def mutate(self) -> Genome:
        """Create a mutated copy."""
        new_cfg = copy.copy(self.config)

        r = random.random

        # Always vary noise — this is the primary routing control
        new_cfg.noise_seed = random.randint(0, 999999)
        new_cfg.distance_noise = random.uniform(0.01, 0.5)

        # Shelf randomness — controls which shelf copy is chosen
        if r() < 0.3:
            new_cfg.shelf_randomness = random.uniform(0.0, 0.4)

        # Guidance params — affect congestion routing
        if r() < 0.2:
            new_cfg.guidance_alpha = random.choice([0.5, 1.0, 2.0, 3.0, 4.0])
        if r() < 0.2:
            new_cfg.guidance_beta = random.choice([1.0, 2.0, 3.0, 5.0, 8.0])
        if r() < 0.2:
            new_cfg.guidance_update_interval = random.choice([2, 3, 5, 8])
        if r() < 0.15:
            new_cfg.guidance_decay = random.choice([0.3, 0.5, 0.7, 0.9])

        # Switching penalty affects stickiness
        if r() < 0.15:
            new_cfg.switching_penalty = random.uniform(1.0, 6.0)

        # Endgame threshold
        if r() < 0.1:
            new_cfg.endgame_threshold = random.choice([40, 50, 60, 70, 80])

        # Route items
        if r() < 0.1:
            new_cfg.max_route_items = random.choice([1, 2, 3])

        # Pre-pick params
        if r() < 0.1:
            new_cfg.pre_pick_rush_remaining = random.choice([1, 2, 3, 4])

        return Genome(config=new_cfg)

    def crossover(self, other: Genome) -> Genome:
        """Combine two genomes."""
        new_cfg = copy.copy(self.config)
        other_cfg = other.config

        # Take noise from better parent
        if other.score > self.score:
            new_cfg.noise_seed = other_cfg.noise_seed
            new_cfg.distance_noise = other_cfg.distance_noise

        # Mix guidance from both
        if random.random() < 0.5:
            new_cfg.guidance_alpha = other_cfg.guidance_alpha
            new_cfg.guidance_beta = other_cfg.guidance_beta

        # New noise seed for variety
        new_cfg.noise_seed = random.randint(0, 999999)

        return Genome(config=new_cfg)


def evaluate(genome: Genome, recon_path: str) -> Genome:
    """Run sim and set score."""
    sim = Simulator.from_recon_file(recon_path)
    adapter = BotAdapter(suppress_logs=True, config=genome.config)
    result = sim.run(adapter)
    genome.score = result["score"]
    genome.orders = result["orders_completed"]
    return genome


def evolve(
    recon_path: str,
    base_config: CoordinatorConfig,
    population_size: int = 20,
    generations: int = 50,
    elite_count: int = 5,
) -> Genome:
    """Evolutionary search for optimal routing."""

    # Initialize population
    population: list[Genome] = []
    for i in range(population_size):
        if i == 0:
            g = Genome(config=copy.copy(base_config))
        else:
            g = Genome(config=copy.copy(base_config)).mutate()
        evaluate(g, recon_path)
        population.append(g)

    population.sort(key=lambda g: -g.score)
    best = population[0]
    print(f"Gen 0: best={best.score}, orders={best.orders}, "
          f"avg={sum(g.score for g in population)/len(population):.0f}", flush=True)

    for gen in range(1, generations + 1):
        # Selection: keep elite
        elite = population[:elite_count]

        # Breed new generation
        children: list[Genome] = []

        # Mutations of elite
        for e in elite:
            for _ in range(2):
                child = e.mutate()
                evaluate(child, recon_path)
                children.append(child)

        # Crossovers
        while len(children) < population_size - elite_count:
            p1 = random.choice(elite)
            p2 = random.choice(elite[:3])
            child = p1.crossover(p2).mutate()
            evaluate(child, recon_path)
            children.append(child)

        population = elite + children
        population.sort(key=lambda g: -g.score)
        population = population[:population_size]

        if population[0].score > best.score:
            best = population[0]
            Path("logs/best_nightmare_config.json").write_text(
                json.dumps(best.config.to_dict(), indent=2))
            print(f"Gen {gen}: NEW BEST {best.score} (orders={best.orders}) ***", flush=True)
        elif gen % 5 == 0:
            print(f"Gen {gen}: best={best.score}, "
                  f"avg={sum(g.score for g in population)/len(population):.0f}", flush=True)

    return best


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=50)
    args = parser.parse_args()

    base = CoordinatorConfig.from_dict(
        json.loads(Path("logs/best_nightmare_config.json").read_text()))

    t0 = time.time()
    best = evolve(args.recon, base, population_size=args.pop, generations=args.gens)
    elapsed = time.time() - t0

    print(f"\n=== BEST: score={best.score}, orders={best.orders} ===")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(json.dumps(best.config.to_dict(), indent=2))
