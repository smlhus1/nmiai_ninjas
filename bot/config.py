"""
Tunable parameters for the Coordinator pipeline.

Used by the optimizer to search for best settings per map.
All defaults match the current hardcoded values.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, asdict


@dataclass
class CoordinatorConfig:
    # --- Route building ---
    max_route_items: int = 3
    max_route_candidates: int = 8
    max_route_starts: int = 8

    # --- Hungarian assignment ---
    order_completion_bonus: int = 5
    switching_penalty: float = 3.0

    # --- Endgame ---
    endgame_threshold: int = 40

    # --- Stuck detection ---
    stuck_transit_rounds: int = 6
    stuck_pick_rounds: int = 5
    stuck_deliver_rounds: int = 2

    # --- Oscillation detection ---
    oscillation_window: int = 8
    oscillation_max_unique: int = 3
    oscillation_min_target_dist: int = 2

    # --- Pre-pick ---
    pre_pick_max_inventory: int = 2
    pre_pick_rush_remaining: int = 2

    # --- Timefeasibility margin ---
    time_margin: int = 2

    # --- Exploration noise ---
    # Adds controlled randomness to distance calcs, making each config
    # produce different routing decisions. 0 = deterministic (default).
    distance_noise: float = 0.0
    # Seed for reproducible noise. Different seeds = different decisions.
    noise_seed: int = 0
    # Shelf selection bias: 0 = nearest, 1 = random shelf choice
    shelf_randomness: float = 0.0

    def mutate(self, temperature: float = 1.0) -> CoordinatorConfig:
        """Return a mutated copy for hill-climbing optimization."""
        p = copy.copy(self)
        r = random.random

        if r() < 0.3 * temperature:
            p.max_route_items = random.choice([1, 2, 3])
        if r() < 0.2 * temperature:
            p.max_route_candidates = max(2, p.max_route_candidates + random.choice([-2, -1, 1, 2]))
        if r() < 0.2 * temperature:
            p.max_route_starts = max(2, p.max_route_starts + random.choice([-2, -1, 1, 2]))
        if r() < 0.3 * temperature:
            p.order_completion_bonus = max(0, p.order_completion_bonus + random.choice([-2, -1, 1, 2, 3]))
        if r() < 0.25 * temperature:
            p.switching_penalty = max(0.0, p.switching_penalty + random.uniform(-1.5, 1.5))
        if r() < 0.3 * temperature:
            p.endgame_threshold = max(10, p.endgame_threshold + random.choice([-10, -5, 5, 10]))
        if r() < 0.2 * temperature:
            p.stuck_transit_rounds = max(3, p.stuck_transit_rounds + random.choice([-2, -1, 1, 2]))
        if r() < 0.2 * temperature:
            p.stuck_pick_rounds = max(2, p.stuck_pick_rounds + random.choice([-2, -1, 1, 2]))
        if r() < 0.15 * temperature:
            p.stuck_deliver_rounds = max(1, p.stuck_deliver_rounds + random.choice([-1, 1]))
        if r() < 0.2 * temperature:
            p.oscillation_window = max(4, p.oscillation_window + random.choice([-2, -1, 1, 2]))
        if r() < 0.15 * temperature:
            p.oscillation_max_unique = random.choice([2, 3, 4])
        if r() < 0.2 * temperature:
            p.pre_pick_max_inventory = random.choice([1, 2, 3])
        if r() < 0.2 * temperature:
            p.pre_pick_rush_remaining = random.choice([1, 2, 3])
        if r() < 0.2 * temperature:
            p.time_margin = random.choice([0, 1, 2, 3, 4])
        # ALWAYS vary noise/seed — this is the primary source of exploration
        p.distance_noise = random.uniform(0.0, 3.0)
        p.noise_seed = random.randint(0, 100000)
        p.shelf_randomness = random.uniform(0.0, 0.5)

        return p

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CoordinatorConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
