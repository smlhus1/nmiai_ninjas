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

    # --- Replay ---
    replay_enabled: bool = False

    # --- Pre-pick ---
    pre_pick_max_inventory: int = 2
    pre_pick_rush_remaining: int = 4  # Rush preview holders when active has <= this many items left
    pre_pick_rush_max_rounds: int = 25  # Only rush when order completes within this many rounds

    # --- Timefeasibility margin ---
    time_margin: int = 2

    # --- Deliver vs pick (planner) ---
    deliver_detour_threshold: int = 10
    endgame_multi_item_min_rounds: int = 8

    # --- Hungarian (route cost) ---
    parallelism_penalty: float = 2.0

    # --- WorldModel (endgame / feasibility) ---
    rounds_per_item_estimate: int = 6
    endgame_min_rounds: int = 15

    # --- RouteBuilder (preview pre-pick) ---
    preview_detour_tolerance: int = 4
    extra_shelf_alternatives: int = 2

    # --- Planner validation (stuck pickup blacklist) ---
    blacklist_expiry_rounds: int = 8

    # --- Completion gate (pipeline: hold active until preview ready) ---
    gate_max_delay: int = 3  # Max rounds to hold last active item while preview gets ready.
    gate_min_rounds_remaining: int = 40  # Below this, never gate (endgame)

    # --- Planner version ---
    planner_version: int = 2  # 1=v1 legacy, 2=v2 order-centric

    # --- Time-space planning ---
    tsp_horizon: int = 30          # Planning horizon (timesteps ahead)
    tsp_max_planning_ms: int = 500  # Time budget for path planning

    # --- Guidance graph (congestion-aware routing) ---
    guidance_enabled: bool = False
    guidance_alpha: float = 2.0
    guidance_beta: float = 3.0
    guidance_decay: float = 0.7
    guidance_update_interval: int = 5

    # --- LNS refinement ---
    lns_enabled: bool = False
    lns_budget_ms: int = 500
    lns_neighborhood_size: int = 5

    # --- Exploration noise ---
    # Adds controlled randomness to distance calcs, making each config
    # produce different routing decisions. 0 = deterministic (default).
    distance_noise: float = 0.0
    # Seed for reproducible noise. Different seeds = different decisions.
    noise_seed: int = 0
    # Shelf selection bias: 0 = nearest, 1 = random shelf choice
    shelf_randomness: float = 0.0
    # Genome shelf preference: item_type -> preferred shelf index
    # Used by evolutionary search to override nearest-item selection
    shelf_preference: dict = None

    # Known order sequence from recon — enables extended pre-picking (N+2, N+3)
    # List of order dicts with "items_required" field
    order_sequence: list = None

    @classmethod
    def for_difficulty(cls, n_bots: int) -> CoordinatorConfig:
        """Auto-select config preset based on bot count."""
        if n_bots == 1:
            return cls.easy()
        if n_bots <= 3:
            return cls.medium()
        if n_bots >= 20:
            return cls.nightmare()
        if n_bots >= 10:
            return cls.expert()
        return cls.hard()

    @classmethod
    def easy(cls) -> CoordinatorConfig:
        """Single-bot preset: replay disabled.

        Reactive planner scores 124 vs replay 93 on current map.
        Replay plan generation (OfflinePlanner) is suboptimal and causes regression.
        """
        return cls(replay_enabled=False)

    @classmethod
    def hard(cls) -> CoordinatorConfig:
        """5+ bot preset: larger map, more items, longer distances."""
        return cls(replay_enabled=False, endgame_threshold=50)

    @classmethod
    def expert(cls) -> CoordinatorConfig:
        """10 bot preset: large map, coordination-heavy."""
        return cls(replay_enabled=False, endgame_threshold=50)

    @classmethod
    def nightmare(cls) -> CoordinatorConfig:
        """10+ bot preset: massive map, 20 bots, drop-off bottleneck."""
        return cls(
            replay_enabled=False,
            endgame_threshold=60,
            pre_pick_max_inventory=3,
            guidance_enabled=True,
            guidance_update_interval=3,
            guidance_alpha=1.0,
        )

    @classmethod
    def medium(cls) -> CoordinatorConfig:
        """Multi-bot preset: replay disabled.

        Replay doesn't improve multi-bot because the bottleneck is coordination
        (drop-off congestion, collisions) not item selection. The reactive
        planner with Hungarian matching already optimizes assignment well.
        """
        return cls(replay_enabled=False)

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
        if r() < 0.2 * temperature:
            p.pre_pick_rush_max_rounds = max(10, min(40, p.pre_pick_rush_max_rounds + random.choice([-5, -3, 3, 5])))
        if r() < 0.15 * temperature:
            p.rounds_per_item_estimate = max(3, min(12, p.rounds_per_item_estimate + random.choice([-2, -1, 1, 2])))
        if r() < 0.2 * temperature:
            p.gate_max_delay = max(0, min(10, p.gate_max_delay + random.choice([-2, -1, 0, 1, 2, 3])))
        if r() < 0.15 * temperature:
            p.gate_min_rounds_remaining = max(20, min(80, p.gate_min_rounds_remaining + random.choice([-10, -5, 5, 10])))
        # Guidance params
        if r() < 0.3 * temperature:
            p.guidance_enabled = random.choice([True, False])
        if r() < 0.3 * temperature:
            p.guidance_alpha = random.choice([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
        if r() < 0.3 * temperature:
            p.guidance_beta = random.choice([1.0, 2.0, 3.0, 5.0, 8.0])
        if r() < 0.3 * temperature:
            p.guidance_update_interval = random.choice([2, 3, 5, 8, 10])
        if r() < 0.2 * temperature:
            p.guidance_decay = random.choice([0.5, 0.6, 0.7, 0.8, 0.9])

        return p

    def mutate_big(self, times: int = 4) -> CoordinatorConfig:
        """Large jump: apply mutate(1.0) repeatedly to escape local optima."""
        p = self
        for _ in range(times):
            p = p.mutate(temperature=1.0)
        return p

    @classmethod
    def random_restart(cls) -> CoordinatorConfig:
        """Sample a full config uniformly in bounds. For multi-start optimization."""
        return cls(
            max_route_items=random.choice([1, 2, 3]),
            max_route_candidates=random.randint(4, 16),
            max_route_starts=random.randint(4, 16),
            order_completion_bonus=random.randint(3, 10),
            switching_penalty=random.uniform(1.0, 6.0),
            endgame_threshold=random.randint(20, 60),
            stuck_transit_rounds=random.randint(3, 8),
            stuck_pick_rounds=random.randint(3, 7),
            stuck_deliver_rounds=random.randint(1, 4),
            pre_pick_rush_remaining=random.choice([2, 3, 4]),
            pre_pick_rush_max_rounds=random.randint(15, 35),
            rounds_per_item_estimate=random.randint(4, 10),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CoordinatorConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
