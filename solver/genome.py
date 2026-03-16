"""Genome representation for nightmare bot decisions.

A genome encodes WHICH bot picks WHICH item from WHICH shelf for each order.
The strategy reads the genome and executes the decisions using PIBT navigation.
"""

from __future__ import annotations

import random
import copy
from dataclasses import dataclass, field

Pos = tuple[int, int]


@dataclass
class ItemAssignment:
    """One item pickup assignment."""
    item_type: str
    shelf_index: int  # index into shelf_map[item_type]
    bot_id: int


@dataclass
class OrderAssignment:
    """All assignments for one order."""
    assignments: list[ItemAssignment] = field(default_factory=list)


@dataclass
class Genome:
    """Full game plan: assignments for all orders + routing parameters."""
    orders: list[OrderAssignment] = field(default_factory=list)

    # Route preferences: per-bot zone drop-off preference (0=left, 1=center, 2=right)
    # -1 means "auto" (use load-balanced selection)
    bot_zone_pref: dict[int, int] = field(default_factory=dict)

    # Guidance graph tuning (affects congestion-weighted routing)
    guidance_alpha: float = 2.0  # Congestion penalty weight
    guidance_beta: float = 3.0   # Historical congestion weight
    guidance_decay: float = 0.7  # Decay rate for historical traffic

    # Drop-off load balance factor (higher = more spreading)
    dropoff_load_factor: int = 5

    # Max concurrent deliverers (0 = auto)
    max_deliverers: int = 0

    # Sprint team size: how many bots focus ONLY on active order (rest do pipeline)
    # 0 = auto (ceil(remaining_items / 3))
    sprint_team_size: int = 0

    # Max deliverers per drop-off zone (caps gridlock at each zone)
    max_deliverers_per_zone: int = 2

    # Pre-position rounds: send preview-holders to drop-off N rounds before order completes
    preposition_rounds: int = 5

    def mutate(self, n_bots: int = 20, max_shelf: int = 8) -> Genome:
        """Create mutated copy."""
        new = copy.deepcopy(self)
        if not new.orders:
            return new

        r = random.random()

        if r < 0.3:
            # Swap bot assignment for random item
            oi = random.randint(0, len(new.orders) - 1)
            if new.orders[oi].assignments:
                ai = random.randint(0, len(new.orders[oi].assignments) - 1)
                new.orders[oi].assignments[ai].bot_id = random.randint(0, n_bots - 1)

        elif r < 0.55:
            # Change shelf for random item
            oi = random.randint(0, len(new.orders) - 1)
            if new.orders[oi].assignments:
                ai = random.randint(0, len(new.orders[oi].assignments) - 1)
                new.orders[oi].assignments[ai].shelf_index = random.randint(0, max_shelf - 1)

        elif r < 0.7:
            # Swap order of two items within an order
            oi = random.randint(0, len(new.orders) - 1)
            a = new.orders[oi].assignments
            if len(a) >= 2:
                i, j = random.sample(range(len(a)), 2)
                a[i], a[j] = a[j], a[i]

        elif r < 0.8:
            # Swap bot assignments between two items across orders
            if len(new.orders) >= 2:
                oi1, oi2 = random.sample(range(len(new.orders)), 2)
                a1 = new.orders[oi1].assignments
                a2 = new.orders[oi2].assignments
                if a1 and a2:
                    ai1 = random.randint(0, len(a1) - 1)
                    ai2 = random.randint(0, len(a2) - 1)
                    a1[ai1].bot_id, a2[ai2].bot_id = a2[ai2].bot_id, a1[ai1].bot_id

        elif r < 0.88:
            # Mutate guidance parameters
            param = random.choice(["alpha", "beta", "decay"])
            if param == "alpha":
                new.guidance_alpha = max(0.5, new.guidance_alpha + random.gauss(0, 0.5))
            elif param == "beta":
                new.guidance_beta = max(0.5, new.guidance_beta + random.gauss(0, 0.5))
            else:
                new.guidance_decay = max(0.1, min(0.99, new.guidance_decay + random.gauss(0, 0.1)))

        elif r < 0.91:
            # Mutate drop-off load factor
            new.dropoff_load_factor = max(1, min(15, new.dropoff_load_factor + random.choice([-2, -1, 1, 2])))

        elif r < 0.94:
            # Mutate sprint team size
            new.sprint_team_size = random.choice([0, 4, 5, 6, 7, 8, 10])

        elif r < 0.97:
            # Mutate max deliverers per zone
            new.max_deliverers_per_zone = random.choice([1, 2, 3, 4])

        else:
            # Mutate preposition rounds + max deliverers
            new.preposition_rounds = random.choice([0, 3, 5, 8, 10])
            new.max_deliverers = random.choice([0, 3, 6, 9, 12])

        return new

    def crossover(self, other: Genome) -> Genome:
        """Combine two genomes — take first half from self, second from other."""
        new = copy.deepcopy(self)
        mid = len(new.orders) // 2
        for i in range(mid, min(len(new.orders), len(other.orders))):
            new.orders[i] = copy.deepcopy(other.orders[i])
        # Mix routing params from other
        if random.random() < 0.5:
            new.guidance_alpha = other.guidance_alpha
            new.guidance_beta = other.guidance_beta
            new.guidance_decay = other.guidance_decay
        if random.random() < 0.5:
            new.dropoff_load_factor = other.dropoff_load_factor
        if random.random() < 0.5:
            new.max_deliverers = other.max_deliverers
        if random.random() < 0.5:
            new.sprint_team_size = other.sprint_team_size
        if random.random() < 0.5:
            new.max_deliverers_per_zone = other.max_deliverers_per_zone
        if random.random() < 0.5:
            new.preposition_rounds = other.preposition_rounds
        return new


def generate_genome(
    order_sequence: list[dict],
    shelf_map: dict[str, list],
    n_bots: int = 20,
    strategy: str = "greedy_nearest",  # or "random", "zone_affinity"
    drop_off_zones: list[Pos] | None = None,
) -> Genome:
    """Generate a genome from order sequence.

    Each order gets assignments for its required items PLUS pre-pick items
    from the next order, filling up to n_bots assignments total.
    This ensures ALL bots have work from round 1.

    strategy="greedy_nearest": assign nearest bot and shelf (baseline)
    strategy="random": random bot and shelf (for diversity)
    strategy="zone_affinity": assign bots to zones, pick zone-local shelves
    """
    genome = Genome()

    # Simple round-robin for greedy
    next_bot = 0

    for oi, order in enumerate(order_sequence):
        oa = OrderAssignment()
        items = list(order.get("items_required", []))

        # Add pre-pick items from next order to fill remaining bots
        if oi + 1 < len(order_sequence):
            next_items = list(order_sequence[oi + 1].get("items_required", []))
            # Fill up to n_bots total assignments
            remaining_slots = n_bots - len(items)
            if remaining_slots > 0:
                items.extend(next_items[:remaining_slots])

        for item_type in items:
            shelves = shelf_map.get(item_type, [])
            n_shelves = len(shelves)

            if strategy == "random":
                shelf_idx = random.randint(0, max(0, n_shelves - 1))
                bot_id = random.randint(0, n_bots - 1)
            elif strategy == "zone_affinity" and drop_off_zones:
                # Pick shelf closest to a random zone
                zone = random.choice(drop_off_zones)
                best_idx = 0
                best_d = 9999
                for i, sp in enumerate(shelves):
                    d = abs(sp[0] - zone[0]) + abs(sp[1] - zone[1])
                    if d < best_d:
                        best_d = d
                        best_idx = i
                shelf_idx = best_idx
                bot_id = next_bot % n_bots
                next_bot += 1
            else:  # greedy_nearest
                shelf_idx = 0  # nearest (shelf_map is sorted by distance)
                bot_id = next_bot % n_bots
                next_bot += 1

            oa.assignments.append(ItemAssignment(
                item_type=item_type,
                shelf_index=shelf_idx % max(1, n_shelves),
                bot_id=bot_id,
            ))

        genome.orders.append(oa)

    return genome
