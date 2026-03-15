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
    """Full game plan: assignments for all orders."""
    orders: list[OrderAssignment] = field(default_factory=list)

    def mutate(self, n_bots: int = 20, max_shelf: int = 8) -> Genome:
        """Create mutated copy."""
        new = copy.deepcopy(self)
        if not new.orders:
            return new

        r = random.random()

        if r < 0.4:
            # Swap bot assignment for random item
            oi = random.randint(0, len(new.orders) - 1)
            if new.orders[oi].assignments:
                ai = random.randint(0, len(new.orders[oi].assignments) - 1)
                new.orders[oi].assignments[ai].bot_id = random.randint(0, n_bots - 1)

        elif r < 0.7:
            # Change shelf for random item
            oi = random.randint(0, len(new.orders) - 1)
            if new.orders[oi].assignments:
                ai = random.randint(0, len(new.orders[oi].assignments) - 1)
                new.orders[oi].assignments[ai].shelf_index = random.randint(0, max_shelf - 1)

        elif r < 0.9:
            # Swap order of two items within an order
            oi = random.randint(0, len(new.orders) - 1)
            a = new.orders[oi].assignments
            if len(a) >= 2:
                i, j = random.sample(range(len(a)), 2)
                a[i], a[j] = a[j], a[i]

        else:
            # Swap bot assignments between two items across orders
            if len(new.orders) >= 2:
                oi1, oi2 = random.sample(range(len(new.orders)), 2)
                a1 = new.orders[oi1].assignments
                a2 = new.orders[oi2].assignments
                if a1 and a2:
                    ai1 = random.randint(0, len(a1) - 1)
                    ai2 = random.randint(0, len(a2) - 1)
                    a1[ai1].bot_id, a2[ai2].bot_id = a2[ai2].bot_id, a1[ai1].bot_id

        return new

    def crossover(self, other: Genome) -> Genome:
        """Combine two genomes — take first half from self, second from other."""
        new = copy.deepcopy(self)
        mid = len(new.orders) // 2
        for i in range(mid, min(len(new.orders), len(other.orders))):
            new.orders[i] = copy.deepcopy(other.orders[i])
        return new


def generate_genome(
    order_sequence: list[dict],
    shelf_map: dict[str, list],
    n_bots: int = 20,
    strategy: str = "greedy_nearest",  # or "random", "zone_affinity"
    drop_off_zones: list[Pos] | None = None,
) -> Genome:
    """Generate a genome from order sequence.

    strategy="greedy_nearest": assign nearest bot and shelf (baseline)
    strategy="random": random bot and shelf (for diversity)
    strategy="zone_affinity": assign bots to zones, pick zone-local shelves
    """
    genome = Genome()

    # Simple round-robin for greedy
    next_bot = 0

    for order in order_sequence:
        oa = OrderAssignment()
        items = order.get("items_required", [])

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
