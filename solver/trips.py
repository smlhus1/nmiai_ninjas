"""Trip planner: generate and evaluate item pickup trips for bots."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from collections import Counter

from .grid import GameMap, Pos, pickup_positions
from .pathfinding import DistanceCache
from .orders import ShelfIndex, ShelfEntry


@dataclass(frozen=True)
class PickupStep:
    """One step in a trip: go to pickup_pos, pick up item_type from shelf_pos."""

    item_type: str
    shelf_pos: Pos
    pickup_pos: Pos


@dataclass(frozen=True)
class Trip:
    """A planned trip: sequence of pickups followed by delivery to a drop-off."""

    steps: tuple[PickupStep, ...]
    drop_off: Pos
    cost: int  # total steps: bot_pos -> pickup1 -> ... -> pickupN -> drop_off

    @property
    def item_count(self) -> int:
        return len(self.steps)

    @property
    def item_types(self) -> list[str]:
        return [s.item_type for s in self.steps]


def trip_cost(
    bot_pos: Pos,
    steps: tuple[PickupStep, ...],
    drop_off: Pos,
    dist_cache: DistanceCache,
) -> int | None:
    """Calculate total movement cost for a trip.

    Returns None if any segment is unreachable.
    """
    total = 0
    current = bot_pos

    for step in steps:
        d = dist_cache.distance(current, step.pickup_pos)
        if d is None:
            return None
        total += d
        current = step.pickup_pos

    # Final leg: last pickup -> drop-off
    d = dist_cache.distance(current, drop_off)
    if d is None:
        return None
    total += d

    return total


def _best_permutation(
    bot_pos: Pos,
    steps: list[PickupStep],
    drop_off: Pos,
    dist_cache: DistanceCache,
) -> tuple[tuple[PickupStep, ...], int] | None:
    """Find the optimal ordering of pickup steps (TSP brute force, max 3! = 6)."""
    best_order = None
    best_cost = float("inf")

    for perm in permutations(steps):
        c = trip_cost(bot_pos, perm, drop_off, dist_cache)
        if c is not None and c < best_cost:
            best_cost = c
            best_order = perm

    if best_order is None:
        return None
    return best_order, int(best_cost)


class TripPlanner:
    """Generates candidate trips for bots given an order's item requirements."""

    def __init__(
        self,
        game_map: GameMap,
        dist_cache: DistanceCache,
        shelf_index: ShelfIndex,
    ) -> None:
        self.game_map = game_map
        self.dist_cache = dist_cache
        self.shelf_index = shelf_index

    def generate_trips(
        self,
        items_needed: Counter[str],
        bot_pos: Pos,
        max_items: int = 3,
        drop_off: Pos | None = None,
    ) -> list[Trip]:
        """Generate a set of trips that covers all items_needed.

        Strategy: greedy batching — assign items to the nearest drop-off zone,
        batch up to max_items per trip, optimize pickup order via TSP.

        Returns a list of trips. Each trip picks up 1-3 items and delivers to a zone.
        """
        if drop_off is None:
            # Use nearest drop-off to bot
            best_dz = None
            best_dist = float("inf")
            for dz in self.game_map.drop_off_zones:
                d = self.dist_cache.distance(bot_pos, dz)
                if d is not None and d < best_dist:
                    best_dist = d
                    best_dz = dz
            drop_off = best_dz or self.game_map.drop_off_zones[0]

        # Assign each needed item to its best shelf (nearest to drop_off)
        item_list: list[PickupStep] = []
        for item_type, count in items_needed.items():
            entries = self.shelf_index.get(item_type, drop_off)
            if not entries:
                continue
            # Use first `count` entries (they're sorted by distance)
            for i in range(count):
                entry = entries[i % len(entries)]
                item_list.append(PickupStep(
                    item_type=item_type,
                    shelf_pos=entry.shelf_pos,
                    pickup_pos=entry.pickup_pos,
                ))

        # Batch into trips of max_items
        trips: list[Trip] = []
        for i in range(0, len(item_list), max_items):
            batch = item_list[i : i + max_items]
            result = _best_permutation(bot_pos, batch, drop_off, self.dist_cache)
            if result is None:
                continue
            ordered_steps, cost = result
            trips.append(Trip(
                steps=ordered_steps,
                drop_off=drop_off,
                cost=cost,
            ))
            # Update bot_pos for next trip to be the drop_off
            bot_pos = drop_off

        return trips

    def generate_zone_trips(
        self,
        items_needed: Counter[str],
        bot_pos: Pos,
        max_items: int = 3,
    ) -> list[Trip]:
        """Generate trips with zone-aware shelf assignment.

        For each item, pick the shelf closest to the nearest drop-off zone.
        Group items by their best zone, then batch within each zone.
        """
        # Assign items to zones
        zone_items: dict[Pos, list[PickupStep]] = {
            dz: [] for dz in self.game_map.drop_off_zones
        }

        for item_type, count in items_needed.items():
            for _ in range(count):
                best_step = None
                best_zone = None
                best_dist = float("inf")

                for dz in self.game_map.drop_off_zones:
                    entry = self.shelf_index.nearest(item_type, dz)
                    if entry and entry.distance_to_dropoff < best_dist:
                        best_dist = entry.distance_to_dropoff
                        best_zone = dz
                        best_step = PickupStep(
                            item_type=item_type,
                            shelf_pos=entry.shelf_pos,
                            pickup_pos=entry.pickup_pos,
                        )

                if best_step and best_zone:
                    zone_items[best_zone].append(best_step)

        # Generate trips per zone
        trips: list[Trip] = []
        current_pos = bot_pos

        for dz in self.game_map.drop_off_zones:
            items = zone_items[dz]
            if not items:
                continue
            for i in range(0, len(items), max_items):
                batch = items[i : i + max_items]
                result = _best_permutation(current_pos, batch, dz, self.dist_cache)
                if result is None:
                    continue
                ordered_steps, cost = result
                trips.append(Trip(
                    steps=ordered_steps,
                    drop_off=dz,
                    cost=cost,
                ))
                current_pos = dz

        return trips
