"""Order sequence management and shelf indexing."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from .grid import GameMap, Pos, pickup_positions
from .pathfinding import DistanceCache


@dataclass
class Order:
    """A single customer order."""

    id: str
    items_required: list[str]
    first_seen_round: int | None = None
    activated_round: int | None = None
    completed_round: int | None = None

    @property
    def item_count(self) -> int:
        return len(self.items_required)

    def items_as_counter(self) -> Counter[str]:
        return Counter(self.items_required)


class OrderQueue:
    """Manages the order sequence from recon data."""

    def __init__(self, orders: list[Order]) -> None:
        self._orders = list(orders)
        self._active_idx = 0
        self._completed: list[Order] = []

    @staticmethod
    def from_recon(path: str) -> OrderQueue:
        """Load order sequence from recon JSON."""
        with open(path) as f:
            data = json.load(f)

        orders = []
        for o in data["order_sequence"]:
            orders.append(Order(
                id=o["id"],
                items_required=o["items_required"],
                first_seen_round=o.get("first_seen_round"),
                activated_round=o.get("activated_round"),
                completed_round=o.get("completed_round"),
            ))
        return OrderQueue(orders)

    @property
    def active(self) -> Order | None:
        """Currently active order."""
        if self._active_idx < len(self._orders):
            return self._orders[self._active_idx]
        return None

    @property
    def preview(self) -> Order | None:
        """Next order (preview)."""
        idx = self._active_idx + 1
        if idx < len(self._orders):
            return self._orders[idx]
        return None

    @property
    def completed(self) -> list[Order]:
        return list(self._completed)

    @property
    def remaining(self) -> list[Order]:
        """All orders not yet completed."""
        return self._orders[self._active_idx:]

    def advance(self) -> Order | None:
        """Mark active order as complete, promote preview to active.

        Returns the newly active order, or None if no orders remain.
        """
        if self._active_idx < len(self._orders):
            self._completed.append(self._orders[self._active_idx])
            self._active_idx += 1
        return self.active

    def items_needed(self) -> Counter[str]:
        """Item types still needed for the active order."""
        if self.active is None:
            return Counter()
        return self.active.items_as_counter()

    def __len__(self) -> int:
        return len(self._orders)

    def __iter__(self):
        return iter(self._orders)


@dataclass
class ShelfEntry:
    """A shelf position and its best pickup position relative to a drop-off."""

    shelf_pos: Pos
    pickup_pos: Pos
    distance_to_dropoff: int


class ShelfIndex:
    """For each item type, provides shelf/pickup positions sorted by distance to drop-off zones."""

    def __init__(self, game_map: GameMap, dist_cache: DistanceCache) -> None:
        self.game_map = game_map
        self.dist_cache = dist_cache

        # item_type -> drop_off_pos -> sorted list of ShelfEntry
        self._index: dict[str, dict[Pos, list[ShelfEntry]]] = {}
        self._build()

    def _build(self) -> None:
        grid = self.game_map.grid

        for item_type, shelf_positions in self.game_map.shelf_map.items():
            self._index[item_type] = {}

            for dz in self.game_map.drop_off_zones:
                dz_distances = self.dist_cache.distance_from(dz)
                entries: list[ShelfEntry] = []

                for shelf_pos in shelf_positions:
                    pickups = pickup_positions(grid, shelf_pos)
                    if not pickups:
                        continue

                    # Find the pickup position closest to this drop-off
                    best_pickup = None
                    best_dist = float("inf")
                    for pp in pickups:
                        d = dz_distances.get(pp)
                        if d is not None and d < best_dist:
                            best_dist = d
                            best_pickup = pp

                    if best_pickup is not None:
                        entries.append(ShelfEntry(
                            shelf_pos=shelf_pos,
                            pickup_pos=best_pickup,
                            distance_to_dropoff=int(best_dist),
                        ))

                entries.sort(key=lambda e: e.distance_to_dropoff)
                self._index[item_type][dz] = entries

    def get(self, item_type: str, drop_off: Pos) -> list[ShelfEntry]:
        """Get shelf entries for item_type sorted by distance to drop_off."""
        return self._index.get(item_type, {}).get(drop_off, [])

    def nearest(self, item_type: str, drop_off: Pos) -> ShelfEntry | None:
        """Get the nearest shelf entry for item_type relative to drop_off."""
        entries = self.get(item_type, drop_off)
        return entries[0] if entries else None

    def item_types(self) -> list[str]:
        """All available item types."""
        return list(self._index.keys())
