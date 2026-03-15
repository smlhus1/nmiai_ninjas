"""
WorldModel: enriched, queryable view of the game state.

This is the "brain's whiteboard" — it takes raw GameState and provides
higher-level queries like "what items are available for this order?"
and "which bot is closest to this item?".

Created fresh each round from GameState + persistent PathEngine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from bot.models import GameState, Bot, Item, Order, Pos, OrderStatus
from bot.engine.pathfinding import PathEngine


@dataclass
class ItemAvailability:
    """An item on the map that could fulfill part of an order."""

    item: Item
    distance_to_bot: int
    distance_to_dropoff: int
    pickup_pos: Optional[Pos] = None  # Walkable cell adjacent to shelf

    @property
    def total_trip(self) -> int:
        """Total steps: bot -> pickup_pos -> drop_off."""
        return self.distance_to_bot + self.distance_to_dropoff


class WorldModel:
    """
    Per-round enriched world view. NOT persisted between rounds.

    Usage:
        world = WorldModel(game_state, path_engine)
        candidates = world.items_for_order(order)
        nearest = world.nearest_item(bot, "milk")
    """

    def __init__(self, state: GameState, path_engine: PathEngine, *, max_route_items: int = 3) -> None:
        self.state = state
        self.path = path_engine
        self.max_route_items = max_route_items

        # Merged grid (includes shelves as walls) — use for walkability checks
        self._grid = path_engine._grid

        # Index items by type for fast lookup
        self._items_by_type: dict[str, list[Item]] = {}
        for item in state.items:
            self._items_by_type.setdefault(item.type, []).append(item)

        # Bot positions set (for obstacle awareness)
        self._bot_positions: set[Pos] = {b.position for b in state.bots}

        # Zone assignment for nightmare (3 zones, 20 bots)
        self._bot_zones: dict[int, Pos] = {}  # bot_id -> assigned drop-off zone
        self._zone_x_ranges: dict[Pos, tuple[int, int]] = {}  # zone -> (min_x, max_x)
        if len(state.drop_off_zones) >= 3 and len(state.bots) >= 10:
            self._setup_zones()

    def _setup_zones(self) -> None:
        """Assign bots to zones based on current position (nearest zone).

        Each bot is assigned to the zone whose drop-off is closest.
        This is recalculated every round, so bots naturally work
        in the zone they're currently in.
        """
        zones = sorted(self.state.drop_off_zones, key=lambda z: z[0])
        width = self._grid.width

        # Compute x-range boundaries (midpoints between adjacent zones)
        boundaries = [0]
        for i in range(len(zones) - 1):
            mid = (zones[i][0] + zones[i + 1][0]) // 2
            boundaries.append(mid)
        boundaries.append(width)

        for i, zone in enumerate(zones):
            self._zone_x_ranges[zone] = (boundaries[i], boundaries[i + 1])

        # Assign each bot to its nearest zone
        for bot in self.state.bots:
            nearest = min(zones, key=lambda z: self.path.distance(bot.position, z))
            self._bot_zones[bot.id] = nearest

    @property
    def rounds_remaining(self) -> int:
        return self.state.rounds_remaining

    def bot_zone(self, bot_id: int) -> Pos | None:
        """Return the assigned drop-off zone for a bot, or None if no zones.

        Zone is based on bot's CURRENT nearest drop-off, updated each round.
        This means bots naturally work in the zone they're closest to.
        """
        return self._bot_zones.get(bot_id)

    def item_in_zone(self, item_pos: Pos, zone: Pos) -> bool:
        """Check if an item position is within a zone's x-range."""
        x_range = self._zone_x_ranges.get(zone)
        if not x_range:
            return True  # No zones defined — everything is in zone
        return x_range[0] <= item_pos[0] < x_range[1]

    def nearest_drop_off(self, pos: Pos, bot_id: int | None = None) -> Pos:
        """Return the nearest drop-off zone to the given position.

        bot_id parameter is accepted but NOT used for zone override —
        bots always go to their geographically nearest zone.
        Zone assignment (bot_zone) is used for item filtering only.
        """
        zones = self.state.drop_off_zones
        if len(zones) == 1:
            return zones[0]
        return min(zones, key=lambda z: self.path.distance(pos, z))

    def items_of_type(self, item_type: str) -> list[Item]:
        """All items of a given type currently on the map."""
        return self._items_by_type.get(item_type, [])

    def items_for_order(self, order: Order) -> dict[str, list[Item]]:
        """Map of item_type -> available items on map, for each remaining item in order."""
        result: dict[str, list[Item]] = {}
        for item_type in order.items_remaining:
            available = self.items_of_type(item_type)
            if available:
                result[item_type] = available
        return result

    def pickup_positions(self, item_pos: Pos) -> list[Pos]:
        """Walkable cells adjacent to an item's shelf position."""
        result = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            pos = (item_pos[0] + dx, item_pos[1] + dy)
            # Use merged grid (includes shelves as walls) for accurate walkability
            if self._grid is not None and self._grid.is_walkable(pos):
                result.append(pos)
            elif self._grid is None and self.state.grid.is_walkable(pos):
                result.append(pos)
        return result

    def best_pickup_position(self, bot_pos: Pos, item_pos: Pos) -> Pos | None:
        """Nearest walkable cell adjacent to item, from bot's perspective."""
        positions = self.pickup_positions(item_pos)
        if not positions:
            return None
        best = min(positions, key=lambda p: self.path.distance(bot_pos, p))
        # Filter out unreachable positions (shelf cells that state.grid
        # considers walkable but are actually non-walkable in merged grid)
        if self.path.distance(bot_pos, best) >= 9999:
            return None
        return best

    def distance(self, a: Pos, b: Pos) -> int:
        """Pathfinding distance between two positions."""
        return self.path.distance(a, b)

    def nearest_item(self, bot: Bot, item_type: str) -> Optional[ItemAvailability]:
        """Find the nearest item of a type to a bot, with trip cost info.

        With zone assignment: strongly prefers items in bot's zone.
        """
        items = self.items_of_type(item_type)
        if not items:
            return None

        zone = self.bot_zone(bot.id)
        best: Optional[ItemAvailability] = None
        best_in_zone: Optional[ItemAvailability] = None
        for item in items:
            pickup_pos = self.best_pickup_position(bot.position, item.position)
            if pickup_pos is None:
                continue
            d_bot = self.path.distance(bot.position, pickup_pos)
            d_drop = self.path.distance(pickup_pos, self.nearest_drop_off(pickup_pos, bot.id))
            candidate = ItemAvailability(item, d_bot, d_drop, pickup_pos)
            if best is None or candidate.total_trip < best.total_trip:
                best = candidate
            if zone and self.item_in_zone(item.position, zone):
                if best_in_zone is None or candidate.total_trip < best_in_zone.total_trip:
                    best_in_zone = candidate

        # Prefer in-zone item; fall back to any if none in zone
        return best_in_zone if best_in_zone is not None else best

    def can_complete_trip(self, bot: Bot, item_pos: Pos) -> bool:
        """Check if a bot can pick up an item and deliver it before game ends."""
        pickup_pos = self.best_pickup_position(bot.position, item_pos)
        if pickup_pos is None:
            return False
        d_to_pickup = self.path.distance(bot.position, pickup_pos)
        d_to_drop = self.path.distance(pickup_pos, self.nearest_drop_off(pickup_pos))
        # +2 for pick_up + drop_off actions
        return (d_to_pickup + d_to_drop + 2) <= self.rounds_remaining

    def order_value(self, order: Order) -> float:
        """
        Estimate the value of working on this order.
        Higher = more attractive.
        """
        remaining = order.items_remaining
        delivered = len(order.items_delivered)
        total = len(order.items_required)

        if not remaining:
            return 0.0

        # Completion bonus potential
        items_left = len(remaining)
        # +5 bonus if we can complete, scaled by how close we are
        completion_bonus = 5.0 * (delivered / total) if total > 0 else 0.0

        # Check if items are actually available on the map
        available_count = sum(
            1 for item_type in remaining if self.items_of_type(item_type)
        )

        if available_count == 0:
            return 0.1  # Almost worthless if no items available

        # Value per item (base +1) plus weighted completion bonus
        availability_ratio = available_count / items_left
        return (available_count + completion_bonus) * availability_ratio

    def dropoff_adjacent_positions(self) -> list[Pos]:
        """Walkable cells adjacent to any drop-off zone."""
        result = []
        seen: set[Pos] = set()
        for drop in self.state.drop_off_zones:
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                pos = (drop[0] + dx, drop[1] + dy)
                if pos not in seen and self._grid.is_walkable(pos):
                    result.append(pos)
                    seen.add(pos)
        return result

    def staging_positions(self, distance: int = 2) -> list[Pos]:
        """Walkable cells near any drop-off zone for staging (2-3 cells away).

        Only includes cells the bot can reach AND leave (round-trip reachable).
        Avoids one-way traps where bots get stuck.
        """
        result = []
        seen: set[Pos] = set()
        for drop in self.state.drop_off_zones:
            for dx in range(-distance - 1, distance + 2):
                for dy in range(-distance - 1, distance + 2):
                    pos = (drop[0] + dx, drop[1] + dy)
                    if pos in seen:
                        continue
                    manhattan = abs(dx) + abs(dy)
                    if manhattan < 2 or manhattan > distance + 1:
                        continue
                    if not self._grid.is_walkable(pos):
                        continue
                    d_to = self.path.distance(drop, pos)
                    d_from = self.path.distance(pos, drop)
                    if d_to < 9999 and d_from < 9999:
                        result.append(pos)
                        seen.add(pos)
        return result

    def parking_positions(self) -> list[Pos]:
        """Walkable cells far from drop-off for idle bot parking.

        Prefers right edge of map to stay out of pickup aisles.
        """
        drop = self.state.drop_off
        width = self._grid.width
        height = self._grid.height

        candidates = []
        # Scan right edge first (columns width-1 down to width-4)
        for x in range(width - 1, max(width - 4, 0), -1):
            for y in range(height):
                pos = (x, y)
                if self._grid.is_walkable(pos):
                    d = self.path.distance(pos, drop)
                    if d >= 4:
                        candidates.append((d, pos))

        if candidates:
            candidates.sort(key=lambda t: -t[0])
            return [pos for _, pos in candidates[:6]]

        # Fallback: any walkable cell far from drop-off
        for x in range(width):
            for y in range(height):
                pos = (x, y)
                if self._grid.is_walkable(pos):
                    d = self.path.distance(pos, drop)
                    if d >= 4:
                        candidates.append((d, pos))

        candidates.sort(key=lambda t: -t[0])
        return [pos for _, pos in candidates[:6]]

    def bot_positions_except(self, bot_id: int) -> set[Pos]:
        """Positions of all bots except the given one."""
        return {b.position for b in self.state.bots if b.id != bot_id}

    def is_endgame(self, threshold: int = 40) -> bool:
        """True if we're in endgame. Threshold scales with bot count."""
        n_bots = max(len(self.state.bots), 1)
        dynamic_threshold = max(15, threshold * 2 // (n_bots + 1))
        return self.rounds_remaining <= dynamic_threshold

    def can_complete_active_order(self) -> bool:
        """Estimate whether the active order can be completed in remaining rounds.

        Uses actual map distances for bots with matching items, and a
        per-bot-count heuristic for remaining items to pick.
        """
        active = self.state.active_orders
        if not active:
            return False
        remaining = active[0].items_remaining
        if not remaining:
            return True
        remaining_types = set(remaining)

        # Count bots that can pick (have space) or deliver (have matching)
        bots_with_space = sum(1 for b in self.state.bots if len(b.inventory) < 3)
        bots_with_match = sum(
            1 for b in self.state.bots
            if any(inv in remaining_types for inv in b.inventory)
        )
        if bots_with_space == 0 and bots_with_match == 0:
            return False  # No bot can contribute

        # Estimate rounds for bots already carrying matching items to deliver
        delivery_rounds = 0
        items_covered = 0
        for bot in self.state.bots:
            matching = [inv for inv in bot.inventory if inv in remaining_types]
            if matching:
                d = self.distance(bot.position, self.nearest_drop_off(bot.position))
                delivery_rounds = max(delivery_rounds, d + 1)
                items_covered += len(matching)

        # Remaining items to pick up
        items_to_pick = max(0, len(remaining) - items_covered)
        available_bots = max(bots_with_space, 1)
        # Scale rounds-per-item with bot count: more bots = faster
        rounds_per_item = max(4, 8 // max(available_bots, 1))
        pickup_rounds = items_to_pick * rounds_per_item / available_bots

        # Drop-off sequencing: multiple bots delivering = queue
        delivering_bots = min(bots_with_match + (1 if items_to_pick > 0 else 0), len(self.state.bots))
        queue_overhead = max(0, delivering_bots - 1)  # 1 round per extra deliverer

        total_estimate = max(delivery_rounds, pickup_rounds) + queue_overhead
        return total_estimate <= self.rounds_remaining
