"""
WorldModel: enriched, queryable view of the game state.

This is the "brain's whiteboard" — it takes raw GameState and provides
higher-level queries like "what items are available for this order?"
and "which bot is closest to this item?".

Created fresh each round from GameState + persistent PathEngine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

    def __init__(self, state: GameState, path_engine: PathEngine) -> None:
        self.state = state
        self.path = path_engine

        # Index items by type for fast lookup
        self._items_by_type: dict[str, list[Item]] = {}
        for item in state.items:
            self._items_by_type.setdefault(item.type, []).append(item)

        # Bot positions set (for obstacle awareness)
        self._bot_positions: set[Pos] = {b.position for b in state.bots}

    @property
    def rounds_remaining(self) -> int:
        return self.state.rounds_remaining

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
            if self.state.grid.is_walkable(pos):
                result.append(pos)
        return result

    def best_pickup_position(self, bot_pos: Pos, item_pos: Pos) -> Pos | None:
        """Nearest walkable cell adjacent to item, from bot's perspective."""
        positions = self.pickup_positions(item_pos)
        if not positions:
            return None
        return min(positions, key=lambda p: self.path.distance(bot_pos, p))

    def distance(self, a: Pos, b: Pos) -> int:
        """Pathfinding distance between two positions."""
        return self.path.distance(a, b)

    def nearest_item(self, bot: Bot, item_type: str) -> Optional[ItemAvailability]:
        """Find the nearest item of a type to a bot, with trip cost info."""
        items = self.items_of_type(item_type)
        if not items:
            return None

        best: Optional[ItemAvailability] = None
        for item in items:
            pickup_pos = self.best_pickup_position(bot.position, item.position)
            if pickup_pos is None:
                continue
            d_bot = self.path.distance(bot.position, pickup_pos)
            d_drop = self.path.distance(pickup_pos, self.state.drop_off)
            candidate = ItemAvailability(item, d_bot, d_drop, pickup_pos)
            if best is None or candidate.total_trip < best.total_trip:
                best = candidate

        return best

    def can_complete_trip(self, bot: Bot, item_pos: Pos) -> bool:
        """Check if a bot can pick up an item and deliver it before game ends."""
        pickup_pos = self.best_pickup_position(bot.position, item_pos)
        if pickup_pos is None:
            return False
        d_to_pickup = self.path.distance(bot.position, pickup_pos)
        d_to_drop = self.path.distance(pickup_pos, self.state.drop_off)
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
        """Walkable cells adjacent to the drop-off zone."""
        result = []
        drop = self.state.drop_off
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            pos = (drop[0] + dx, drop[1] + dy)
            if self.state.grid.is_walkable(pos):
                result.append(pos)
        return result

    def staging_positions(self, distance: int = 2) -> list[Pos]:
        """Walkable cells near drop-off for staging (2-3 cells away)."""
        result = []
        drop = self.state.drop_off
        for dx in range(-distance - 1, distance + 2):
            for dy in range(-distance - 1, distance + 2):
                pos = (drop[0] + dx, drop[1] + dy)
                manhattan = abs(dx) + abs(dy)
                if manhattan < 2 or manhattan > distance + 1:
                    continue
                if self.state.grid.is_walkable(pos):
                    result.append(pos)
        return result

    def bot_positions_except(self, bot_id: int) -> set[Pos]:
        """Positions of all bots except the given one."""
        return {b.position for b in self.state.bots if b.id != bot_id}

    def is_endgame(self, threshold: int = 40) -> bool:
        """True if we're in the last `threshold` rounds."""
        return self.rounds_remaining <= threshold

    def can_complete_active_order(self) -> bool:
        """Estimate whether the active order can be completed in remaining rounds."""
        active = self.state.active_orders
        if not active:
            return False
        remaining = active[0].items_remaining
        if not remaining:
            return True
        # Rough estimate: each item needs ~8 rounds (pick + navigate + deliver)
        return len(remaining) * 8 <= self.rounds_remaining
