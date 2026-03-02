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

    def __init__(self, state: GameState, path_engine: PathEngine) -> None:
        self.state = state
        self.path = path_engine

        # Index items by type for fast lookup.
        # Sort by position to ensure deterministic ordering regardless of
        # server item order. Without this, tie-breaking in route selection
        # depends on item list order, causing divergence between simulator
        # and live server (observed 44-105 score range from ordering alone).
        self._items_by_type: dict[str, list[Item]] = {}
        for item in sorted(state.items, key=lambda i: (i.position[0], i.position[1], i.id)):
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
        grid = self.path._grid if self.path._grid else self.state.grid
        result = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            pos = (item_pos[0] + dx, item_pos[1] + dy)
            if grid.is_walkable(pos):
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
        grid = self.path._grid if self.path._grid else self.state.grid
        result = []
        drop = self.state.drop_off
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            pos = (drop[0] + dx, drop[1] + dy)
            if grid.is_walkable(pos):
                result.append(pos)
        return result

    def staging_positions(self, distance: int = 2) -> list[Pos]:
        """Walkable cells near drop-off for staging (2-3 cells away)."""
        grid = self.path._grid if self.path._grid else self.state.grid
        result = []
        drop = self.state.drop_off
        for dx in range(-distance - 1, distance + 2):
            for dy in range(-distance - 1, distance + 2):
                pos = (drop[0] + dx, drop[1] + dy)
                manhattan = abs(dx) + abs(dy)
                if manhattan < 2 or manhattan > distance + 1:
                    continue
                if grid.is_walkable(pos):
                    result.append(pos)
        return result

    def parking_positions(self) -> list[Pos]:
        """Walkable cells far from aisles and drop-off, for idling out of the way.

        Prefers positions on the right side of the map (far from shelf aisles)
        to avoid blocking narrow corridors. Falls back to any wide-open area.
        """
        grid = self.path._grid if self.path._grid else self.state.grid
        w, h = grid.width, grid.height
        drop = self.state.drop_off

        # Prefer the right edge of the map (x >= w-4) — wide open corridor
        result = []
        for x in range(max(0, w - 4), w):
            for y in range(h):
                pos = (x, y)
                if pos == drop:
                    continue
                if pos[1] == drop[1] and abs(pos[0] - drop[0]) <= 3:
                    continue  # Skip near drop-off row
                if grid.is_walkable(pos):
                    result.append(pos)

        if not result:
            # Fallback: any walkable cell far from drop-off
            for x in range(w):
                for y in range(h):
                    pos = (x, y)
                    manhattan = abs(pos[0] - drop[0]) + abs(pos[1] - drop[1])
                    if manhattan >= 6 and grid.is_walkable(pos):
                        result.append(pos)

        return result if result else self.staging_positions(distance=3)

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

        Uses actual distances from available bots to items for accuracy.
        """
        active = self.state.active_orders
        if not active:
            return False
        remaining = list(active[0].items_remaining)
        if not remaining:
            return True

        # Check which bots already have matching items in inventory
        # Also track their delivery cost (must still reach drop-off)
        inventory_matches: list[int] = []  # delivery costs
        for bot in self.state.bots:
            matched = False
            for inv_item in bot.inventory:
                if inv_item in remaining:
                    remaining.remove(inv_item)
                    matched = True
            if matched:
                d_drop = self.distance(bot.position, self.state.drop_off) + 1
                inventory_matches.append(d_drop)
        if not remaining:
            # All items in inventory — check if bots can deliver in time
            if inventory_matches:
                # Sequential delivery (only 1 drop_off per round at drop-off)
                # Simulate: each bot delivers when it arrives, or queues behind
                inventory_matches.sort()
                total = inventory_matches[0]
                for i in range(1, len(inventory_matches)):
                    total = max(total + 1, inventory_matches[i])
                return total <= self.rounds_remaining
            return True

        # For each remaining item type, find best (closest available bot, closest item)
        # and compute the actual trip cost
        available_bots = [b for b in self.state.bots if len(b.inventory) < 3]
        if not available_bots:
            return False

        # Greedy: assign each remaining item to closest available bot
        trip_costs = []
        used_bots: set[int] = set()
        for item_type in remaining:
            best_cost = 9999
            for bot in available_bots:
                if bot.id in used_bots:
                    continue
                for item in self.items_of_type(item_type):
                    pp = self.best_pickup_position(bot.position, item.position)
                    if pp is None:
                        continue
                    d_pick = self.distance(bot.position, pp)
                    d_drop = self.distance(pp, self.state.drop_off)
                    cost = d_pick + d_drop + 2  # pick + drop actions
                    if cost < best_cost:
                        best_cost = cost
                        best_bot_id = bot.id
            if best_cost < 9999:
                trip_costs.append(best_cost)
                used_bots.add(best_bot_id)
            else:
                return False  # Can't reach this item type

        # Parallel execution: the bottleneck is the longest trip
        if not trip_costs:
            return False
        parallel_time = max(trip_costs)
        # Add margin for delivery queueing at drop-off
        parallel_time += max(0, len(trip_costs) - 1)
        return parallel_time <= self.rounds_remaining
