"""Execution engine: converts scheduled trips into round-by-round actions.

Takes the high-level plan (trips with timing) and produces concrete
move/pick/drop actions per bot per round, with PIBT-style collision avoidance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from .grid import GameMap, Grid, Pos
from .pathfinding import DistanceCache, bfs_distances


class Action(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    PICK_UP = "pick_up"
    DROP_OFF = "drop_off"
    WAIT = "wait"


def direction_action(from_pos: Pos, to_pos: Pos) -> Action:
    """Get the movement action to go from from_pos to to_pos (adjacent)."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dx == 1:
        return Action.RIGHT
    elif dx == -1:
        return Action.LEFT
    elif dy == 1:
        return Action.DOWN
    elif dy == -1:
        return Action.UP
    return Action.WAIT


@dataclass
class BotExecState:
    """Runtime state for one bot during execution."""

    bot_id: int
    pos: Pos
    inventory: list[str] = field(default_factory=list)
    path: list[Pos] = field(default_factory=list)  # remaining path (next positions)
    current_goal: str = "idle"  # "pickup", "deliver", "idle"
    target_item: str | None = None
    target_pos: Pos | None = None


class Executor:
    """Converts a schedule into round-by-round actions with collision avoidance."""

    def __init__(self, game_map: GameMap, dist_cache: DistanceCache) -> None:
        self.game_map = game_map
        self.grid = game_map.grid
        self.dist_cache = dist_cache

    def compute_path(self, start: Pos, end: Pos) -> list[Pos]:
        """BFS shortest path from start to end. Returns list of positions (excluding start)."""
        if start == end:
            return []

        parent: dict[Pos, Pos] = {start: start}
        queue = deque([start])

        while queue:
            pos = queue.popleft()
            if pos == end:
                # Reconstruct
                path = []
                current = end
                while current != start:
                    path.append(current)
                    current = parent[current]
                path.reverse()
                return path

            for nb in self.grid.neighbors(pos):
                if nb not in parent:
                    parent[nb] = pos
                    queue.append(nb)

        return []  # unreachable

    def resolve_collisions(
        self,
        bots: list[BotExecState],
        desired_positions: dict[int, Pos],
    ) -> dict[int, Pos]:
        """Resolve collisions using ID-priority (lower ID wins).

        Rules:
        - Two bots can't occupy same cell
        - Two bots can't swap positions
        - Lower bot_id has priority
        """
        # Process in ID order (lower ID = higher priority)
        final: dict[int, Pos] = {}
        occupied: set[Pos] = set()
        # Track original positions for swap detection
        original: dict[int, Pos] = {b.bot_id: b.pos for b in bots}

        sorted_bots = sorted(bots, key=lambda b: b.bot_id)

        for bot in sorted_bots:
            desired = desired_positions.get(bot.bot_id, bot.pos)

            # Check vertex collision
            if desired in occupied:
                desired = bot.pos  # stay put

            # Check swap collision
            if desired != bot.pos:
                for other_id, other_final in final.items():
                    other_original = original[other_id]
                    if desired == other_original and other_final == bot.pos:
                        desired = bot.pos  # swap detected, stay put
                        break

            # Check if staying put but our position is taken
            if desired == bot.pos and desired in occupied:
                # We're stuck — this shouldn't happen often
                desired = bot.pos  # still stay put, collision resolver will handle

            final[bot.bot_id] = desired
            occupied.add(desired)

        return final

    def execute_round(
        self,
        bots: list[BotExecState],
        active_order_items: set[str] | None = None,
    ) -> dict[int, Action]:
        """Execute one round: compute desired moves and resolve collisions.

        Returns action per bot.
        """
        desired: dict[int, Pos] = {}
        pre_actions: dict[int, Action] = {}  # pickup/dropoff actions override movement

        for bot in bots:
            # Check for pickup action
            if bot.current_goal == "pickup" and bot.pos == bot.target_pos:
                pre_actions[bot.bot_id] = Action.PICK_UP
                desired[bot.bot_id] = bot.pos
                continue

            # Check for dropoff action
            if bot.current_goal == "deliver" and bot.pos in self.game_map.drop_off_zones:
                if bot.inventory and active_order_items:
                    # Only drop if we have items matching active order
                    if any(item in active_order_items for item in bot.inventory):
                        pre_actions[bot.bot_id] = Action.DROP_OFF
                        desired[bot.bot_id] = bot.pos
                        continue

            # Movement: follow path
            if bot.path:
                next_pos = bot.path[0]
                desired[bot.bot_id] = next_pos
            else:
                desired[bot.bot_id] = bot.pos  # wait

        # Resolve collisions
        final_positions = self.resolve_collisions(bots, desired)

        # Generate actions
        actions: dict[int, Action] = {}
        for bot in bots:
            if bot.bot_id in pre_actions:
                actions[bot.bot_id] = pre_actions[bot.bot_id]
            else:
                final_pos = final_positions[bot.bot_id]
                if final_pos == bot.pos:
                    actions[bot.bot_id] = Action.WAIT
                else:
                    actions[bot.bot_id] = direction_action(bot.pos, final_pos)
                    # Advance path
                    if bot.path and bot.path[0] == final_pos:
                        bot.path.pop(0)

            # Update position
            new_pos = final_positions[bot.bot_id]
            bot.pos = new_pos

        return actions
