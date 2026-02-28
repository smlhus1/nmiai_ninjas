"""
PIBT (Priority Inheritance with Backtracking) collision resolution.

Replaces sequential collision avoidance with a cooperative algorithm
that eliminates deadlocks in narrow corridors.

Algorithm:
1. Assign dynamic priority: closer to target = higher priority, low ID breaks ties
2. Process bots in priority order
3. For each bot, try candidate positions (sorted by distance to goal)
4. If candidate is occupied by lower-priority bot, recursively push it away
5. If push fails, backtrack and try next candidate
6. Last resort: stay in place
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from bot.models import Pos, Grid

logger = logging.getLogger(__name__)


class PIBTResolver:
    """
    PIBT collision resolver. Given bots and their targets,
    returns collision-free next positions for all bots.
    """

    def __init__(self, grid: Grid, distance_fn: Callable[[Pos, Pos], int]) -> None:
        self._grid = grid
        self._distance = distance_fn

    def resolve(
        self,
        bots: dict[int, Pos],        # bot_id -> current position
        targets: dict[int, Pos],      # bot_id -> target position
        tiebreak_offset: int = 0,     # round number for tie-breaking variation
    ) -> dict[int, Pos]:
        """
        Compute collision-free next positions for all bots.

        Returns dict[bot_id, next_position].
        """
        # Compute priorities: (distance_to_target, bot_id) — lower = higher priority
        priorities: dict[int, tuple[int, int]] = {}
        for bot_id, pos in bots.items():
            target = targets.get(bot_id, pos)
            d = self._distance(pos, target)
            # Tiebreak: (bot_id + offset) % 100 so priority rotates by round
            priorities[bot_id] = (d, (bot_id + tiebreak_offset) % 100)

        # Sort by priority: closest to target first, then lowest ID
        sorted_ids = sorted(priorities.keys(), key=lambda bid: priorities[bid])

        # State tracking
        claimed: dict[Pos, int] = {}  # pos -> bot_id that claimed it
        result: dict[int, Pos] = {}
        decided: set[int] = set()

        # Recursive PIBT planning
        def plan(bot_id: int, depth: int = 0) -> bool:
            if bot_id in decided:
                return True
            if depth > len(bots) + 2:
                # Prevent infinite recursion
                result[bot_id] = bots[bot_id]
                claimed[bots[bot_id]] = bot_id
                decided.add(bot_id)
                return True

            current = bots[bot_id]
            target = targets.get(bot_id, current)

            # Generate candidates: neighbors + current, sorted by distance to target
            neighbors = self._get_neighbors(current)
            candidates = sorted(
                neighbors + [current],
                key=lambda p: (self._distance(p, target), p != current),
            )

            for candidate in candidates:
                if candidate in claimed:
                    occupant = claimed[candidate]
                    if occupant == bot_id:
                        # Already claimed by us (shouldn't happen, but safe)
                        result[bot_id] = candidate
                        decided.add(bot_id)
                        return True

                    # Occupied by already-decided bot — can't push
                    if occupant in decided:
                        continue

                    # Try to push the occupant away (priority inheritance)
                    if priorities[bot_id] < priorities[occupant]:
                        # We have higher priority — try to push
                        if plan(occupant, depth + 1):
                            # Occupant moved, claim the spot
                            if candidate not in claimed or claimed[candidate] != occupant:
                                # Occupant successfully moved away
                                claimed[candidate] = bot_id
                                result[bot_id] = candidate
                                decided.add(bot_id)
                                return True
                        # Push failed, try next candidate
                        continue
                    else:
                        # Lower priority than occupant, skip
                        continue

                # Position is free — claim it
                claimed[candidate] = bot_id
                result[bot_id] = candidate
                decided.add(bot_id)
                return True

            # All candidates failed — stay in place (absolute fallback)
            result[bot_id] = current
            claimed[current] = bot_id
            decided.add(bot_id)
            return True

        # Process all bots in priority order
        for bot_id in sorted_ids:
            if bot_id not in decided:
                plan(bot_id)

        # Ensure all bots have a position
        for bot_id in bots:
            if bot_id not in result:
                result[bot_id] = bots[bot_id]

        return result

    def _get_neighbors(self, pos: Pos) -> list[Pos]:
        """Get walkable neighbors of a position."""
        x, y = pos
        neighbors = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            candidate = (nx, ny)
            if self._grid.is_walkable(candidate):
                neighbors.append(candidate)
        return neighbors
