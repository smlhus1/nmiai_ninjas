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

    def __init__(
        self,
        grid: Grid,
        distance_fn: Callable[[Pos, Pos], int],
        corridors: frozenset[Pos] | None = None,
    ) -> None:
        self._grid = grid
        self._distance = distance_fn
        self._corridors = corridors or frozenset()

    def resolve(
        self,
        bots: dict[int, Pos],        # bot_id -> current position
        targets: dict[int, Pos],      # bot_id -> target position
        tiebreak_offset: int = 0,     # round number for tie-breaking variation
        idle_bots: set[int] | None = None,  # bots that should always get lowest priority
        high_priority_bots: set[int] | None = None,  # bots that get priority boost (deliverers)
    ) -> dict[int, Pos]:
        """
        Compute collision-free next positions for all bots.

        Returns dict[bot_id, next_position].
        """
        idle_bots = idle_bots or set()
        high_priority_bots = high_priority_bots or set()

        # Compute priorities: (distance_to_target, bot_id) — lower = higher priority
        priorities: dict[int, tuple[int, int]] = {}
        for bot_id, pos in bots.items():
            target = targets.get(bot_id, pos)
            d = self._distance(pos, target)
            if pos == target or bot_id in idle_bots:
                d = 9999  # IDLE bots get lowest priority so active bots can push them
            elif bot_id in high_priority_bots:
                d = max(0, d - 5)  # DELIVER bots get priority boost (closer = higher priority)
            # Tiebreak: (bot_id + offset) % 100 so priority rotates by round
            priorities[bot_id] = (d, (bot_id + tiebreak_offset) % 100)

        # Sort by priority: closest to target first, then lowest ID
        sorted_ids = sorted(priorities.keys(), key=lambda bid: priorities[bid])

        # State tracking — pre-claim current positions so PIBT knows occupancy.
        # Without this, bots can produce swap moves (A→B, B→A) which the game
        # engine blocks (sequential ID-order resolution), causing permanent deadlock.
        claimed: dict[Pos, int] = {}
        for bot_id, pos in bots.items():
            claimed[pos] = bot_id  # Last bot wins at shared positions (spawn stacking)
        result: dict[int, Pos] = {}
        decided: set[int] = set()

        # Recursive PIBT planning
        def plan(bot_id: int, depth: int = 0) -> bool:
            if bot_id in decided:
                return True
            if depth > len(bots) + 2:
                # Prevent infinite recursion — stay in place
                result[bot_id] = bots[bot_id]
                claimed[bots[bot_id]] = bot_id
                decided.add(bot_id)
                return depth == 0  # Only "success" at top level

            current = bots[bot_id]
            target = targets.get(bot_id, current)

            # Generate candidates sorted by distance to target
            # Corridor penalty as tiebreak only — never prevents movement
            neighbors = self._get_neighbors(current)
            corridor_set = self._corridors
            candidates = sorted(
                neighbors + [current],
                key=lambda p: (
                    self._distance(p, target),
                    p != current,
                    1 if p in corridor_set else 0,
                ),
            )

            for candidate in candidates:
                if candidate in claimed:
                    occupant = claimed[candidate]
                    if occupant == bot_id:
                        # Own position — defer staying if we should try alternatives:
                        # 1. Active bot at depth 0: explore all neighbors before giving up
                        # 2. IDLE bot being pushed: try to vacate for the pusher
                        should_defer = (
                            (depth == 0 and current != target) or
                            (depth > 0 and priorities[bot_id][0] >= 9999)
                        )
                        if should_defer:
                            continue  # Try other candidates first
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
                                # Occupant successfully moved away — take the spot
                                if claimed.get(current) == bot_id:
                                    del claimed[current]  # Release our old position
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
                if claimed.get(current) == bot_id:
                    del claimed[current]  # Release our old position
                claimed[candidate] = bot_id
                result[bot_id] = candidate
                decided.add(bot_id)
                return True

            # Fallback: stay in place
            result[bot_id] = current
            claimed[current] = bot_id
            decided.add(bot_id)
            return True

        # Process all bots in priority order
        for bot_id in sorted_ids:
            if bot_id not in decided:
                plan(bot_id)

        # Post-process: detect and cancel swaps
        # (Sequential ID-order resolution means swaps always fail in-game)
        for bid_a in list(result):
            if result[bid_a] == bots[bid_a]:
                continue
            for bid_b in list(result):
                if bid_b <= bid_a or result[bid_b] == bots[bid_b]:
                    continue
                if result[bid_a] == bots[bid_b] and result[bid_b] == bots[bid_a]:
                    logger.info("PIBT: cancelled swap between bot %d and %d", bid_a, bid_b)
                    result[bid_a] = bots[bid_a]
                    result[bid_b] = bots[bid_b]

        # Post-process: detect and cancel follow-collisions with IDLE bots
        # Game engine processes moves in bot ID order. If a lower-ID bot
        # moves to a higher-ID IDLE bot's current position, the IDLE bot
        # hasn't moved yet when the active bot's move is resolved → collision.
        # Cancel the active bot's move; next round the path will be clear.
        # Only applies to IDLE bots — active-active follows resolve naturally.
        for bid_a in list(result):
            if result[bid_a] == bots[bid_a]:
                continue  # Not moving
            for bid_b in list(result):
                if bid_b == bid_a:
                    continue
                # Lower-ID bot moving to higher-ID IDLE bot's current position
                if (bid_a < bid_b and result[bid_a] == bots[bid_b]
                        and bid_b in idle_bots and result[bid_b] != bots[bid_b]):
                    logger.debug(
                        "PIBT: cancelled follow-collision bot %d -> idle bot %d's pos %s",
                        bid_a, bid_b, bots[bid_b],
                    )
                    result[bid_a] = bots[bid_a]

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
