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
        one_way: dict[Pos, tuple[int, int]] | None = None,
    ) -> None:
        self._grid = grid
        self._distance = distance_fn
        self._corridors = corridors or frozenset()
        self._one_way = one_way or {}

    def resolve(
        self,
        bots: dict[int, Pos],        # bot_id -> current position
        targets: dict[int, Pos],      # bot_id -> target position
        tiebreak_offset: int = 0,     # round number for tie-breaking variation
        idle_bots: set[int] | None = None,  # bots that should always get lowest priority
        urgency: dict[int, int] | None = None,  # bot_id -> urgency tier (0=highest, 1=mid, 2=low)
    ) -> dict[int, Pos]:
        """
        Compute collision-free next positions for all bots.

        Returns dict[bot_id, next_position].
        """
        idle_bots = idle_bots or set()
        urgency = urgency or {}

        # Compute priorities: (urgency_tier, distance_to_target, tiebreak) — lower = higher priority
        priorities: dict[int, tuple[int, int, int]] = {}
        for bot_id, pos in bots.items():
            target = targets.get(bot_id, pos)
            d = self._distance(pos, target)
            tier = urgency.get(bot_id, 1)  # Default: mid priority
            if pos == target or bot_id in idle_bots:
                d = 9999  # IDLE bots get lowest priority so active bots can push them
                tier = 3
            # Tiebreak: (bot_id + offset) % 100 so priority rotates by round
            priorities[bot_id] = (tier, d, (bot_id + tiebreak_offset) % 100)

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
                        # 2. Pushed bot (depth > 0): yield unless AT target.
                        #    Prevents deadlocks where a higher-priority bot
                        #    can't push through because pushed bots refuse to move.
                        should_defer = (
                            (depth == 0 and current != target) or
                            (depth > 0 and (priorities[bot_id][0] >= 9999 or current != target))
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

        # Post-process: simulate game engine's sequential ID-order resolution.
        # Iterate until no more cancellations needed (handles cascades).
        for _iteration in range(len(bots) + 1):
            cancelled = False
            for bid_a in sorted(result.keys()):
                if result[bid_a] == bots[bid_a]:
                    continue  # Not moving
                target_pos = result[bid_a]
                # Check: is target_pos occupied by a bot that will still be
                # there when bid_a's move resolves?
                for bid_b in result:
                    if bid_b == bid_a:
                        continue
                    if bots[bid_b] != target_pos:
                        continue  # bid_b wasn't at target_pos
                    # bid_b is/was at target_pos. Will bid_b still be there?
                    # Case 1: bid_b is not moving (stays) → collision
                    # Case 2: bid_b has higher ID and is moving away →
                    #          game processes bid_a first, bid_b hasn't moved → collision
                    bid_b_stays = (result[bid_b] == bots[bid_b])
                    bid_b_higher_and_moving = (bid_b > bid_a and result[bid_b] != bots[bid_b])
                    if bid_b_stays or bid_b_higher_and_moving:
                        logger.debug(
                            "PIBT: cancel collision B%d -> %s (B%d %s)",
                            bid_a, target_pos, bid_b,
                            "stays" if bid_b_stays else "higher-ID moving",
                        )
                        result[bid_a] = bots[bid_a]
                        cancelled = True
                        break
            if not cancelled:
                break

        # Ensure all bots have a position
        for bot_id in bots:
            if bot_id not in result:
                result[bot_id] = bots[bot_id]

        return result

    def _get_neighbors(self, pos: Pos) -> list[Pos]:
        """Get walkable neighbors respecting one-way rules."""
        x, y = pos
        rule = self._one_way.get(pos)
        neighbors = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not self._grid.is_walkable((nx, ny)):
                continue
            if rule:
                # Vertical one-way: blocks wrong vertical direction
                if rule[1] != 0 and dx == 0 and dy != 0 and dy != rule[1]:
                    continue
                # Horizontal one-way: blocks wrong horizontal direction
                if rule[0] != 0 and dy == 0 and dx != 0 and dx != rule[0]:
                    continue
            neighbors.append((nx, ny))
        return neighbors
