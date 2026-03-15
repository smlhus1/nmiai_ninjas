"""Enhanced PIBT (EPIBT) — guidance as primary cost.

Key difference from V1 PIBT:
- Candidate scoring: guidance.edge_weight(current, p) + bfs_distance(p, target)
  This makes traffic flow the PRIMARY routing factor, not just a tiebreaker.
- Hindrance as secondary: among equal candidates, prefer non-blocking moves.
- 3-step lookahead for pickup actions (reserve cell for pickup + exit).
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from bot.models import Pos, Grid
from bot.v3.guidance import SoftGuidance

logger = logging.getLogger(__name__)


class EPIBTResolver:
    """Enhanced PIBT with guidance-weighted candidate selection."""

    def __init__(
        self,
        grid: Grid,
        distance_fn: Callable[[Pos, Pos], int],
        guidance: SoftGuidance | None = None,
        path_engine: object | None = None,
    ) -> None:
        self._grid = grid
        self._distance = distance_fn
        self._guidance = guidance
        self._path_engine = path_engine  # For one-way neighbor filtering

    def resolve(
        self,
        bots: dict[int, Pos],
        targets: dict[int, Pos],
        urgency: dict[int, int] | None = None,
        tiebreak_offset: int = 0,
        shelf_positions: frozenset[Pos] | None = None,
    ) -> dict[int, Pos]:
        """Compute collision-free next positions for all bots.

        Args:
            bots: bot_id -> current position
            targets: bot_id -> target position
            urgency: bot_id -> urgency tier (lower = higher priority)
            tiebreak_offset: round number for tie-breaking variation
            shelf_positions: shelf cells for 3-step lookahead
        """
        urgency = urgency or {}

        # Build bot density map (for 10+ bots)
        bot_density: dict[Pos, int] = {}
        if len(bots) >= 10:
            for pos in bots.values():
                bot_density[pos] = bot_density.get(pos, 0) + 1
                if len(bots) >= 20:
                    x, y = pos
                    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                        adj = (x + dx, y + dy)
                        bot_density[adj] = bot_density.get(adj, 0) + 1

        # Compute priorities: (tier, distance, tiebreak)
        priorities: dict[int, tuple[int, int, int]] = {}
        for bot_id, pos in bots.items():
            target = targets.get(bot_id, pos)
            d = self._distance(pos, target)
            tier = urgency.get(bot_id, 1)
            if pos == target:
                d = 9999
                if tier >= 0:  # Don't override negative (ESCAPE) priority
                    tier = max(tier, 3)
            priorities[bot_id] = (tier, d, (bot_id + tiebreak_offset) % 100)

        sorted_ids = sorted(priorities.keys(), key=lambda bid: priorities[bid])

        # State tracking
        claimed: dict[Pos, int] = {}
        for bot_id, pos in bots.items():
            claimed[pos] = bot_id
        result: dict[int, Pos] = {}
        decided: set[int] = set()

        use_hindrance = len(bots) >= 10
        guidance = self._guidance

        def _hindrance(pos: Pos, bot_id: int) -> int:
            if not use_hindrance:
                return 0
            h = 0
            px, py = pos
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                adj = (px + dx, py + dy)
                adj_bot = claimed.get(adj)
                if adj_bot is None or adj_bot == bot_id or adj_bot in decided:
                    continue
                adj_target = targets.get(adj_bot)
                if adj_target is None:
                    continue
                d_through = self._distance(adj, pos) + self._distance(pos, adj_target)
                d_direct = self._distance(adj, adj_target)
                if d_through <= d_direct + 1:
                    h += 1
            return h

        def _score(pos: Pos, current: Pos, target: Pos, bot_id: int) -> tuple:
            bfs = self._distance(pos, target)
            if guidance:
                g = guidance.edge_weight(current, pos)
            else:
                g = 1.0
            return (
                g + bfs,                        # primary: guided cost + BFS
                _hindrance(pos, bot_id),        # secondary: don't block
                pos != current,                 # tertiary: prefer staying (tie only)
                bot_density.get(pos, 0),        # quaternary: avoid crowds
            )

        def plan(bot_id: int, depth: int = 0) -> bool:
            if bot_id in decided:
                return True
            if depth > len(bots) + 2:
                result[bot_id] = bots[bot_id]
                claimed[bots[bot_id]] = bot_id
                decided.add(bot_id)
                return depth == 0

            current = bots[bot_id]
            target = targets.get(bot_id, current)

            neighbors = self._get_neighbors(current)
            candidates = sorted(
                neighbors + [current],
                key=lambda p: _score(p, current, target, bot_id),
            )

            for candidate in candidates:
                if candidate in claimed:
                    occupant = claimed[candidate]
                    if occupant == bot_id:
                        should_defer = (
                            (depth == 0 and current != target) or
                            (depth > 0 and current != target)
                        )
                        if should_defer:
                            continue
                        result[bot_id] = candidate
                        decided.add(bot_id)
                        return True

                    if occupant in decided:
                        continue

                    if priorities[bot_id] < priorities[occupant]:
                        if plan(occupant, depth + 1):
                            if candidate not in claimed or claimed[candidate] != occupant:
                                if claimed.get(current) == bot_id:
                                    del claimed[current]
                                claimed[candidate] = bot_id
                                result[bot_id] = candidate
                                decided.add(bot_id)
                                return True
                        continue
                    else:
                        continue

                # Free position
                if claimed.get(current) == bot_id:
                    del claimed[current]
                claimed[candidate] = bot_id
                result[bot_id] = candidate
                decided.add(bot_id)
                return True

            # Fallback: stay
            result[bot_id] = current
            claimed[current] = bot_id
            decided.add(bot_id)
            return True

        for bot_id in sorted_ids:
            if bot_id not in decided:
                plan(bot_id)

        # Post-process: cancel swaps
        for bid_a in list(result):
            if result[bid_a] == bots[bid_a]:
                continue
            for bid_b in list(result):
                if bid_b <= bid_a or result[bid_b] == bots[bid_b]:
                    continue
                if result[bid_a] == bots[bid_b] and result[bid_b] == bots[bid_a]:
                    result[bid_a] = bots[bid_a]
                    result[bid_b] = bots[bid_b]

        # Sequential ID-order collision resolution
        for _iteration in range(len(bots) + 1):
            cancelled = False
            for bid_a in sorted(result.keys()):
                if result[bid_a] == bots[bid_a]:
                    continue
                target_pos = result[bid_a]
                for bid_b in result:
                    if bid_b == bid_a:
                        continue
                    if bots[bid_b] != target_pos:
                        continue
                    bid_b_stays = (result[bid_b] == bots[bid_b])
                    bid_b_higher = (bid_b > bid_a and result[bid_b] != bots[bid_b])
                    if bid_b_stays or bid_b_higher:
                        result[bid_a] = bots[bid_a]
                        cancelled = True
                        break
            if not cancelled:
                break

        for bot_id in bots:
            if bot_id not in result:
                result[bot_id] = bots[bot_id]

        return result

    def _get_neighbors(self, pos: Pos) -> list[Pos]:
        """Get walkable neighbors, respecting one-way rules if PathEngine has them."""
        if self._path_engine is not None:
            return self._path_engine._directed_neighbors(pos)
        x, y = pos
        neighbors = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if self._grid.is_walkable((nx, ny)):
                neighbors.append((nx, ny))
        return neighbors
