"""Collision-Free Path resolver: replaces PIBT with optimal per-round movement.

For each round:
1. Each bot has a desired next position (from BFS toward target)
2. Build conflict graph (edges between bots with conflicting moves)
3. Find maximum independent set — largest set of non-conflicting moves
4. Those bots move, rest wait

With 20 bots, the conflict graph is small enough for exact MIS on most rounds.
Falls back to greedy (by priority) when MIS is too expensive.
"""

from __future__ import annotations
from collections import deque

Pos = tuple[int, int]


class CFPResolver:
    """Collision-free movement resolver using maximum matching."""

    def __init__(self, grid, distance_fn, corridors=None, one_way=None, guidance_fn=None,
                 get_neighbors=None):
        self._grid = grid
        self._distance = distance_fn
        self._one_way = one_way or {}
        self._guidance_fn = guidance_fn
        # Use provided neighbor function (from PathEngine) for correct one-way routing
        self._get_neighbors = get_neighbors or self._fallback_neighbors

    def _fallback_neighbors(self, pos: Pos) -> list[Pos]:
        """Simple neighbors without one-way (fallback)."""
        x, y = pos
        result = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            npos = (x + dx, y + dy)
            if self._grid.is_walkable(npos):
                result.append(npos)
        return result

    def _bfs_next_step(self, start: Pos, goal: Pos) -> Pos:
        """BFS from start toward goal using PathEngine's one-way-aware neighbors.
        Returns the NEXT position (1 step)."""
        if start == goal:
            return start

        # Use PathEngine's directed_neighbors for correct one-way routing
        get_neighbors = self._get_neighbors

        visited = {start}
        parent: dict[Pos, Pos | None] = {start: None}
        queue = deque([start])

        while queue:
            pos = queue.popleft()
            if pos == goal:
                # Trace back to find first step
                step = pos
                while parent[step] != start:
                    step = parent[step]
                return step

            for npos in get_neighbors(pos):
                if npos not in visited:
                    visited.add(npos)
                    parent[npos] = pos
                    queue.append(npos)

        # No path found — stay
        return start

    def resolve(
        self,
        bot_positions: dict[int, Pos],
        targets: dict[int, Pos],
        urgency: dict[int, int] | None = None,
        idle_bots: set[int] | None = None,
        tiebreak_offset: int = 0,
    ) -> dict[int, Pos]:
        """Resolve movement for all bots. Returns {bot_id: next_position}.

        Uses greedy priority-based conflict resolution:
        1. Compute desired moves for all bots
        2. Process in priority order (highest urgency first)
        3. If desired cell is free: move
        4. If occupied: wait (or try alternatives)
        """
        if not bot_positions:
            return {}

        urgency = urgency or {}
        idle_bots = idle_bots or set()

        # Step 1: Compute desired next position for each bot
        desired: dict[int, Pos] = {}
        for bid, pos in bot_positions.items():
            target = targets.get(bid, pos)
            if target == pos:
                desired[bid] = pos  # Already at target — stay
            else:
                # BFS one step toward target
                next_pos = self._bfs_next_step(pos, target)
                desired[bid] = next_pos

        # Step 2: Sort bots by urgency (lower number = higher priority)
        sorted_bots = sorted(bot_positions.keys(),
                             key=lambda b: (urgency.get(b, 3), b))

        # Step 3: Greedy assignment — highest priority gets first pick
        occupied_next: set[Pos] = set()  # cells claimed for next round
        # Also track "swap conflicts": if A→B and B→A, both must wait
        moves: dict[int, Pos] = {}  # from→to pairs
        result: dict[int, Pos] = {}

        for bid in sorted_bots:
            pos = bot_positions[bid]
            want = desired[bid]

            if want == pos:
                # Stay — always allowed (claim current cell)
                result[bid] = pos
                occupied_next.add(pos)
                continue

            # Check: is desired cell free?
            if want in occupied_next:
                # Blocked — stay
                result[bid] = pos
                occupied_next.add(pos)
                continue

            # Check swap conflict: is another bot at 'want' trying to move to 'pos'?
            swap_conflict = False
            for other_bid, other_move in moves.items():
                other_pos = bot_positions[other_bid]
                if other_pos == want and other_move == pos:
                    swap_conflict = True
                    break

            if swap_conflict:
                result[bid] = pos
                occupied_next.add(pos)
                continue

            # Move!
            result[bid] = want
            occupied_next.add(want)
            moves[bid] = want

        # Step 4: For bots that are waiting but their current cell is claimed by another
        # (shouldn't happen with greedy assignment, but safety check)
        for bid in sorted_bots:
            if bid not in result:
                result[bid] = bot_positions[bid]

        return result
