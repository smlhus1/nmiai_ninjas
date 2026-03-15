"""
A* pathfinding with caching.

Performance budget: ~1ms per path on a 28x18 grid.
With 10 bots, that's ~10ms worst case (no cache hits).

Key design decisions:
- BFS distance map cached per destination (many bots go to same drop-off)
- A* for individual paths with dynamic obstacle avoidance
- Paths are lists of Pos, NOT actions — action conversion happens in ActionResolver
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Optional

from bot.models import Grid, Pos


def _one_way_allows(rule: tuple[int, int], dx: int, dy: int) -> bool:
    """Check if a movement (dx,dy) is allowed by a one-way rule.

    Rule (rx, ry): if rx != 0, horizontal movement must match rx.
                   if ry != 0, vertical movement must match ry.
    Movement on the perpendicular axis is always allowed.
    """
    # Vertical one-way: blocks vertical movement in wrong direction
    if rule[1] != 0 and dx == 0 and dy != 0:
        return dy == rule[1]
    # Horizontal one-way: blocks horizontal movement in wrong direction
    if rule[0] != 0 and dy == 0 and dx != 0:
        return dx == rule[0]
    return True


class PathEngine:
    """
    Pathfinding engine. Created once, grid cached between rounds.
    Call new_round() to clear per-round caches (dynamic obstacles).
    """

    def __init__(self) -> None:
        self._grid: Optional[Grid] = None
        # BFS distance maps: destination -> {pos: distance}
        self._distance_cache: dict[Pos, dict[Pos, int]] = {}
        # Dynamic obstacles for current round (other bot positions, etc.)
        self._obstacles: set[Pos] = set()
        # Corridor cells: 1-wide passages with exactly 2 collinear walkable neighbors
        self._corridors: frozenset[Pos] = frozenset()
        # One-way rules: pos -> allowed direction (dx, dy)
        # Vertical aisles: (0, 1) DOWN or (0, -1) UP
        # Horizontal corridors: (1, 0) RIGHT or (-1, 0) LEFT
        self._one_way: dict[Pos, tuple[int, int]] = {}
        self._one_way_enabled: bool = False
        self._strip_vertical_one_way: bool = False
        self._drop_off: Optional[Pos] = None

    def set_grid(self, grid: Grid, drop_off: Pos | None = None) -> None:
        """Update grid. Clears all caches if grid changed."""
        if drop_off is not None:
            self._drop_off = drop_off
        if self._grid != grid:
            self._grid = grid
            self._distance_cache.clear()
            self._corridors = self._detect_corridors(grid)
            if self._one_way_enabled:
                self._one_way = self._detect_one_way_aisles(grid, self._drop_off)
                if self._strip_vertical_one_way:
                    # Keep only horizontal rules — vertical aisles become bidirectional.
                    # Reduces path lengths (3x shorter to drop-off) at cost of
                    # occasional head-on collisions that PIBT resolves.
                    self._one_way = {
                        pos: d for pos, d in self._one_way.items() if d[1] == 0
                    }
            else:
                self._one_way = {}

    def enable_one_way(self, enabled: bool = True) -> None:
        """Enable/disable one-way aisle system. Call before set_grid."""
        self._one_way_enabled = enabled

    @property
    def corridors(self) -> frozenset[Pos]:
        """1-wide corridor cells detected at grid setup."""
        return self._corridors

    def new_round(self, obstacles: set[Pos] | None = None) -> None:
        """Reset per-round state. Call at start of each round."""
        self._obstacles = obstacles or set()

    def distance(self, start: Pos, end: Pos) -> int:
        """
        Get shortest distance ignoring dynamic obstacles.
        Uses cached BFS distance map from end position.
        Returns large number if unreachable.
        """
        if start == end:
            return 0

        if end not in self._distance_cache:
            self._distance_cache[end] = self._bfs_distances(end)

        return self._distance_cache[end].get(start, 9999)

    def manhattan(self, a: Pos, b: Pos) -> int:
        """Manhattan distance (fast heuristic, no wall awareness)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(
        self,
        start: Pos,
        end: Pos,
        extra_obstacles: set[Pos] | None = None,
    ) -> list[Pos]:
        """
        A* path from start to end, respecting walls and dynamic obstacles.
        Returns list of positions INCLUDING start, EXCLUDING if already there.
        Returns empty list if unreachable.
        """
        if start == end:
            return [start]

        grid = self._grid
        if grid is None:
            return []

        obstacles = self._obstacles
        if extra_obstacles:
            obstacles = obstacles | extra_obstacles

        # A* search
        # Priority queue: (f_score, counter, position)
        counter = 0
        open_set: list[tuple[int, int, Pos]] = [(0, counter, start)]
        came_from: dict[Pos, Pos] = {}
        g_score: dict[Pos, int] = {start: 0}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self._directed_neighbors(current):
                # Allow moving TO the end position even if it's an "obstacle"
                if neighbor != end and neighbor in obstacles:
                    continue

                tentative_g = g_score[current] + 1

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.manhattan(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

        return []  # No path found

    def _neighbors(self, pos: Pos) -> list[Pos]:
        """Get walkable neighbors of a position."""
        grid = self._grid
        if grid is None:
            return []

        x, y = pos
        result = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if grid.is_walkable((nx, ny)):
                result.append((nx, ny))
        return result

    @staticmethod
    def _detect_corridors(grid: Grid) -> frozenset[Pos]:
        """Detect 1-wide corridor cells (exactly 2 collinear walkable neighbors)."""
        corridors: set[Pos] = set()
        for x in range(grid.width):
            for y in range(grid.height):
                pos = (x, y)
                if not grid.is_walkable(pos):
                    continue
                neighbors = []
                for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    n = (x + dx, y + dy)
                    if grid.is_walkable(n):
                        neighbors.append(n)
                if len(neighbors) == 2:
                    # Check collinear: both same x (vertical corridor) or same y (horizontal)
                    if neighbors[0][0] == neighbors[1][0] or neighbors[0][1] == neighbors[1][1]:
                        corridors.add(pos)
        return frozenset(corridors)

    def _directed_neighbors(self, pos: Pos) -> list[Pos]:
        """Walkable neighbors respecting one-way rules.

        In a one-way cell, movement along the rule's axis is restricted
        to the allowed direction. Movement on the other axis is always OK.
        """
        if not self._one_way:
            return self._neighbors(pos)

        grid = self._grid
        if grid is None:
            return []

        x, y = pos
        rule = self._one_way.get(pos)
        result = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not grid.is_walkable((nx, ny)):
                continue
            if rule:
                if not _one_way_allows(rule, dx, dy):
                    continue
            result.append((nx, ny))
        return result

    def _reverse_neighbors(self, pos: Pos) -> list[Pos]:
        """Cells that can reach pos following one-way rules.

        Used in BFS from destination: "from which cells can I reach pos?"
        Checks if each neighbor is allowed to move TO pos under its own rule.
        """
        if not self._one_way:
            return self._neighbors(pos)

        grid = self._grid
        if grid is None:
            return []

        x, y = pos
        result = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if not grid.is_walkable((nx, ny)):
                continue
            # Check if the NEIGHBOR can move to pos (direction = -dx, -dy)
            neighbor = (nx, ny)
            rule = self._one_way.get(neighbor)
            if rule:
                # Neighbor moves (-dx, -dy) to reach pos
                if not _one_way_allows(rule, -dx, -dy):
                    continue
            result.append(neighbor)
        return result

    def _bfs_distances(self, origin: Pos) -> dict[Pos, int]:
        """BFS from origin to all reachable positions.

        Uses reverse neighbors when one-way is enabled, so distances[pos]
        gives the correct shortest path FROM pos TO origin following one-way rules.
        """
        grid = self._grid
        if grid is None:
            return {}

        distances: dict[Pos, int] = {origin: 0}
        queue = deque([origin])
        use_reverse = bool(self._one_way)

        while queue:
            pos = queue.popleft()
            d = distances[pos]

            neighbors = self._reverse_neighbors(pos) if use_reverse else self._neighbors(pos)
            for neighbor in neighbors:
                if neighbor not in distances:
                    distances[neighbor] = d + 1
                    queue.append(neighbor)

        return distances

    @staticmethod
    def _detect_one_way_aisles(
        grid: Grid,
        drop_off: Pos | None = None,
    ) -> dict[Pos, tuple[int, int]]:
        """Auto-detect aisles and assign one-way directions.

        Vertical aisles: alternating UP/DOWN based on drop-off position.
        Horizontal corridors: "keep right" rule where 2+ rows are adjacent.

        Detection algorithm:
        1. Find cross-corridors: rows where 60%+ cells are walkable
        2. Find aisle columns: walkable between cross-corridors, walls on sides
        3. Assign vertical directions optimized for drop-off traffic flow
        4. Assign horizontal keep-right on adjacent cross-corridor pairs
        """
        w, h = grid.width, grid.height

        # Step 1: Cross-corridors
        cross_rows: set[int] = set()
        for y in range(h):
            walkable = sum(1 for x in range(w) if grid.is_walkable((x, y)))
            if walkable >= w * 0.6:
                cross_rows.add(y)

        if not cross_rows:
            return {}

        # Step 2: Aisle columns (check only between outermost cross-corridors)
        min_cr = min(cross_rows)
        max_cr = max(cross_rows)
        aisle_columns: list[int] = []
        for x in range(1, w - 1):
            is_aisle = True
            non_corridor_cells = 0
            has_wall_neighbor = False
            for y in range(min_cr, max_cr + 1):
                if y in cross_rows:
                    continue
                if not grid.is_walkable((x, y)):
                    is_aisle = False
                    break
                non_corridor_cells += 1
                if not grid.is_walkable((x - 1, y)) or not grid.is_walkable((x + 1, y)):
                    has_wall_neighbor = True
            if is_aisle and non_corridor_cells >= 2 and has_wall_neighbor:
                aisle_columns.append(x)

        if not aisle_columns:
            return {}

        one_way: dict[Pos, tuple[int, int]] = {}

        # Step 3: Vertical aisle directions
        # Drop-off aware: nearest aisle to drop-off gets direction TOWARD
        # drop-off (return lane), next one gets AWAY (outbound lane), alternating.
        drop_x = drop_off[0] if drop_off else 0
        drop_y = drop_off[1] if drop_off else max_cr

        # Direction toward drop-off vertically
        toward_drop = 1 if drop_y > (min_cr + max_cr) / 2 else -1  # DOWN if drop at bottom

        # Sort aisles by distance to drop-off x
        sorted_aisles = sorted(aisle_columns, key=lambda ax: abs(ax - drop_x))

        # Nearest aisle = return lane (toward drop-off), then alternate
        aisle_directions: dict[int, int] = {}
        for i, ax in enumerate(sorted_aisles):
            if i % 2 == 0:
                aisle_directions[ax] = toward_drop      # Return lane
            else:
                aisle_directions[ax] = -toward_drop     # Outbound lane

        for x in aisle_columns:
            dy = aisle_directions[x]
            for y in range(min_cr, max_cr + 1):
                if y in cross_rows:
                    continue
                if grid.is_walkable((x, y)):
                    one_way[(x, y)] = (0, dy)

        # Step 4: Horizontal keep-right on adjacent cross-corridor rows
        # Find pairs of adjacent cross-corridor rows (2-wide horizontal corridors)
        sorted_crs = sorted(cross_rows)
        for i in range(len(sorted_crs) - 1):
            y_top = sorted_crs[i]
            y_bot = sorted_crs[i + 1]
            if y_bot - y_top != 1:
                continue  # Not adjacent

            # Keep-right: determine which direction is "outbound" vs "return"
            # based on drop-off position
            if drop_off:
                # Outbound = away from drop-off (higher row = further if drop at bottom)
                # Keep-right convention: outbound on bottom row, return on top row
                if drop_y >= y_bot:
                    # Drop-off at or below this pair: top=RIGHT (away), bottom=LEFT (return)
                    dir_top, dir_bot = (1, 0), (-1, 0)
                else:
                    # Drop-off above this pair: top=LEFT (return), bottom=RIGHT (away)
                    dir_top, dir_bot = (-1, 0), (1, 0)
            else:
                dir_top, dir_bot = (1, 0), (-1, 0)

            for x in range(1, w - 1):
                if grid.is_walkable((x, y_top)):
                    one_way[(x, y_top)] = dir_top
                if grid.is_walkable((x, y_bot)):
                    one_way[(x, y_bot)] = dir_bot

        # Step 5: Nightmare override (30x18, 20 bots)
        # Single horizontal corridors (not adjacent pairs) get LEFT direction
        # to create circular flow: pick up top/right, deliver bottom/left.
        # Vertical aisles swap direction (except outermost x=1 stays DOWN toward drop).
        if w >= 30 and h >= 18 and len(aisle_columns) >= 7:
            one_way = PathEngine._nightmare_one_way(
                grid, aisle_columns, cross_rows, drop_off,
            )

        import logging
        logging.getLogger(__name__).info(
            "One-way: aisles=%s dirs=%s, cross_rows=%s, %d directed cells",
            aisle_columns,
            {x: ('DOWN' if aisle_directions[x] == 1 else 'UP') for x in aisle_columns},
            sorted_crs,
            len(one_way),
        )
        return one_way

    @staticmethod
    def _nightmare_one_way(
        grid: Grid,
        aisle_columns: list[int],
        cross_rows: set[int],
        drop_off: Pos | None,
    ) -> dict[Pos, tuple[int, int]]:
        """Custom one-way for nightmare map (30x18, 20 bots, 3 drop-off zones).

        Keep-right system for multi-zone:
        - y=16: LEFT  — delivery lane (approach drop-offs from right)
        - y=15: RIGHT — return lane (exit after delivery, go to pickup aisles)
        - y=1:  RIGHT — return from top
        - y=9:  LEFT  — approach from top aisles

        This creates circulation in each zone:
        pick → DOWN to y=16 → LEFT to drop-off →
        UP to y=15 → RIGHT to pickup aisle → UP to shelves → repeat

        Vertical aisles alternate DOWN/UP to feed both corridors.
        """
        w, h = grid.width, grid.height
        one_way: dict[Pos, tuple[int, int]] = {}

        min_cr = min(cross_rows)
        max_cr = max(cross_rows)
        sorted_crs = sorted(cross_rows)

        # Horizontal highways: only restrict y=16 (delivery lane)
        # y=16: LEFT only — prevents head-on collisions on delivery approach
        # y=15: FREE (bidirectional) — escape lane, spawn dispersal
        # y=1, y=9: FREE — allow flexible routing in pickup area
        bottom_row = sorted_crs[-1]  # y=16
        for x in range(1, w - 1):
            if grid.is_walkable((x, bottom_row)):
                one_way[(x, bottom_row)] = (-1, 0)  # LEFT only

        # Vertical aisles: alternate DOWN/UP
        left_to_right = sorted(aisle_columns)
        for i, ax in enumerate(left_to_right):
            # Alternate: even index DOWN (feed y=16 delivery), odd UP (feed y=15 return)
            dy = 1 if (i % 2 == 0) else -1

            for y in range(min_cr, max_cr + 1):
                if y in cross_rows:
                    continue
                if grid.is_walkable((ax, y)):
                    one_way[(ax, y)] = (0, dy)

        return one_way
