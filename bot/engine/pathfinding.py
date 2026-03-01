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

    def set_grid(self, grid: Grid) -> None:
        """Update grid. Clears all caches if grid changed."""
        if self._grid != grid:
            self._grid = grid
            self._distance_cache.clear()
            self._corridors = self._detect_corridors(grid)

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

            for neighbor in self._neighbors(current):
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

    def _bfs_distances(self, origin: Pos) -> dict[Pos, int]:
        """BFS from origin to all reachable positions. Ignores dynamic obstacles."""
        grid = self._grid
        if grid is None:
            return {}

        distances: dict[Pos, int] = {origin: 0}
        queue = deque([origin])

        while queue:
            pos = queue.popleft()
            d = distances[pos]

            for neighbor in self._neighbors(pos):
                if neighbor not in distances:
                    distances[neighbor] = d + 1
                    queue.append(neighbor)

        return distances
