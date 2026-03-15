"""BFS-based distance computation and caching."""

from __future__ import annotations

from collections import deque

from .grid import Grid, Pos, pickup_positions


def bfs_distances(grid: Grid, origin: Pos) -> dict[Pos, int]:
    """Compute shortest distance from origin to all reachable walkable cells."""
    dist: dict[Pos, int] = {origin: 0}
    queue = deque([origin])

    while queue:
        pos = queue.popleft()
        d = dist[pos]
        for nb in grid.neighbors(pos, respect_one_way=False):
            if nb not in dist:
                dist[nb] = d + 1
                queue.append(nb)

    return dist


class DistanceCache:
    """Lazily computes and caches BFS distance maps keyed by origin."""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        self._cache: dict[Pos, dict[Pos, int]] = {}

    def _get_map(self, origin: Pos) -> dict[Pos, int]:
        if origin not in self._cache:
            self._cache[origin] = bfs_distances(self.grid, origin)
        return self._cache[origin]

    def distance(self, a: Pos, b: Pos) -> int | None:
        """Return shortest distance between a and b, or None if unreachable."""
        dist_map = self._get_map(a)
        return dist_map.get(b)

    def distance_from(self, origin: Pos) -> dict[Pos, int]:
        """Return full BFS distance map from origin (cached)."""
        return self._get_map(origin)


def nearest_pickup(grid: Grid, bot_pos: Pos, shelf_pos: Pos) -> Pos | None:
    """Return the pickup position (walkable neighbor of shelf) closest to bot_pos.

    Uses BFS from bot_pos. Returns None if no pickup positions are reachable.
    """
    positions = pickup_positions(grid, shelf_pos)
    if not positions:
        return None

    dist_map = bfs_distances(grid, bot_pos)

    best: Pos | None = None
    best_dist = float("inf")
    for p in positions:
        d = dist_map.get(p)
        if d is not None and d < best_dist:
            best_dist = d
            best = p

    return best
