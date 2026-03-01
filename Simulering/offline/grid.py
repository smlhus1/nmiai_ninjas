"""
Offline optimization toolkit for NM i AI Grocery Bot.
Core utilities: grid, BFS, A* — zero external dependencies.

Drop this entire 'offline/' folder into your project root.
"""
from __future__ import annotations
from collections import deque
from typing import Optional

Pos = tuple[int, int]

MOVES = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}

def move_action(fr: Pos, to: Pos) -> str:
    dx, dy = to[0] - fr[0], to[1] - fr[1]
    if dx == 1: return "move_right"
    if dx == -1: return "move_left"
    if dy == 1: return "move_down"
    if dy == -1: return "move_up"
    return "wait"


class Grid:
    """Immutable grid with BFS distance cache."""

    def __init__(self, width: int, height: int,
                 walls: set[Pos], shelves: set[Pos]):
        self.width = width
        self.height = height
        self.blocked = walls | shelves
        self.walls = walls
        self.shelves = shelves
        self._bfs_cache: dict[Pos, dict[Pos, int]] = {}
        self._path_cache: dict[tuple[Pos, Pos], list[Pos]] = {}

    def in_bounds(self, p: Pos) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def walkable(self, p: Pos) -> bool:
        return self.in_bounds(p) and p not in self.blocked

    def neighbors(self, p: Pos) -> list[Pos]:
        result = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            n = (p[0] + dx, p[1] + dy)
            if self.walkable(n):
                result.append(n)
        return result

    def bfs_from(self, start: Pos) -> dict[Pos, int]:
        """Full BFS from start, cached."""
        if start in self._bfs_cache:
            return self._bfs_cache[start]
        dist = {start: 0}
        q = deque([start])
        while q:
            pos = q.popleft()
            for n in self.neighbors(pos):
                if n not in dist:
                    dist[n] = dist[pos] + 1
                    q.append(n)
        self._bfs_cache[start] = dist
        return dist

    def distance(self, a: Pos, b: Pos) -> int:
        """BFS shortest distance. Returns 9999 if unreachable."""
        dists = self.bfs_from(b)  # cache from destination (many sources query same dest)
        return dists.get(a, 9999)

    def find_path(self, start: Pos, end: Pos) -> list[Pos]:
        """BFS shortest path, returns list of positions (excluding start)."""
        key = (start, end)
        if key in self._path_cache:
            return self._path_cache[key]
        if start == end:
            return []
        prev: dict[Pos, Optional[Pos]] = {start: None}
        q = deque([start])
        while q:
            pos = q.popleft()
            if pos == end:
                path = []
                cur = end
                while cur != start:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                self._path_cache[key] = path
                return path
            for n in self.neighbors(pos):
                if n not in prev:
                    prev[n] = pos
                    q.append(n)
        self._path_cache[key] = []
        return []

    def adjacent_walkable(self, shelf_pos: Pos) -> list[Pos]:
        """Walkable cells adjacent to a shelf (pickup positions)."""
        result = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            n = (shelf_pos[0] + dx, shelf_pos[1] + dy)
            if self.walkable(n):
                result.append(n)
        return result

    def best_pickup_pos(self, bot_pos: Pos, shelf_pos: Pos) -> Optional[Pos]:
        """Nearest walkable cell adjacent to shelf, relative to bot."""
        candidates = self.adjacent_walkable(shelf_pos)
        if not candidates:
            return None
        return min(candidates, key=lambda c: self.distance(bot_pos, c))

    def precompute_all_bfs(self):
        """Precompute BFS from every walkable cell. ~50ms for 28x18."""
        for x in range(self.width):
            for y in range(self.height):
                p = (x, y)
                if self.walkable(p):
                    self.bfs_from(p)

    @classmethod
    def from_game_state(cls, state: dict) -> "Grid":
        """Build grid from a game_state JSON dict."""
        g = state["grid"]
        walls = {tuple(w) for w in g["walls"]}
        # Items on shelves = shelf positions (non-walkable)
        shelves = {tuple(i["position"]) for i in state["items"]}
        return cls(g["width"], g["height"], walls, shelves)
