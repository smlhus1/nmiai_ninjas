"""Time-Space A* pathfinding with reservation table.

Finds collision-free paths for multiple bots in a grid where
state = (x, y, timestep). No two bots can occupy the same cell
at the same timestep, and no two bots can swap positions.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

Pos = tuple[int, int]


class ReservationTable:
    """Tracks which (pos, timestep) are reserved by which bot."""

    def __init__(self):
        self._vertex: dict[tuple[Pos, int], int] = {}  # (pos, t) -> bot_id
        self._edge: set[tuple[Pos, Pos, int]] = set()   # (from, to, t) — swap prevention

    def reserve(self, bot_id: int, pos: Pos, t: int) -> None:
        self._vertex[(pos, t)] = bot_id

    def reserve_edge(self, from_pos: Pos, to_pos: Pos, t: int) -> None:
        self._edge.add((from_pos, to_pos, t))

    def is_free(self, pos: Pos, t: int, bot_id: int) -> bool:
        """Check if (pos, t) is free or reserved by same bot."""
        owner = self._vertex.get((pos, t))
        return owner is None or owner == bot_id

    def has_swap(self, from_pos: Pos, to_pos: Pos, t: int) -> bool:
        """Check if moving from_pos->to_pos at time t would cause a swap."""
        return (to_pos, from_pos, t) in self._edge

    def reserve_path(self, bot_id: int, path: list[Pos], start_t: int) -> None:
        """Reserve all positions along a path starting at start_t."""
        for i, pos in enumerate(path):
            t = start_t + i
            self.reserve(bot_id, pos, t)
            if i > 0:
                self.reserve_edge(path[i-1], pos, t)

    def clear_bot(self, bot_id: int) -> None:
        """Remove all reservations for a bot."""
        to_remove = [k for k, v in self._vertex.items() if v == bot_id]
        for k in to_remove:
            del self._vertex[k]


def time_space_astar(
    grid_neighbors: callable,  # (pos) -> list[Pos]
    start: Pos,
    goal: Pos,
    start_t: int,
    bot_id: int,
    reservations: ReservationTable,
    max_t: int = 500,
    heuristic: callable = None,  # (pos, goal) -> int (manhattan)
) -> list[Pos] | None:
    """Find shortest collision-free path from start to goal.

    Returns list of positions [start, ..., goal] or None if no path found.
    Each position corresponds to one timestep.
    """
    if heuristic is None:
        heuristic = lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1])

    # State: (pos, timestep)
    # Cost: number of timesteps
    open_set: list[tuple[int, int, Pos, int]] = []  # (f, g, pos, t)
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, start_t))

    came_from: dict[tuple[Pos, int], tuple[Pos, int]] = {}
    g_score: dict[tuple[Pos, int], int] = {(start, start_t): 0}

    while open_set:
        f, g, pos, t = heapq.heappop(open_set)

        if pos == goal:
            # Reconstruct path
            path = []
            state = (pos, t)
            while state in came_from:
                path.append(state[0])
                state = came_from[state]
            path.append(state[0])
            path.reverse()
            return path

        if t >= max_t:
            continue

        next_t = t + 1

        # Neighbors: move to adjacent cell OR wait
        candidates = grid_neighbors(pos) + [pos]

        for next_pos in candidates:
            # Check reservations
            if not reservations.is_free(next_pos, next_t, bot_id):
                continue

            # Check swap collision
            if next_pos != pos and reservations.has_swap(pos, next_pos, next_t):
                continue

            new_g = g + 1
            state_key = (next_pos, next_t)

            if state_key in g_score and g_score[state_key] <= new_g:
                continue

            g_score[state_key] = new_g
            f_score = new_g + heuristic(next_pos, goal)
            came_from[state_key] = (pos, t)
            heapq.heappush(open_set, (f_score, new_g, next_pos, next_t))

    return None  # No path found


def plan_bot_paths(
    grid_neighbors: callable,
    bot_starts: dict[int, Pos],
    bot_goals: list[tuple[int, Pos, str]],  # (bot_id, goal_pos, action_at_goal)
    max_t: int = 500,
) -> tuple[dict[int, list[Pos]], ReservationTable]:
    """Plan collision-free paths for multiple bots using prioritized planning.

    Bots are planned in order of bot_goals list. Earlier = higher priority.
    Each bot plans around already-reserved positions.

    Returns (bot_paths, reservation_table).
    """
    reservations = ReservationTable()
    bot_paths: dict[int, list[Pos]] = {}

    # Reserve starting positions for all bots at t=0
    for bot_id, start_pos in bot_starts.items():
        reservations.reserve(bot_id, start_pos, 0)

    for bot_id, goal, action in bot_goals:
        start = bot_starts.get(bot_id)
        if start is None:
            continue

        path = time_space_astar(
            grid_neighbors, start, goal, 0, bot_id, reservations, max_t,
        )

        if path:
            reservations.reserve_path(bot_id, path, 0)
            bot_paths[bot_id] = path
            # Update bot position for next planning
            bot_starts[bot_id] = path[-1]
        else:
            # No path found — bot stays at start
            bot_paths[bot_id] = [start] * max_t

    return bot_paths, reservations
