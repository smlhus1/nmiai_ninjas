"""Time-Space A* pathfinding with reservation table.

Finds collision-free paths for multiple bots in a grid where
state = (x, y, timestep). No two bots can occupy the same cell
at the same timestep, and no two bots can swap positions.

ReservationTable uses SET of bot_ids per (pos, t) — supports
multi-occupancy at spawn and other shared positions.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

Pos = tuple[int, int]


class ReservationTable:
    """Tracks which (pos, timestep) are occupied by which bots.

    Uses set[int] per (pos, t) to support multi-occupancy at spawn.
    """

    def __init__(self):
        self._vertex: dict[tuple[Pos, int], set[int]] = {}  # (pos, t) -> {bot_ids}
        self._edge: set[tuple[Pos, Pos, int]] = set()

    def reserve(self, bot_id: int, pos: Pos, t: int) -> None:
        key = (pos, t)
        if key not in self._vertex:
            self._vertex[key] = set()
        self._vertex[key].add(bot_id)

    def reserve_edge(self, from_pos: Pos, to_pos: Pos, t: int) -> None:
        self._edge.add((from_pos, to_pos, t))

    def is_free(self, pos: Pos, t: int, bot_id: int) -> bool:
        """Check if (pos, t) is free for bot_id.

        Free if: no other bot is there, OR only bot_id itself is there.
        """
        occupants = self._vertex.get((pos, t))
        if occupants is None:
            return True
        # Free if the only occupant is ourselves
        return occupants <= {bot_id}

    def has_swap(self, from_pos: Pos, to_pos: Pos, t: int) -> bool:
        return (to_pos, from_pos, t) in self._edge

    def reserve_path(self, bot_id: int, path: list[Pos], start_t: int) -> None:
        """Reserve path AND mark departure cells.

        When bot is at path[i] at time t and moves to path[i+1] at t+1,
        we reserve path[i] at BOTH t and t+1. This ensures that other bots
        planned later won't try to enter path[i] at t+1 thinking it's free
        (in sequential sim, the bot is still there until it actually moves).
        """
        for i, pos in enumerate(path):
            t = start_t + i
            self.reserve(bot_id, pos, t)
        # Also reserve departure cells at t+1 (sequential collision guard)
        for i in range(len(path) - 1):
            if path[i] != path[i+1]:  # Bot is actually moving
                t = start_t + i + 1
                self.reserve(bot_id, path[i], t)

    def reserve_stay(self, bot_id: int, pos: Pos, from_t: int, to_t: int) -> None:
        """Reserve a bot staying at pos from from_t to to_t (inclusive)."""
        for t in range(from_t, to_t + 1):
            self.reserve(bot_id, pos, t)

    def clear_bot(self, bot_id: int) -> None:
        """Remove all reservations for a bot."""
        to_clean = []
        for key, occupants in self._vertex.items():
            if bot_id in occupants:
                to_clean.append(key)
        for key in to_clean:
            self._vertex[key].discard(bot_id)
            if not self._vertex[key]:
                del self._vertex[key]

    def occupant_count(self, pos: Pos, t: int) -> int:
        occupants = self._vertex.get((pos, t))
        return len(occupants) if occupants else 0


def time_space_astar(
    grid_neighbors: callable,
    start: Pos,
    goal: Pos,
    start_t: int,
    bot_id: int,
    reservations: ReservationTable,
    max_t: int = 500,
    heuristic: callable = None,
) -> list[Pos] | None:
    """Find shortest collision-free path from start to goal.

    Returns list of positions [start, ..., goal] or None if no path found.
    Each position corresponds to one timestep.
    """
    if heuristic is None:
        heuristic = lambda a, b: abs(a[0]-b[0]) + abs(a[1]-b[1])

    open_set: list[tuple[int, int, Pos, int]] = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, start_t))

    came_from: dict[tuple[Pos, int], tuple[Pos, int]] = {}
    g_score: dict[tuple[Pos, int], int] = {(start, start_t): 0}

    while open_set:
        f, g, pos, t = heapq.heappop(open_set)

        if pos == goal:
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
        candidates = grid_neighbors(pos) + [pos]  # neighbors + wait

        for next_pos in candidates:
            if not reservations.is_free(next_pos, next_t, bot_id):
                continue
            # Swap check disabled — sim's sequential model allows swaps
            # (bot A moves first, then bot B can move to A's old cell)
            # if next_pos != pos and reservations.has_swap(pos, next_pos, next_t):
            #     continue

            new_g = g + 1
            state_key = (next_pos, next_t)

            if state_key in g_score and g_score[state_key] <= new_g:
                continue

            g_score[state_key] = new_g
            f_score = new_g + heuristic(next_pos, goal)
            came_from[state_key] = (pos, t)
            heapq.heappush(open_set, (f_score, new_g, next_pos, next_t))

    return None
