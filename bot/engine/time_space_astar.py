"""
Time-Space A*: pathfinding in (x, y, t) space with reservation table.

Finds shortest path from start to goal while avoiding:
- Static obstacles (walls, shelves)
- Reserved cells (other bots' planned paths)
- Edge/swap conflicts (two bots swapping positions)

Supports WAIT moves (staying in place costs 1 timestep).
Uses BFS distance as heuristic (admissible + consistent).
"""

from __future__ import annotations

import heapq
from typing import Callable

from bot.models import Grid, Pos
from bot.engine.reservation import ReservationTable


def find_path_tsa(
    start: Pos,
    goal: Pos,
    start_t: int,
    grid: Grid,
    reservations: ReservationTable,
    bot_id: int,
    directed_neighbors_fn: Callable[[Pos], list[Pos]],
    distance_fn: Callable[[Pos, Pos], int],
    max_t: int = 50,
    deadline_ms: float = 0,
    goal_hold: int = 0,
    sequential_mode: bool = False,
) -> list[Pos] | None:
    """
    Find collision-free path from start to goal in time-space.

    Args:
        start: Starting position
        goal: Goal position
        start_t: Current timestep
        grid: Game grid (for walkability)
        reservations: Space-time reservation table
        bot_id: ID of the bot being planned (own reservations don't block)
        directed_neighbors_fn: Function returning walkable neighbors (respects one-way)
        distance_fn: BFS distance function (heuristic)
        max_t: Maximum timestep to search up to
        deadline_ms: Time budget in ms (0 = unlimited)
        goal_hold: Number of extra rounds the goal must be free after arrival.
                   Use 2 for pickups (action + next segment start), 1 for dropoff.
        sequential_mode: If True, skip edge/swap and following conflict checks.
                   Use for offline MAPF planning in ID order where the game
                   processes bots sequentially (lower ID first). Lower-ID bots
                   move before higher-ID bots, so edge conflicts with
                   already-planned lower-ID bots are not real conflicts.

    Returns:
        List of positions [pos_at_t0, pos_at_t1, ...] or None if no path found.
        First element is start position.
    """
    import time
    t_deadline = time.perf_counter() + deadline_ms / 1000.0 if deadline_ms > 0 else 0

    h_start = distance_fn(start, goal)
    if h_start >= 9999:
        return None  # Goal unreachable even without reservations

    if start == goal:
        # Check start is free AND holdable for goal_hold rounds
        ok = reservations.is_free(start, start_t, bot_id)
        if ok:
            for dt in range(1, goal_hold + 1):
                if not reservations.is_free(start, start_t + dt, bot_id):
                    ok = False
                    break
        if ok:
            return [start]
        # Fall through to search (might need to wait)

    # State: (x, y, t)
    # Priority queue: (f_score, counter, x, y, t)
    counter = 0
    open_set: list[tuple[int, int, int, int, int]] = [
        (h_start, counter, start[0], start[1], start_t)
    ]
    # g_score: (x, y, t) -> cost from start
    g_score: dict[tuple[int, int, int], int] = {(start[0], start[1], start_t): 0}
    # came_from: (x, y, t) -> (x, y, t)
    came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    max_absolute_t = start_t + max_t  # uses original start_t, not actual_start_t

    while open_set:
        # Time budget check (every 256 iterations)
        if t_deadline and (counter & 255) == 0:
            if time.perf_counter() > t_deadline:
                return None

        f, _, cx, cy, ct = heapq.heappop(open_set)
        current_state = (cx, cy, ct)

        current_g = g_score.get(current_state)
        if current_g is None or f > current_g + distance_fn((cx, cy), goal):
            continue  # Stale entry

        # Goal reached — check we can hold position for goal_hold rounds
        if (cx, cy) == goal:
            holdable = True
            for dt in range(1, goal_hold + 1):
                if not reservations.is_free(goal, ct + dt, bot_id):
                    holdable = False
                    break
            if holdable:
                # Reconstruct path
                path_states: list[tuple[int, int, int]] = [current_state]
                state = current_state
                while state in came_from:
                    state = came_from[state]
                    path_states.append(state)
                path_states.reverse()
                return [(s[0], s[1]) for s in path_states]
            # Can't hold at goal yet — continue searching for later arrival

        if ct >= max_absolute_t:
            continue  # Don't expand past horizon

        next_t = ct + 1

        # Generate successors: 4 cardinal moves + WAIT
        neighbors = directed_neighbors_fn((cx, cy))
        # Add WAIT (stay in place)
        candidates = [(cx, cy)] + [(n[0], n[1]) for n in neighbors]

        for nx, ny in candidates:
            # Check vertex conflict at next_t
            if not reservations.is_free((nx, ny), next_t, bot_id):
                continue

            # Check edge/swap conflict (skip in sequential mode —
            # lower-ID bots are already processed so swaps/follows are safe)
            if not sequential_mode and (nx, ny) != (cx, cy):
                if reservations.has_edge_conflict((cx, cy), (nx, ny), ct, bot_id):
                    continue
                if reservations.has_following_conflict((nx, ny), ct, bot_id):
                    continue

            next_state = (nx, ny, next_t)
            tentative_g = g_score[current_state] + 1

            if tentative_g < g_score.get(next_state, float("inf")):
                g_score[next_state] = tentative_g
                h = distance_fn((nx, ny), goal)
                counter += 1
                heapq.heappush(open_set, (tentative_g + h, counter, nx, ny, next_t))
                came_from[next_state] = current_state

    return None  # No path found
