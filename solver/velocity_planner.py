"""Velocity planner: pre-compute collision-free paths for maximum score/round.

Phase 1: Spawn scatter — get 20 bots from (28,16) to unique positions in <15 rounds
Phase 2: Trip planning — assign items to bots, compute paths with reservation table
Phase 3: Execute — replay plan through sim

Uses prioritized planning: plan high-priority bots first, reserve their paths,
lower-priority bots plan around reservations. Window-based (30-round horizon).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

Pos = tuple[int, int]

# Direction deltas
DIRS = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]  # wait, up, down, left, right
DIR_NAMES = {(0, 0): "wait", (0, -1): "move_up", (0, 1): "move_down",
             (-1, 0): "move_left", (1, 0): "move_right"}


class ReservationTable:
    """Time-space reservation table for collision-free planning."""

    def __init__(self):
        self._reserved: set[tuple[int, int, int]] = set()  # (x, y, t)
        self._edge_reserved: set[tuple[int, int, int, int, int]] = set()  # (x1,y1,x2,y2,t)

    def reserve(self, pos: Pos, t: int):
        self._reserved.add((pos[0], pos[1], t))

    def reserve_edge(self, from_pos: Pos, to_pos: Pos, t: int):
        """Reserve edge to prevent swap conflicts."""
        self._edge_reserved.add((from_pos[0], from_pos[1], to_pos[0], to_pos[1], t))

    def is_free(self, pos: Pos, t: int) -> bool:
        return (pos[0], pos[1], t) not in self._reserved

    def has_edge_conflict(self, from_pos: Pos, to_pos: Pos, t: int) -> bool:
        """Check if moving from->to at time t conflicts with another bot moving to->from."""
        return (to_pos[0], to_pos[1], from_pos[0], from_pos[1], t) in self._edge_reserved

    def reserve_path(self, path: list[Pos], start_t: int):
        """Reserve all positions along a path."""
        for i, pos in enumerate(path):
            t = start_t + i
            self.reserve(pos, t)
            if i > 0:
                self.reserve_edge(path[i - 1], pos, t)


class VelocityPlanner:
    """Plan collision-free paths for 20 bots to maximize velocity."""

    def __init__(self, recon: dict):
        self.recon = recon
        grid_w, grid_h = recon["grid_size"]
        self.w = grid_w
        self.h = grid_h

        # Build grid
        walls = set(tuple(w) for w in recon["walls"])
        shelves: set[Pos] = set()
        self._shelf_map: dict[str, list[Pos]] = {}
        for item_type, positions in recon["shelf_map"].items():
            self._shelf_map[item_type] = [tuple(p) for p in positions]
            for p in positions:
                shelves.add(tuple(p))
        self.obstacles = walls | shelves

        self.drop_offs = [tuple(z) for z in recon["drop_off_zones"]]
        self.spawn = (28, 16)  # nightmare spawn
        self.orders = recon.get("order_sequence", [])
        self.n_bots = recon.get("bot_count", 20)

        # PathEngine for BFS distances
        from bot.models import Grid as BotGrid
        from bot.engine.pathfinding import PathEngine

        self._grid = BotGrid(grid_w, grid_h, frozenset(self.obstacles))
        self._pe = PathEngine()
        self._pe.set_grid(self._grid, drop_off=self.drop_offs[0])
        self._pe._one_way = self._pe._detect_one_way_aisles(self._grid, self.drop_offs[0])
        self._pe._one_way_enabled = True

    def _neighbors(self, pos: Pos) -> list[Pos]:
        """Walkable neighbors respecting one-way aisles."""
        return list(self._pe._directed_neighbors(pos))

    def _time_space_astar(self, start: Pos, goal: Pos, start_t: int,
                          reservations: ReservationTable, max_t: int = 100) -> list[Pos] | None:
        """A* in time-space. Returns path (list of positions per timestep) or None."""
        import heapq

        # Heuristic: BFS distance (admissible)
        h = self._pe.distance(start, goal) or 9999
        if h >= 9999:
            return None

        # (f, g, t, x, y)
        open_set = [(h, 0, start_t, start[0], start[1])]
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        g_score: dict[tuple[int, int, int], int] = {(start[0], start[1], start_t): 0}

        while open_set:
            f, g, t, x, y = heapq.heappop(open_set)
            pos = (x, y)

            if pos == goal and t > start_t:
                # Reconstruct path
                path = []
                state = (x, y, t)
                while state in came_from:
                    path.append((state[0], state[1]))
                    state = came_from[state]
                path.append((state[0], state[1]))
                path.reverse()
                return path

            if t >= start_t + max_t:
                continue

            next_t = t + 1

            # Try all moves: wait + 4 directions
            candidates = [pos] + self._neighbors(pos)
            for npos in candidates:
                if not reservations.is_free(npos, next_t):
                    continue
                if npos != pos and reservations.has_edge_conflict(pos, npos, next_t):
                    continue

                new_g = g + 1
                state_key = (npos[0], npos[1], next_t)

                if state_key in g_score and g_score[state_key] <= new_g:
                    continue

                g_score[state_key] = new_g
                h_val = self._pe.distance(npos, goal) or 9999
                came_from[state_key] = (x, y, t)
                heapq.heappush(open_set, (new_g + h_val, new_g, next_t, npos[0], npos[1]))

        return None  # No path found

    def _pickup_pos(self, shelf_pos: Pos) -> Pos | None:
        """Best walkable cell adjacent to shelf."""
        best, best_d = None, 9999
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            adj = (shelf_pos[0] + dx, shelf_pos[1] + dy)
            if 0 <= adj[0] < self.w and 0 <= adj[1] < self.h and adj not in self.obstacles:
                # Prefer positions closer to nearest drop-off
                d = min(self._pe.distance(adj, dz) or 9999 for dz in self.drop_offs)
                if d < best_d:
                    best_d = d
                    best = adj
        return best

    def _nearest_dropoff(self, pos: Pos) -> Pos:
        best, best_d = self.drop_offs[0], 9999
        for dz in self.drop_offs:
            d = self._pe.distance(pos, dz) or 9999
            if d < best_d:
                best_d = d
                best = dz
        return best

    def plan_scatter(self, reservations: ReservationTable) -> dict[int, list[Pos]]:
        """Plan collision-free scatter from spawn for all bots.

        Returns dict of bot_id -> path (list of positions per timestep).
        Plans in bot ID order (low ID = high priority).
        """
        paths: dict[int, list[Pos]] = {}

        # BFS to find scatter targets (unique positions reachable from spawn)
        targets: list[Pos] = []
        visited = set()
        queue = deque([(self.spawn, 0)])
        visited.add(self.spawn)

        while queue and len(targets) < self.n_bots:
            pos, dist = queue.popleft()
            if dist > 0:
                targets.append(pos)
            for npos in self._neighbors(pos):
                if npos not in visited:
                    visited.add(npos)
                    queue.append((npos, dist + 1))

        # Assign targets to bots: bot 0 gets closest, bot 19 gets furthest
        # But for velocity: assign based on which shelf zone each bot will serve
        for bot_id in range(self.n_bots):
            if bot_id < len(targets):
                target = targets[bot_id]
            else:
                target = self.spawn

            path = self._time_space_astar(self.spawn, target, 0, reservations, max_t=40)
            if path:
                reservations.reserve_path(path, 0)
                paths[bot_id] = path
            else:
                # Fallback: just wait at spawn
                paths[bot_id] = [self.spawn] * 5
                for t in range(5):
                    reservations.reserve(self.spawn, t)

        return paths

    def plan_trips(self) -> list[dict]:
        """Plan all trips for all orders. Returns MAPF-style plan."""
        reservations = ReservationTable()

        # Phase 1: Scatter
        print("Planning scatter...", flush=True)
        scatter_paths = self.plan_scatter(reservations)

        scatter_end = max(len(p) for p in scatter_paths.values())
        print(f"Scatter complete: {scatter_end} rounds, {len(scatter_paths)} bots", flush=True)

        # Bot positions after scatter
        bot_pos: dict[int, Pos] = {}
        bot_available: dict[int, int] = {}  # round when bot is free
        for bot_id, path in scatter_paths.items():
            bot_pos[bot_id] = path[-1]
            bot_available[bot_id] = len(path) - 1

        # Phase 2: Trip planning — order by order
        all_actions: dict[int, list[dict]] = {i: [] for i in range(self.n_bots)}

        # Initialize actions from scatter
        for bot_id, path in scatter_paths.items():
            for t in range(len(path)):
                pos = path[t]
                prev = path[t - 1] if t > 0 else pos
                dx, dy = pos[0] - prev[0], pos[1] - prev[1]
                action_name = DIR_NAMES.get((dx, dy), "wait")
                all_actions[bot_id].append({"action": action_name, "position": list(pos)})

        total_rounds = scatter_end
        total_score = 0
        orders_completed = 0

        for oi, order in enumerate(self.orders):
            items_needed = list(order["items_required"])
            n_items = len(items_needed)

            # Assign items to nearest available bots
            assignments: list[tuple[int, str, Pos, Pos, Pos]] = []  # (bot_id, type, shelf, pickup, dropoff)

            for item_type in items_needed:
                shelves = self._shelf_map.get(item_type, [])
                if not shelves:
                    continue

                # Find best bot + shelf combo
                best_bot = -1
                best_cost = 9999
                best_shelf = shelves[0]
                best_pickup = None
                best_dropoff = None

                for shelf in shelves:
                    pp = self._pickup_pos(shelf)
                    if not pp:
                        continue
                    dz = self._nearest_dropoff(pp)

                    for bot_id in range(self.n_bots):
                        if bot_available[bot_id] > total_rounds + 50:
                            continue
                        d_to_pickup = self._pe.distance(bot_pos[bot_id], pp) or 9999
                        d_to_drop = self._pe.distance(pp, dz) or 9999
                        cost = d_to_pickup + d_to_drop + max(0, bot_available[bot_id] - total_rounds)
                        if cost < best_cost:
                            best_cost = cost
                            best_bot = bot_id
                            best_shelf = shelf
                            best_pickup = pp
                            best_dropoff = dz

                if best_bot >= 0 and best_pickup:
                    assignments.append((best_bot, item_type, best_shelf, best_pickup, best_dropoff))
                    # Reserve this bot
                    bot_available[best_bot] = total_rounds + best_cost + 2

            if not assignments:
                continue

            # Plan paths for each assignment
            order_end = total_rounds
            for bot_id, item_type, shelf, pickup, dropoff in assignments:
                start_t = max(total_rounds, bot_available.get(bot_id, total_rounds))

                # Path: current pos -> pickup
                path_to_pickup = self._time_space_astar(
                    bot_pos[bot_id], pickup, start_t, reservations, max_t=60
                )
                if not path_to_pickup:
                    continue

                pickup_t = start_t + len(path_to_pickup) - 1

                # Path: pickup -> dropoff (after 1 round for pick action)
                path_to_drop = self._time_space_astar(
                    pickup, dropoff, pickup_t + 1, reservations, max_t=60
                )
                if not path_to_drop:
                    continue

                drop_t = pickup_t + 1 + len(path_to_drop) - 1

                # Reserve both paths
                reservations.reserve_path(path_to_pickup, start_t)
                reservations.reserve(pickup, pickup_t + 1)  # pick action round
                reservations.reserve_path(path_to_drop, pickup_t + 1)

                # Update bot state
                bot_pos[bot_id] = dropoff
                bot_available[bot_id] = drop_t + 1
                order_end = max(order_end, drop_t + 1)

            # Score
            total_score += n_items + 5
            orders_completed += 1
            total_rounds = order_end

            if oi < 10 or oi % 5 == 0:
                velocity = total_score / max(total_rounds, 1)
                print(f"Order {oi:2d}: {n_items} items, ends R{total_rounds}, "
                      f"score={total_score}, vel={velocity:.2f}/r", flush=True)

            if total_rounds > 500:
                print(f"Exceeded 500 rounds at order {oi}", flush=True)
                break

        print(f"\n=== RESULT: {total_score} score, {orders_completed} orders, "
              f"{total_rounds} rounds, {total_score/max(total_rounds,1):.2f}/round ===", flush=True)

        if total_rounds <= 180:
            print(f"Score@180: {total_score} ({total_score/180:.2f}/round)", flush=True)
        else:
            # Estimate score at 180
            # Linear interpolation based on order completion times
            print(f"Exceeds 180 rounds — full game velocity: {total_score/max(total_rounds,1):.2f}/round", flush=True)

        return []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    with open(args.recon) as f:
        recon = json.load(f)

    planner = VelocityPlanner(recon)
    t0 = time.time()
    planner.plan_trips()
    print(f"Time: {time.time()-t0:.1f}s")
