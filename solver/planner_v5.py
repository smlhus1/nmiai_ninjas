"""Offline planner v5: sim-validated batched planner.

Uses the SIMULATOR as ground truth for collision resolution.
Plans one bot's trip at a time, executes in sim, observes actual
positions, then plans next bot.

Key insight from Erik: "Brute force 1 og 1 bot til sine 3 items
og til en drop via A*. Marker posisjoner som okupert basert på
alle planlagte moves."

Usage:
    py -m solver.planner_v5 --recon logs/74001e7f_2026-03-16_score274_recon.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.grid import GameMap, Grid, Pos, pickup_positions
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue
from solver.time_space import ReservationTable, time_space_astar
from solver.scripted_strategy import ScriptedStrategy
from Simulering.offline.simulator import Simulator


def bfs_path(grid: Grid, start: Pos, end: Pos) -> list[Pos]:
    from collections import deque
    if start == end:
        return [start]
    parent = {start: None}
    q = deque([start])
    while q:
        pos = q.popleft()
        if pos == end:
            path = []
            while pos is not None:
                path.append(pos)
                pos = parent[pos]
            return list(reversed(path))
        for nb in grid.neighbors(pos, respect_one_way=False):
            if nb not in parent:
                parent[nb] = pos
                q.append(nb)
    return [start]


class SimValidatedPlanner:
    """Plans bot trips using BFS paths, validates with simulator."""

    def __init__(self, recon_path: str):
        self.gm = GameMap.from_recon(recon_path)
        self.grid = self.gm.grid
        self.dist_cache = DistanceCache(self.grid)
        self.oq = OrderQueue.from_recon(recon_path)
        self.recon_path = recon_path
        self.n_bots = self.gm.bot_count
        self.spawn = self.gm.spawn
        self.max_rounds = 500
        self.grid_nb = lambda pos: self.grid.neighbors(pos, respect_one_way=False)

    def _best_shelf(self, item_type: str, near: Pos) -> tuple[Pos, Pos] | None:
        shelves = self.gm.shelf_map.get(item_type, [])
        best = None
        best_d = 9999
        for shelf in shelves:
            for pp in pickup_positions(self.grid, shelf):
                d = self.dist_cache.distance(near, pp)
                if d is not None and d < best_d:
                    best_d = d
                    best = (shelf, pp)
        return best

    def plan(self) -> dict:
        """Build a complete action plan, validated against sim."""
        orders = list(self.oq)

        # Pre-compute all trips: list of (item_types, shelf_positions, pickup_positions)
        trips = []
        for order in orders:
            items = list(order.items_required)
            # Group into batches of 3
            while items:
                batch_items = items[:3]
                items = items[3:]
                batch_shelves = []
                batch_pickups = []
                cur = self.spawn
                for it in batch_items:
                    result = self._best_shelf(it, cur)
                    if result:
                        batch_shelves.append(result[0])
                        batch_pickups.append(result[1])
                        cur = result[1]
                    else:
                        batch_shelves.append(self.spawn)
                        batch_pickups.append(self.spawn)
                trips.append((batch_items, batch_shelves, batch_pickups))

        print(f"Total trips: {len(trips)} for {len(orders)} orders", flush=True)

        # Build reservation table from BFS paths
        # Plan ALL bot trips with time-space A*
        reservations = ReservationTable()
        bot_pos = {i: self.spawn for i in range(self.n_bots)}
        bot_avail = {i: i for i in range(self.n_bots)}
        actions: dict[int, dict[int, str]] = {}

        trip_idx = 0
        total_items = 0
        total_orders = 0

        while trip_idx < len(trips):
            bot_id = min(bot_avail, key=lambda b: bot_avail[b])
            start_t = bot_avail[bot_id]
            if start_t >= self.max_rounds - 5:
                break

            batch_items, batch_shelves, batch_pickups = trips[trip_idx]
            cur = bot_pos[bot_id]

            # Plan path: cur -> pickup1 -> pickup2 -> pickup3 -> nearest_dz
            for it, shelf, pp in zip(batch_items, batch_shelves, batch_pickups):
                # BFS path (no reservation — sim handles collisions)
                path = bfs_path(self.grid, cur, pp)
                for j in range(1, len(path)):
                    r = start_t + j
                    self._set_move(actions, r, bot_id, path[j-1], path[j])
                arrive = start_t + len(path) - 1
                pick_r = arrive + 1
                self._set_raw(actions, pick_r, bot_id, f"pick_up:{it}")
                start_t = pick_r + 1
                cur = pp

            # Path to nearest drop-off
            dz = min(self.gm.drop_off_zones,
                     key=lambda z: self.dist_cache.distance(cur, z) or 999)
            path = bfs_path(self.grid, cur, dz)
            for j in range(1, len(path)):
                r = start_t + j
                self._set_move(actions, r, bot_id, path[j-1], path[j])
            arrive = start_t + len(path) - 1
            drop_r = arrive + 1
            self._set_raw(actions, drop_r, bot_id, "drop_off")

            # Bot returns to spawn after drop
            park_start = drop_r + 1
            path_back = bfs_path(self.grid, dz, self.spawn)
            for j in range(1, len(path_back)):
                r = park_start + j
                self._set_move(actions, r, bot_id, path_back[j-1], path_back[j])
            final_r = park_start + len(path_back) - 1

            bot_pos[bot_id] = self.spawn
            bot_avail[bot_id] = final_r + 1
            total_items += len(batch_items)
            trip_idx += 1

            # Track orders
            items_per_order = [len(o.items_required) for o in orders]
            cum = 0
            total_orders = 0
            for n in items_per_order:
                cum += n
                batches_needed = (n + 2) // 3
                # Count how many trips are done for this order
                # This is approximate — just count orders where all batches are planned

            if trip_idx % 10 == 0:
                print(f"Trip {trip_idx}/{len(trips)}: B{bot_id} drop_r={drop_r}, "
                      f"items={total_items}", flush=True)

        # Validate in sim
        strategy = ScriptedStrategy(actions)
        sim = Simulator.from_recon_file(self.recon_path)
        result = sim.run(strategy)

        print(f"\nSim score: {result['score']}, orders: {result['orders_completed']}, "
              f"items: {result['items_delivered']}, rounds: {result['rounds_used']}")

        return actions, result

    def _set_move(self, actions, r, bot_id, from_pos, to_pos):
        if r >= self.max_rounds:
            return
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        move = {(1,0): "move_right", (-1,0): "move_left",
                (0,1): "move_down", (0,-1): "move_up"}.get((dx, dy), "wait")
        self._set_raw(actions, r, bot_id, move)

    def _set_raw(self, actions, r, bot_id, action):
        if r >= self.max_rounds:
            return
        actions.setdefault(r, {})[bot_id] = action


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--save", help="Save plan as JSON")
    args = parser.parse_args()

    t0 = time.time()
    planner = SimValidatedPlanner(args.recon)
    plan, result = planner.plan()
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s")

    if args.save:
        Path(args.save).write_text(json.dumps(plan, indent=2))
