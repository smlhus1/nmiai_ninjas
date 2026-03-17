"""Sim-in-the-loop planner: uses actual simulator for collision resolution.

Runs the sim step by step. Each round:
1. Assign idle bots to trip batches (3 items → deliver → return to spawn)
2. For each bot, compute next move via BFS avoiding other bots' positions
3. Step sim, observe actual results, react to collisions

Zero sim-mismatch because we USE the sim directly.

Usage:
    py -m solver.sim_planner --recon logs/74001e7f_2026-03-16_score274_recon.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.grid import GameMap, Grid, Pos, pickup_positions
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue
from Simulering.offline.simulator import Simulator


def bfs_path(grid: Grid, start: Pos, end: Pos, obstacles: set[Pos] | None = None) -> list[Pos]:
    """BFS avoiding obstacles."""
    if start == end:
        return [start]
    blocked = obstacles or set()
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
            if nb not in parent and (nb not in blocked or nb == end):
                parent[nb] = pos
                q.append(nb)
    if obstacles:
        return bfs_path(grid, start, end, None)
    return [start]


class SimPlanner:
    def __init__(self, recon_path: str):
        self.gm = GameMap.from_recon(recon_path)
        self.grid = self.gm.grid
        self.dist = DistanceCache(self.grid)
        self.oq = OrderQueue.from_recon(recon_path)
        self.recon_path = recon_path
        self.n_bots = self.gm.bot_count
        self.spawn = self.gm.spawn
        self.max_rounds = 500
        self.dzs = set(self.gm.drop_off_zones)
        self.dz_list = list(self.gm.drop_off_zones)

    def _best_pickup(self, item_type: str, near: Pos) -> tuple[Pos, Pos] | None:
        best, best_d = None, 9999
        for shelf in self.gm.shelf_map.get(item_type, []):
            for pp in pickup_positions(self.grid, shelf):
                d = self.dist.distance(near, pp)
                if d is not None and d < best_d:
                    best_d, best = d, (shelf, pp)
        return best

    def _nearest_dz(self, pos: Pos) -> Pos:
        return min(self.dz_list, key=lambda z: self.dist.distance(pos, z) or 999)

    def plan_and_run(self) -> dict:
        sim = Simulator.from_recon_file(self.recon_path)
        state = sim.reset()

        # Build trip queue: batches of 3 items
        trips: list[list[tuple[str, Pos, Pos]]] = []
        for order in self.oq:
            items = list(order.items_required)
            while items:
                batch, items = items[:3], items[3:]
                trip = []
                cur = self.spawn
                for it in batch:
                    r = self._best_pickup(it, cur)
                    if r:
                        trip.append((it, r[0], r[1]))
                        cur = r[1]
                if trip:
                    trips.append(trip)

        # Bot state
        IDLE, PICKUP, DELIVER, RETURN = "idle", "pickup", "deliver", "return"
        task = {i: IDLE for i in range(self.n_bots)}
        pick_list: dict[int, list[tuple[str, Pos, Pos]]] = {}
        pick_idx: dict[int, int] = {}
        trip_idx = 0

        for rnd in range(self.max_rounds):
            sd = state.to_dict()
            bots = {b["id"]: b for b in sd["bots"]}
            items_map = sd["items"]
            occupied = {tuple(b["position"]) for b in sd["bots"]}

            # Assign idle bots
            for bid in sorted(task, key=lambda b: task[b] != IDLE):
                if task[bid] == IDLE and trip_idx < len(trips):
                    pick_list[bid] = trips[trip_idx]
                    pick_idx[bid] = 0
                    task[bid] = PICKUP
                    trip_idx += 1

            actions = []
            for bid in range(self.n_bots):
                pos = tuple(bots[bid]["position"])
                inv = bots[bid]["inventory"]

                # PICKUP: navigate to items, pick them up one by one
                if task[bid] == PICKUP:
                    idx = pick_idx.get(bid, 0)
                    if idx >= len(pick_list.get(bid, [])):
                        task[bid] = DELIVER
                    else:
                        it_type, shelf, pp = pick_list[bid][idx]
                        # Adjacent to shelf? Pick up
                        if abs(pos[0]-shelf[0]) + abs(pos[1]-shelf[1]) == 1:
                            item_id = next(
                                (i["id"] for i in items_map
                                 if i["type"] == it_type and tuple(i["position"]) == shelf),
                                None,
                            )
                            if item_id:
                                actions.append({"bot": bid, "action": "pick_up", "item_id": item_id})
                                pick_idx[bid] = idx + 1
                                continue
                        # Navigate to pickup pos
                        others = occupied - {pos}
                        path = bfs_path(self.grid, pos, pp, others)
                        if len(path) >= 2:
                            actions.append(self._move_action(bid, pos, path[1]))
                            continue
                        actions.append({"bot": bid, "action": "wait"})
                        continue

                # DELIVER: go to drop-off and drop
                if task[bid] == DELIVER:
                    if pos in self.dzs:
                        actions.append({"bot": bid, "action": "drop_off"})
                        continue
                    dz = self._nearest_dz(pos)
                    others = occupied - {pos}
                    path = bfs_path(self.grid, pos, dz, others)
                    if len(path) >= 2:
                        actions.append(self._move_action(bid, pos, path[1]))
                        continue
                    actions.append({"bot": bid, "action": "wait"})
                    continue

                # RETURN: go back to spawn
                if task[bid] == RETURN:
                    if pos == self.spawn:
                        task[bid] = IDLE
                    else:
                        others = occupied - {pos}
                        path = bfs_path(self.grid, pos, self.spawn, others)
                        if len(path) >= 2:
                            actions.append(self._move_action(bid, pos, path[1]))
                            continue

                # IDLE or stuck
                actions.append({"bot": bid, "action": "wait"})

            # Step sim
            state, done = sim.step(actions)

            # Post-step: check actual state and update tasks
            nd = state.to_dict()
            for bot in nd["bots"]:
                bid = bot["id"]
                if task[bid] == DELIVER and len(bot["inventory"]) == 0:
                    task[bid] = RETURN

            if done:
                break

            if rnd % 100 == 0:
                tc = {}
                for t in task.values():
                    tc[t] = tc.get(t, 0) + 1
                print(f"R{rnd}: score={sim._score} trips={trip_idx}/{len(trips)} {tc}",
                      flush=True)

        print(f"\nFinal: score={sim._score} orders={sim._orders_completed} "
              f"items={sim._items_delivered} rounds={sim._round}", flush=True)
        return {"score": sim._score, "orders": sim._orders_completed,
                "items": sim._items_delivered, "rounds": sim._round}

    @staticmethod
    def _move_action(bid: int, from_pos: Pos, to_pos: Pos) -> dict:
        dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        move = {(0,-1): "move_up", (0,1): "move_down",
                (-1,0): "move_left", (1,0): "move_right"}.get((dx, dy), "wait")
        return {"bot": bid, "action": move}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    t0 = time.time()
    planner = SimPlanner(args.recon)
    result = planner.plan_and_run()
    print(f"Time: {time.time() - t0:.1f}s")
