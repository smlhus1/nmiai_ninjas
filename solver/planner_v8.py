"""Planner v8: time-space BFS with sim-actual occupancy.

For each bot (0→19):
1. Build occupancy grid from bots 0..K-1's ACTUAL sim positions
2. Use bfs_timed to plan complete journey (all trips)
3. Run sim with bot K following planned path + all others
4. Record bot K's actual positions for next bot's occupancy

Usage:
    py -m solver.planner_v8 --recon logs/74001e7f_2026-03-17_recon.json
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

Pos = tuple[int, int]


def bfs_timed(grid: Grid, start: Pos, end: Pos, start_t: int,
              occupied: dict[tuple[int,int,int], bool],
              spawn: Pos, dzs: set[Pos],
              max_t: int = 500) -> list[Pos] | None:
    """BFS in (pos, t) space. Occupied cells are blocked (except spawn/dz)."""
    if start == end:
        return [start]
    initial = (start, start_t)
    parent = {initial: None}
    q = deque([initial])
    while q:
        (pos, t) = q.popleft()
        if pos == end:
            path = []
            state = (pos, t)
            while state is not None:
                path.append(state[0])
                state = parent[state]
            path.reverse()
            return path
        if t >= max_t - 1:
            continue
        nt = t + 1
        for nb in grid.neighbors(pos, respect_one_way=False) + [pos]:
            if nb != spawn and nb not in dzs and occupied.get((nb[0], nb[1], nt), False):
                continue
            state = (nb, nt)
            if state not in parent:
                parent[state] = (pos, t)
                q.append(state)
    return None


class SimPlanner:
    def __init__(self, recon_path: str):
        self.gm = GameMap.from_recon(recon_path)
        self.grid = self.gm.grid
        self.dist = DistanceCache(self.grid)
        self.oq = OrderQueue.from_recon(recon_path)
        self.recon_path = recon_path
        self.n_bots = self.gm.bot_count
        self.spawn = self.gm.spawn
        self.dzs = set(self.gm.drop_off_zones)
        self.dz_list = list(self.gm.drop_off_zones)
        self.max_t = 500

    def _best_shelf(self, item_type: str, near: Pos) -> tuple[Pos, Pos] | None:
        best, best_d = None, 9999
        for shelf in self.gm.shelf_map.get(item_type, []):
            for pp in pickup_positions(self.grid, shelf):
                d = self.dist.distance(near, pp)
                if d is not None and d < best_d:
                    best_d, best = d, (shelf, pp)
        return best

    def plan(self) -> tuple[dict, dict]:
        # Build trips per bot
        item_queue = []
        for order in self.oq:
            item_queue.extend(order.items_required)

        bot_trips = {i: [] for i in range(self.n_bots)}
        batch_idx = 0
        i = 0
        while i < len(item_queue):
            bid = batch_idx % self.n_bots
            trip = []
            cur = self.spawn
            for _ in range(min(3, len(item_queue) - i)):
                it = item_queue[i]
                r = self._best_shelf(it, cur)
                if r:
                    trip.append((it, r[0], r[1]))
                    cur = r[1]
                i += 1
            if trip:
                bot_trips[bid].append(trip)
            batch_idx += 1

        print(f"Bots: {self.n_bots}, Trips: {sum(len(t) for t in bot_trips.values())}",
              flush=True)

        # Occupancy grid: filled from sim-actual positions of planned bots
        occupied: dict[tuple[int,int,int], bool] = {}
        # recorded_actions[bot_id][round] = action_dict
        recorded: dict[int, dict[int, dict]] = {}
        # actual_positions[bot_id][round] = pos (from sim)
        actual_pos: dict[int, dict[int, Pos]] = {}

        for plan_bot in range(self.n_bots):
            trips = bot_trips[plan_bot]

            # Step 1: Pre-compute complete path using bfs_timed + occupancy
            precomputed: dict[int, dict] = {}  # round -> action
            cur = self.spawn
            t = plan_bot  # stagger

            for trip in trips:
                if t >= self.max_t - 10:
                    break
                trip_ok = True
                for item_type, shelf, pp in trip:
                    path = bfs_timed(self.grid, cur, pp, t, occupied,
                                     self.spawn, self.dzs, self.max_t)
                    if path is None:
                        trip_ok = False
                        break
                    for j in range(1, len(path)):
                        dx = path[j][0]-path[j-1][0]
                        dy = path[j][1]-path[j-1][1]
                        move = {(0,-1):"move_up",(0,1):"move_down",
                                (-1,0):"move_left",(1,0):"move_right"}.get((dx,dy),"wait")
                        precomputed[t+j] = {"bot": plan_bot, "action": move}
                    arrive = t + len(path) - 1
                    pick_t = arrive + 1
                    precomputed[pick_t] = {"bot": plan_bot, "action": "pick_up",
                                           "item_type": item_type, "shelf": shelf}
                    t = pick_t + 1
                    cur = pp

                if not trip_ok:
                    break

                # Dropoff
                dz = min(self.dz_list, key=lambda z: self.dist.distance(cur, z) or 999)
                path = bfs_timed(self.grid, cur, dz, t, occupied,
                                 self.spawn, self.dzs, self.max_t)
                if path is None:
                    for alt in self.dz_list:
                        path = bfs_timed(self.grid, cur, alt, t, occupied,
                                         self.spawn, self.dzs, self.max_t)
                        if path:
                            dz = alt
                            break
                if path is None:
                    break
                for j in range(1, len(path)):
                    dx = path[j][0]-path[j-1][0]
                    dy = path[j][1]-path[j-1][1]
                    move = {(0,-1):"move_up",(0,1):"move_down",
                            (-1,0):"move_left",(1,0):"move_right"}.get((dx,dy),"wait")
                    precomputed[t+j] = {"bot": plan_bot, "action": move}
                arrive = t + len(path) - 1
                drop_t = arrive + 1
                precomputed[drop_t] = {"bot": plan_bot, "action": "drop_off"}
                t = drop_t + 1
                cur = dz

            # Step 2: Run sim with this bot following precomputed + all others
            sim = Simulator.from_recon_file(self.recon_path)
            state = sim.reset()
            bot_actual: dict[int, Pos] = {}
            bot_rec: dict[int, dict] = {}

            for rnd in range(self.max_t):
                sd = state.to_dict()
                items = sd["items"]
                bots_data = {b["id"]: b for b in sd["bots"]}
                actions = []

                for bid in range(self.n_bots):
                    if bid == plan_bot:
                        pre = precomputed.get(rnd)
                        if pre:
                            act = dict(pre)
                            # Resolve pick_up item_id from sim state
                            if act["action"] == "pick_up":
                                it_type = act.pop("item_type", "")
                                shelf = act.pop("shelf", None)
                                bot_pos = tuple(bots_data[bid]["position"])
                                item_id = None
                                if shelf:
                                    item_id = next(
                                        (it["id"] for it in items
                                         if it["type"] == it_type and tuple(it["position"]) == shelf),
                                        None)
                                if item_id:
                                    act["item_id"] = item_id
                                else:
                                    act = {"bot": bid, "action": "wait"}
                            actions.append(act)
                            bot_rec[rnd] = act
                        else:
                            actions.append({"bot": bid, "action": "wait"})
                            bot_rec[rnd] = {"bot": bid, "action": "wait"}
                    elif bid in recorded:
                        act = recorded[bid].get(rnd, {"bot": bid, "action": "wait"})
                        actions.append(act)
                    else:
                        actions.append({"bot": bid, "action": "wait"})

                state, done = sim.step(actions)

                # Record actual position
                nd = state.to_dict()
                pb = next(b for b in nd["bots"] if b["id"] == plan_bot)
                bot_actual[rnd] = tuple(pb["position"])

                if done:
                    break

            # Step 3: Add actual positions to occupancy grid
            for rnd, pos in bot_actual.items():
                if pos != self.spawn and pos not in self.dzs:
                    occupied[(pos[0], pos[1], rnd)] = True

            recorded[plan_bot] = bot_rec
            actual_pos[plan_bot] = bot_actual

            completed = sum(1 for trip in trips
                           if any(r.get("action") == "drop_off" for r in bot_rec.values()))
            print(f"Bot {plan_bot:2d}: score={sim._score}", flush=True)

        # Validate: all bots replay
        sim = Simulator.from_recon_file(self.recon_path)
        state = sim.reset()
        for rnd in range(self.max_t):
            actions = []
            for bid in range(self.n_bots):
                act = recorded.get(bid, {}).get(rnd, {"bot": bid, "action": "wait"})
                actions.append(act)
            state, done = sim.step(actions)
            if done:
                break

        result = {"score": sim._score, "items": sim._items_delivered,
                  "orders": sim._orders_completed}
        print(f"\nAll replay: score={result['score']} items={result['items']} "
              f"orders={result['orders']}", flush=True)
        return recorded, result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    t0 = time.time()
    planner = SimPlanner(args.recon)
    recorded, result = planner.plan()
    print(f"Time: {time.time()-t0:.1f}s")
