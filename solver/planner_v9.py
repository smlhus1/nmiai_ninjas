"""Planner v9: zone-based planning. 3 zones, ~7 bots each, zero cross-zone conflicts.

Nightmare map: 3 drop-off zones at (1,16), (15,16), (27,16).
Divide map into 3 zones by x-coordinate.
Each bot assigned to one zone, only picks items from that zone's shelves,
delivers to that zone's drop-off.

Bots in different zones NEVER collide (different map areas).
Within a zone: bfs_timed + occupancy grid handles 7-bot coordination.

Usage:
    py -m solver.planner_v9 --recon logs/74001e7f_2026-03-17_recon.json
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
from solver.scripted_strategy import ScriptedStrategy
from Simulering.offline.simulator import Simulator

ZONES = [
    {"name": "left",  "x_min": 0,  "x_max": 9,  "dz": (1, 16)},
    {"name": "mid",   "x_min": 10, "x_max": 19, "dz": (15, 16)},
    {"name": "right", "x_min": 20, "x_max": 29, "dz": (27, 16)},
]


def bfs_timed(grid: Grid, start: Pos, end: Pos, start_t: int,
              occupied: dict, spawn: Pos, dz: Pos,
              max_t: int = 500) -> list[Pos] | None:
    if start == end:
        return [start]
    parent = {(start, start_t): None}
    q = deque([(start, start_t)])
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
            if nb != spawn and nb != dz and occupied.get((nb[0], nb[1], nt), False):
                continue
            state = (nb, nt)
            if state not in parent:
                parent[state] = (pos, t)
                q.append(state)
    return None


class ZonePlanner:
    def __init__(self, recon_path: str):
        self.gm = GameMap.from_recon(recon_path)
        self.grid = self.gm.grid
        self.dist = DistanceCache(self.grid)
        self.oq = OrderQueue.from_recon(recon_path)
        self.recon_path = recon_path
        self.n_bots = self.gm.bot_count
        self.spawn = self.gm.spawn
        self.max_t = 500

        # Classify shelves by zone
        self.zone_shelves: dict[str, dict[str, list[tuple[Pos, Pos]]]] = {}
        for zone in ZONES:
            zname = zone["name"]
            self.zone_shelves[zname] = {}
            for item_type, positions in self.gm.shelf_map.items():
                entries = []
                for shelf in positions:
                    if zone["x_min"] <= shelf[0] <= zone["x_max"]:
                        for pp in pickup_positions(self.grid, shelf):
                            entries.append((shelf, pp))
                if entries:
                    self.zone_shelves[zname][item_type] = entries

    def _find_shelf_in_zone(self, item_type: str, near: Pos, zone_name: str) -> tuple[Pos, Pos] | None:
        entries = self.zone_shelves.get(zone_name, {}).get(item_type, [])
        if not entries:
            # Fallback: try any zone
            for zn in self.zone_shelves:
                entries = self.zone_shelves[zn].get(item_type, [])
                if entries:
                    break
        best, best_d = None, 9999
        for shelf, pp in entries:
            d = self.dist.distance(near, pp)
            if d is not None and d < best_d:
                best_d, best = d, (shelf, pp)
        return best

    def plan(self) -> dict[int, dict[int, str]]:
        # Assign bots to zones: 7/7/6 split
        bot_zones: dict[int, dict] = {}
        zone_bots: dict[str, list[int]] = {z["name"]: [] for z in ZONES}
        bots_per_zone = [7, 7, 6]
        bid = 0
        for i, zone in enumerate(ZONES):
            for _ in range(bots_per_zone[i]):
                bot_zones[bid] = zone
                zone_bots[zone["name"]].append(bid)
                bid += 1

        # Collect items needed per order
        item_queue = []
        for order in self.oq:
            item_queue.extend(order.items_required)

        # Distribute items round-robin across zones (balanced)
        # Each zone gets every 3rd item
        zone_items: dict[str, list[str]] = {z["name"]: [] for z in ZONES}
        zone_names = [z["name"] for z in ZONES]
        for i, item_type in enumerate(item_queue):
            # Assign to zone that HAS this item type and has fewest items
            candidates = [zn for zn in zone_names
                         if item_type in self.zone_shelves.get(zn, {})]
            if not candidates:
                candidates = zone_names
            # Pick zone with fewest items assigned
            target = min(candidates, key=lambda zn: len(zone_items[zn]))
            zone_items[target].append(item_type)

        for zn in zone_items:
            print(f"Zone {zn}: {len(zone_bots[zn])} bots, {len(zone_items[zn])} items",
                  flush=True)

        # Plan each zone independently
        all_actions: dict[int, dict[int, str]] = {}

        for zone in ZONES:
            zname = zone["name"]
            dz = zone["dz"]
            bots = zone_bots[zname]
            items = zone_items[zname]

            if not bots or not items:
                continue

            # Build trips: batches of 3
            trips_per_bot: dict[int, list[list[tuple[str, Pos, Pos]]]] = {b: [] for b in bots}
            batch_idx = 0
            i = 0
            while i < len(items):
                bot_id = bots[batch_idx % len(bots)]
                trip = []
                cur = self.spawn
                for _ in range(min(3, len(items) - i)):
                    it = items[i]
                    r = self._find_shelf_in_zone(it, cur, zname)
                    if r:
                        trip.append((it, r[0], r[1]))
                        cur = r[1]
                    i += 1
                if trip:
                    trips_per_bot[bot_id].append(trip)
                batch_idx += 1

            # Plan with bfs_timed + zone-local occupancy
            occupied: dict[tuple[int,int,int], bool] = {}
            total_items = 0

            for bot_id in bots:
                trips = trips_per_bot[bot_id]
                cur = self.spawn
                t = bot_id * 3  # stagger: 3 rounds per bot for spawn clearance

                for trip in trips:
                    if t >= self.max_t - 10:
                        break
                    trip_ok = True
                    for item_type, shelf, pp in trip:
                        path = bfs_timed(self.grid, cur, pp, t, occupied,
                                        self.spawn, dz, self.max_t)
                        if path is None:
                            trip_ok = False
                            break

                        # Mark occupancy + write actions
                        for j, pos in enumerate(path):
                            occupied[(pos[0], pos[1], t+j)] = True
                        for j in range(1, len(path)):
                            self._set_move(all_actions, t+j, bot_id, path[j-1], path[j])

                        arrive = t + len(path) - 1
                        pick_t = arrive + 1
                        occupied[(pp[0], pp[1], pick_t)] = True
                        occupied[(pp[0], pp[1], pick_t+1)] = True
                        self._set_raw(all_actions, pick_t, bot_id, f"pick_up:{item_type}")
                        t = pick_t + 1
                        cur = pp

                    if not trip_ok:
                        break

                    # Dropoff
                    path = bfs_timed(self.grid, cur, dz, t, occupied,
                                    self.spawn, dz, self.max_t)
                    if path is None:
                        break
                    for j, pos in enumerate(path):
                        occupied[(pos[0], pos[1], t+j)] = True
                    for j in range(1, len(path)):
                        self._set_move(all_actions, t+j, bot_id, path[j-1], path[j])

                    arrive = t + len(path) - 1
                    drop_t = arrive + 1
                    self._set_raw(all_actions, drop_t, bot_id, "drop_off")
                    occupied[(dz[0], dz[1], drop_t)] = True

                    t = drop_t + 1
                    cur = dz
                    total_items += len(trip)

                    # Mark idle
                    for tt in range(t, min(t+50, self.max_t)):
                        occupied[(cur[0], cur[1], tt)] = True

            print(f"  Zone {zname}: {total_items} items planned", flush=True)

        return all_actions

    def _set_move(self, actions, r, bot_id, from_pos, to_pos):
        if r < 0 or r >= self.max_t: return
        dx, dy = to_pos[0]-from_pos[0], to_pos[1]-from_pos[1]
        move = {(1,0):"move_right",(-1,0):"move_left",
                (0,1):"move_down",(0,-1):"move_up"}.get((dx,dy),"wait")
        self._set_raw(actions, r, bot_id, move)

    def _set_raw(self, actions, r, bot_id, action):
        if r < 0 or r >= self.max_t: return
        actions.setdefault(r, {})[bot_id] = action

    def validate(self, plan):
        strategy = ScriptedStrategy(plan)
        sim = Simulator.from_recon_file(self.recon_path)
        return sim.run(strategy)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    t0 = time.time()
    p = ZonePlanner(args.recon)
    plan = p.plan()
    result = p.validate(plan)
    print(f"\nSim: score={result['score']} items={result['items_delivered']} "
          f"orders={result['orders_completed']}")
    print(f"Time: {time.time()-t0:.1f}s")
