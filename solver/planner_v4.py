"""Offline planner v4: Erik-style sequential bot planning.

1. Know all orders (recon)
2. Group items into batches of 3
3. Assign batches to bots upfront
4. Plan bot 0's ENTIRE journey (all trips), mark W×H×T
5. Plan bot 1 around bot 0's marks, etc.

No two bots on same cell at same timestep = zero collisions.

Usage:
    py -m solver.planner_v4 --recon logs/74001e7f_2026-03-17_recon.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.grid import GameMap, Grid, Pos, pickup_positions
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue
from solver.time_space import ReservationTable, time_space_astar
from solver.scripted_strategy import ScriptedStrategy
from Simulering.offline.simulator import Simulator


@dataclass
class Trip:
    items: list[str]
    shelves: list[Pos]
    pickups: list[Pos]


class BatchedPlanner:
    def __init__(self, recon_path: str):
        self.gm = GameMap.from_recon(recon_path)
        self.grid = self.gm.grid
        self.dist = DistanceCache(self.grid)
        self.oq = OrderQueue.from_recon(recon_path)
        self.recon_path = recon_path
        self.n_bots = self.gm.bot_count
        self.spawn = self.gm.spawn
        self.max_rounds = 500
        self.nb = lambda pos: self.grid.neighbors(pos, respect_one_way=False)

    def _best_shelf(self, item_type: str, near: Pos) -> tuple[Pos, Pos] | None:
        best, best_d = None, 9999
        for shelf in self.gm.shelf_map.get(item_type, []):
            for pp in pickup_positions(self.grid, shelf):
                d = self.dist.distance(near, pp)
                if d is not None and d < best_d:
                    best_d, best = d, (shelf, pp)
        return best

    def _make_trips(self, items: list[str], ref: Pos) -> list[Trip]:
        """Group items into batches of 3, greedy nearest."""
        remaining = list(items)
        trips = []
        while remaining:
            t_items, t_shelves, t_pickups = [], [], []
            cur = ref
            for _ in range(min(3, len(remaining))):
                bi, bd, bs, bp = -1, 9999, None, None
                for i, it in enumerate(remaining):
                    r = self._best_shelf(it, cur)
                    if r:
                        d = self.dist.distance(cur, r[1])
                        if d is not None and d < bd:
                            bi, bd, bs, bp = i, d, r[0], r[1]
                if bi < 0:
                    break
                t_items.append(remaining.pop(bi))
                t_shelves.append(bs)
                t_pickups.append(bp)
                cur = bp
            if t_items:
                trips.append(Trip(t_items, t_shelves, t_pickups))
        return trips

    def plan(self) -> dict[int, dict[int, str]]:
        """Plan ALL bots sequentially. Bot 0 first, then bot 1 around bot 0, etc."""
        res = ReservationTable()
        actions: dict[int, dict[int, str]] = {}

        # Step 1: Build ALL trips from order sequence
        all_trips: list[Trip] = []
        for order in self.oq:
            all_trips.extend(self._make_trips(list(order.items_required), self.spawn))

        # Step 2: Distribute trips to bots round-robin
        # (simple, deterministic — can optimize with permutation search later)
        bot_trips: dict[int, list[Trip]] = {i: [] for i in range(self.n_bots)}
        for i, trip in enumerate(all_trips):
            bot_id = i % self.n_bots
            bot_trips[bot_id].append(trip)

        print(f"Trips: {len(all_trips)} for {len(list(self.oq))} orders, "
              f"{self.n_bots} bots ({len(all_trips)//self.n_bots} per bot)",
              flush=True)

        # Step 3: Plan each bot's ENTIRE journey, one bot at a time
        total_items = 0

        # Plan in REVERSE ID order: bot 19 first, bot 0 last.
        # In sim, bot 0 moves first (highest priority). By planning it LAST,
        # bot 0 sees all other bots' reserved paths and routes around them.
        # This matches sim's sequential model: bot 0 will always succeed
        # because it planned around everyone else.
        for bot_id in reversed(range(self.n_bots)):
            trips = bot_trips[bot_id]
            if not trips:
                # Idle bot: reserve at spawn for all time
                res.reserve_stay(bot_id, self.spawn, 0, self.max_rounds)
                continue

            cur = self.spawn
            t = bot_id  # Stagger: bot i starts at time i

            # Reserve at spawn until start time
            res.reserve_stay(bot_id, self.spawn, 0, t)

            for trip in trips:
                if t >= self.max_rounds - 5:
                    break

                # Plan pickups
                trip_ok = True
                for item_type, shelf, pp in zip(trip.items, trip.shelves, trip.pickups):
                    path = time_space_astar(self.nb, cur, pp, t, bot_id, res, self.max_rounds)
                    if path is None:
                        trip_ok = False
                        break

                    # Write movements
                    for j in range(1, len(path)):
                        self._set_move(actions, t + j, bot_id, path[j-1], path[j])
                    res.reserve_path(bot_id, path, t)

                    arrive = t + len(path) - 1
                    pick_t = arrive + 1
                    self._set_raw(actions, pick_t, bot_id, f"pick_up:{item_type}")
                    res.reserve(bot_id, pp, pick_t)

                    t = pick_t + 1
                    cur = pp

                if not trip_ok:
                    # Reserve idle at current position
                    res.reserve_stay(bot_id, cur, t, self.max_rounds)
                    break

                # Plan path to nearest dropoff
                dz = min(self.gm.drop_off_zones,
                         key=lambda z: self.dist.distance(cur, z) or 999)

                path = time_space_astar(self.nb, cur, dz, t, bot_id, res, self.max_rounds)
                if path is None:
                    # Try all drop-off zones
                    for alt in self.gm.drop_off_zones:
                        path = time_space_astar(self.nb, cur, alt, t, bot_id, res, self.max_rounds)
                        if path:
                            dz = alt
                            break

                if path is None:
                    res.reserve_stay(bot_id, cur, t, self.max_rounds)
                    break

                for j in range(1, len(path)):
                    self._set_move(actions, t + j, bot_id, path[j-1], path[j])
                res.reserve_path(bot_id, path, t)

                arrive = t + len(path) - 1
                drop_t = arrive + 1
                self._set_raw(actions, drop_t, bot_id, "drop_off")
                res.reserve(bot_id, dz, drop_t)

                t = drop_t + 1
                cur = dz
                total_items += len(trip.items)

            # Reserve idle at final position for remaining time
            if t < self.max_rounds:
                res.reserve_stay(bot_id, cur, t, self.max_rounds)

            items_so_far = total_items
            print(f"Bot {bot_id:2d}: {len(trips)} trips, ends@{t}, items={items_so_far}",
                  flush=True)

        print(f"\nTotal items planned: {total_items}", flush=True)
        return actions

    def _set_move(self, actions, r, bot_id, from_pos, to_pos):
        if r < 0 or r >= self.max_rounds:
            return
        dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        move = {(1,0): "move_right", (-1,0): "move_left",
                (0,1): "move_down", (0,-1): "move_up"}.get((dx, dy), "wait")
        self._set_raw(actions, r, bot_id, move)

    def _set_raw(self, actions, r, bot_id, action):
        if r < 0 or r >= self.max_rounds:
            return
        actions.setdefault(r, {})[bot_id] = action

    def validate(self, plan: dict) -> dict:
        strategy = ScriptedStrategy(plan)
        sim = Simulator.from_recon_file(self.recon_path)
        return sim.run(strategy)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--save", help="Save plan as JSON")
    args = parser.parse_args()

    t0 = time.time()
    planner = BatchedPlanner(args.recon)
    plan = planner.plan()
    result = planner.validate(plan)
    elapsed = time.time() - t0

    print(f"\nSim: score={result['score']} orders={result['orders_completed']} "
          f"items={result['items_delivered']} rounds={result['rounds_used']}")
    print(f"Time: {elapsed:.1f}s")

    if args.save:
        Path(args.save).write_text(json.dumps(plan, indent=2))
