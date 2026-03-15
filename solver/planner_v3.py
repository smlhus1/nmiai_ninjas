"""Offline planner v3: uses the REAL simulator for collision detection.

Instead of a separate reservation table, this planner:
1. Generates a candidate plan (greedy item assignment + BFS paths)
2. Runs it through the REAL sim to get actual score
3. Mutates the plan and re-tests
4. Keeps the best plan

The sim IS the collision model. No mismatch possible.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import os
import copy
import random
from collections import deque, Counter
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.grid import GameMap, Pos, pickup_positions
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue, ShelfIndex
from solver.scripted_strategy import ScriptedStrategy
from Simulering.offline.simulator import Simulator


def bfs_path(grid, start: Pos, end: Pos) -> list[Pos]:
    """BFS shortest path, returns [start, ..., end]."""
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


class TripPlan:
    """A single bot's trip: go to pickup, pick, go to dropoff, drop."""
    def __init__(self, bot_id: int, item_type: str, pickup_pos: Pos, dropoff: Pos):
        self.bot_id = bot_id
        self.item_type = item_type
        self.pickup_pos = pickup_pos
        self.dropoff = dropoff


class GamePlan:
    """Full game plan: ordered list of trips to execute."""
    def __init__(self, trips: list[TripPlan]):
        self.trips = list(trips)

    def to_scripted(self, grid, spawn: Pos, n_bots: int) -> dict[int, dict[int, str]]:
        """Convert to round-action plan for ScriptedStrategy."""
        actions: dict[int, dict[int, str]] = {}
        bot_pos = {i: spawn for i in range(n_bots)}
        bot_available = {i: 0 for i in range(n_bots)}

        for trip in self.trips:
            bid = trip.bot_id
            start_r = bot_available[bid]

            # Path to pickup
            path_to_pickup = bfs_path(grid, bot_pos[bid], trip.pickup_pos)
            for i in range(1, len(path_to_pickup)):
                r = start_r + i
                self._set_move(actions, r, bid, path_to_pickup[i-1], path_to_pickup[i])

            # Pickup action (after arriving)
            arrive_r = start_r + len(path_to_pickup) - 1
            pickup_r = arrive_r + 1 if len(path_to_pickup) > 1 else arrive_r
            self._set_raw(actions, pickup_r, bid, "pick_up")

            # Path to dropoff
            path_to_dz = bfs_path(grid, trip.pickup_pos, trip.dropoff)
            dz_start = pickup_r + 1
            for i in range(1, len(path_to_dz)):
                r = dz_start + i
                self._set_move(actions, r, bid, path_to_dz[i-1], path_to_dz[i])

            # Dropoff action
            arrive_dz = dz_start + len(path_to_dz) - 1
            dropoff_r = arrive_dz + 1 if len(path_to_dz) > 1 else arrive_dz
            self._set_raw(actions, dropoff_r, bid, "drop_off")

            bot_pos[bid] = trip.dropoff
            bot_available[bid] = dropoff_r + 1

        return actions

    def mutate(self) -> GamePlan:
        """Create a mutated copy of this plan."""
        new_trips = list(self.trips)

        r = random.random()
        if r < 0.3 and len(new_trips) > 1:
            # Swap two trips' bot assignments
            i, j = random.sample(range(len(new_trips)), 2)
            t1, t2 = new_trips[i], new_trips[j]
            new_trips[i] = TripPlan(t2.bot_id, t1.item_type, t1.pickup_pos, t1.dropoff)
            new_trips[j] = TripPlan(t1.bot_id, t2.item_type, t2.pickup_pos, t2.dropoff)
        elif r < 0.6 and len(new_trips) > 2:
            # Swap order of two adjacent trips
            i = random.randint(0, len(new_trips) - 2)
            new_trips[i], new_trips[i+1] = new_trips[i+1], new_trips[i]
        elif r < 0.9:
            # Change bot assignment for a random trip
            i = random.randint(0, len(new_trips) - 1)
            new_bot = random.randint(0, 19)
            t = new_trips[i]
            new_trips[i] = TripPlan(new_bot, t.item_type, t.pickup_pos, t.dropoff)

        return GamePlan(new_trips)

    @staticmethod
    def _set_move(actions, r, bid, from_pos, to_pos):
        if r >= 500:
            return
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        if dx == 1: act = "move_right"
        elif dx == -1: act = "move_left"
        elif dy == 1: act = "move_down"
        elif dy == -1: act = "move_up"
        else: act = "wait"
        GamePlan._set_raw(actions, r, bid, act)

    @staticmethod
    def _set_raw(actions, r, bid, act):
        if r >= 500:
            return
        if r not in actions:
            actions[r] = {}
        actions[r][bid] = act


def generate_initial_plan(gm: GameMap, oq: OrderQueue, si: ShelfIndex, dc: DistanceCache) -> GamePlan:
    """Generate greedy initial plan: assign nearest item to nearest bot."""
    trips = []
    bot_pos = {i: gm.spawn for i in range(gm.bot_count)}
    bot_round = {i: 0 for i in range(gm.bot_count)}

    for order in oq:
        for item_type in order.items_required:
            # Find best bot + shelf combo
            best_bot = None
            best_cost = float("inf")
            best_pickup = None
            best_dz = None

            for dz in gm.drop_off_zones:
                entries = si.get(item_type, dz)
                if not entries:
                    continue
                entry = entries[0]

                for bid in range(gm.bot_count):
                    d = dc.distance(bot_pos[bid], entry.pickup_pos)
                    if d is None:
                        continue
                    d_back = entry.distance_to_dropoff
                    cost = d + d_back + bot_round[bid]  # include wait time
                    if cost < best_cost:
                        best_cost = cost
                        best_bot = bid
                        best_pickup = entry.pickup_pos
                        best_dz = dz

            if best_bot is not None:
                trips.append(TripPlan(best_bot, item_type, best_pickup, best_dz))
                # Update bot state
                trip_time = dc.distance(bot_pos[best_bot], best_pickup) or 10
                trip_time += dc.distance(best_pickup, best_dz) or 10
                trip_time += 2  # pickup + dropoff rounds
                bot_pos[best_bot] = best_dz
                bot_round[best_bot] += trip_time

    return GamePlan(trips)


def evaluate(plan: GamePlan, grid, spawn, n_bots, recon_path) -> int:
    """Run plan through real sim, return score."""
    scripted = plan.to_scripted(grid, spawn, n_bots)
    strategy = ScriptedStrategy(scripted)
    sim = Simulator.from_recon_file(recon_path)
    result = sim.run(strategy)
    return result["score"]


def evolve_plan(recon_path: str, generations: int = 100, population: int = 10):
    """Evolve plans using real sim for fitness evaluation."""
    gm = GameMap.from_recon(recon_path)
    dc = DistanceCache(gm.grid)
    oq = OrderQueue.from_recon(recon_path)
    si = ShelfIndex(gm, dc)

    # Generate initial plan
    base = generate_initial_plan(gm, oq, si, dc)
    base_score = evaluate(base, gm.grid, gm.spawn, gm.bot_count, recon_path)
    print(f"Initial plan: {base_score} score, {len(base.trips)} trips", flush=True)

    best_plan = base
    best_score = base_score

    for gen in range(generations):
        # Generate mutations
        candidates = [best_plan.mutate() for _ in range(population)]
        scores = [evaluate(c, gm.grid, gm.spawn, gm.bot_count, recon_path) for c in candidates]

        gen_best_idx = max(range(len(scores)), key=lambda i: scores[i])
        gen_best = scores[gen_best_idx]

        if gen_best > best_score:
            best_score = gen_best
            best_plan = candidates[gen_best_idx]
            print(f"Gen {gen}: NEW BEST {best_score} ***", flush=True)
        elif gen % 10 == 0:
            print(f"Gen {gen}: best={best_score}, gen_best={gen_best}", flush=True)

    return best_plan, best_score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--gens", type=int, default=100)
    parser.add_argument("--pop", type=int, default=10)
    args = parser.parse_args()

    t0 = time.time()
    plan, score = evolve_plan(args.recon, generations=args.gens, population=args.pop)
    elapsed = time.time() - t0

    print(f"\nBest score: {score}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save as MAPF-compatible plan
    scripted = plan.to_scripted(
        GameMap.from_recon(args.recon).grid,
        GameMap.from_recon(args.recon).spawn,
        GameMap.from_recon(args.recon).bot_count,
    )
    Path("logs/offline_plan_v3.json").write_text(json.dumps(scripted, default=str))
    print(f"Plan saved to logs/offline_plan_v3.json")
