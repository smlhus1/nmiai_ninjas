"""Offline MAPF planner v2: parallel bot planning with time-space A*.

Plans ALL bots simultaneously. Each bot gets a sequence of trips:
pickup1 -> pickup2 -> ... -> dropoff, then next trip.

Validates against real simulator for accurate scoring.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import os
from collections import Counter, deque
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.grid import GameMap, Pos, pickup_positions
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue, ShelfIndex
from solver.time_space import ReservationTable, time_space_astar
from solver.scripted_strategy import ScriptedStrategy
from Simulering.offline.simulator import Simulator


def bfs_path(grid, start: Pos, end: Pos) -> list[Pos]:
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


class ParallelPlanner:
    def __init__(self, recon_path: str):
        self.gm = GameMap.from_recon(recon_path)
        self.grid = self.gm.grid
        self.dist_cache = DistanceCache(self.grid)
        self.oq = OrderQueue.from_recon(recon_path)
        self.si = ShelfIndex(self.gm, self.dist_cache)
        self.recon_path = recon_path
        self.n_bots = self.gm.bot_count
        self.spawn = self.gm.spawn
        self.max_rounds = 500
        self.grid_nb = lambda pos: self.grid.neighbors(pos, respect_one_way=False)

    def plan(self) -> dict[int, dict[int, str]]:
        """Plan all bot movements for all rounds."""
        reservations = ReservationTable()
        bot_pos = {i: self.spawn for i in range(self.n_bots)}
        bot_available_at = {i: 0 for i in range(self.n_bots)}

        # Full action plan: round -> bot_id -> action
        actions: dict[int, dict[int, str]] = {}

        # DON'T reserve spawn at t=0 — all bots stack there, reservation
        # would block everyone. Instead, let each bot's first path handle it.
        # Lower ID bots plan first (matching server collision priority).

        orders = list(self.oq)
        order_idx = 0
        items_delivered = 0
        orders_completed = 0

        while order_idx < len(orders) and bot_available_at[0] < self.max_rounds - 5:
            order = orders[order_idx]
            items_needed = list(order.items_required)

            # Find earliest round when we can start this order
            # (= when enough bots are free to handle all items)
            earliest = min(bot_available_at.values())

            # Assign items to bots greedily: nearest available bot for each item
            item_assignments = self._assign_items(items_needed, bot_pos, bot_available_at)

            if not item_assignments:
                order_idx += 1
                continue

            # Plan each bot's trip: pickup -> dropoff
            order_done_round = earliest
            for bot_id, item_type, pickup_pos, dz in item_assignments:
                start_t = bot_available_at[bot_id]

                # Plan path to pickup
                path_to_pickup = time_space_astar(
                    self.grid_nb, bot_pos[bot_id], pickup_pos,
                    start_t, bot_id, reservations, self.max_rounds,
                )
                if path_to_pickup is None:
                    path_to_pickup = bfs_path(self.grid, bot_pos[bot_id], pickup_pos)

                # Write movement actions
                for i in range(1, len(path_to_pickup)):
                    r = start_t + i
                    self._set_action(actions, r, bot_id, path_to_pickup[i-1], path_to_pickup[i])
                reservations.reserve_path(bot_id, path_to_pickup, start_t)

                # Pickup action — AFTER arriving (separate round)
                arrive_r = start_t + len(path_to_pickup) - 1
                pickup_r = arrive_r + 1
                self._set_action_raw(actions, pickup_r, bot_id, "pick_up")
                reservations.reserve(bot_id, pickup_pos, pickup_r)

                # Plan path to dropoff — starts AFTER pickup
                dz_start = pickup_r + 1
                path_to_dz = time_space_astar(
                    self.grid_nb, pickup_pos, dz,
                    dz_start, bot_id, reservations, self.max_rounds,
                )
                if path_to_dz is None:
                    path_to_dz = bfs_path(self.grid, pickup_pos, dz)

                for i in range(1, len(path_to_dz)):
                    r = dz_start + i
                    self._set_action(actions, r, bot_id, path_to_dz[i-1], path_to_dz[i])
                reservations.reserve_path(bot_id, path_to_dz, dz_start)

                # Dropoff action — AFTER arriving
                arrive_dz_r = dz_start + len(path_to_dz) - 1
                dropoff_r = arrive_dz_r + 1
                self._set_action_raw(actions, dropoff_r, bot_id, "drop_off")
                reservations.reserve(bot_id, dz, dropoff_r)

                # Update bot state
                bot_pos[bot_id] = dz
                bot_available_at[bot_id] = dropoff_r + 1
                order_done_round = max(order_done_round, dropoff_r)

            order_idx += 1
            orders_completed += 1
            items_delivered += len(item_assignments)

            if order_idx % 5 == 0:
                print(f"Order {order_idx}: {orders_completed} done, "
                      f"round {order_done_round}, est score {items_delivered + orders_completed*5}",
                      flush=True)

        print(f"\nPlanned {orders_completed} orders, {items_delivered} items, "
              f"est score {items_delivered + orders_completed*5}", flush=True)
        return actions

    def _assign_items(self, items_needed, bot_pos, bot_available_at):
        """Assign items to bots — nearest available bot for each item."""
        assignments = []
        used_bots = set()

        for item_type in items_needed:
            best_bot = None
            best_cost = float("inf")
            best_pickup = None
            best_dz = None

            for dz in self.gm.drop_off_zones:
                entries = self.si.get(item_type, dz)
                if not entries:
                    continue
                entry = entries[0]

                for bid in range(self.n_bots):
                    if bid in used_bots:
                        continue
                    d = self.dist_cache.distance(bot_pos[bid], entry.pickup_pos)
                    if d is None:
                        continue
                    # Cost = travel + wait time
                    cost = d + bot_available_at[bid]
                    if cost < best_cost:
                        best_cost = cost
                        best_bot = bid
                        best_pickup = entry.pickup_pos
                        best_dz = dz

            if best_bot is not None:
                assignments.append((best_bot, item_type, best_pickup, best_dz))
                used_bots.add(best_bot)

        return assignments

    def _compute_dispersal_targets(self) -> dict[int, Pos]:
        """Compute spread-out targets for spawn dispersal."""
        # BFS from spawn to find N nearest distinct walkable cells
        visited = set()
        q = deque([(self.spawn, 0)])
        visited.add(self.spawn)
        targets = []

        while q and len(targets) < self.n_bots:
            pos, dist = q.popleft()
            if dist > 0:
                targets.append(pos)
            for nb in self.grid_nb(pos):
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, dist + 1))

        result = {}
        for i in range(self.n_bots):
            result[i] = targets[i] if i < len(targets) else self.spawn
        return result

    def _set_action(self, actions, r, bot_id, from_pos, to_pos):
        """Set movement action based on position change."""
        if r >= self.max_rounds:
            return
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        if dx == 1:
            act = "move_right"
        elif dx == -1:
            act = "move_left"
        elif dy == 1:
            act = "move_down"
        elif dy == -1:
            act = "move_up"
        else:
            act = "wait"
        self._set_action_raw(actions, r, bot_id, act)

    def _set_action_raw(self, actions, r, bot_id, action):
        if r >= self.max_rounds:
            return
        if r not in actions:
            actions[r] = {}
        actions[r][bot_id] = action

    def validate(self, plan: dict) -> dict:
        """Run plan through simulator, return result dict."""
        strategy = ScriptedStrategy(plan)
        sim = Simulator.from_recon_file(self.recon_path)
        result = sim.run(strategy)
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    t0 = time.time()
    planner = ParallelPlanner(args.recon)
    plan = planner.plan()
    result = planner.validate(plan)
    elapsed = time.time() - t0

    print(f"\nSim score: {result['score']}, orders: {result['orders_completed']}")
    print(f"Time: {elapsed:.1f}s")
