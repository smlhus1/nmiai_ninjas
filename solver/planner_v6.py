"""Planner v6: sequential-sim-accurate collision-free planning.

Key insight: sim processes bots in ID order. When bot K moves:
- Bots J<K have ALREADY moved → check their NEW position at t
- Bots J>K have NOT moved yet → check their OLD position (= pos at t-1)

We store both pre-move and post-move positions for each bot at each timestep.

Usage:
    py -m solver.planner_v6 --recon logs/74001e7f_2026-03-17_recon.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.grid import GameMap, Grid, Pos, pickup_positions
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue
from solver.scripted_strategy import ScriptedStrategy
from Simulering.offline.simulator import Simulator


class SeqOccupancy:
    """Sequential-sim-accurate occupancy grid.

    Stores pre_pos[bot][t] and post_pos[bot][t].
    pre = position BEFORE bot moves at round t.
    post = position AFTER bot moves at round t.

    For stationary bot: pre == post.
    For moving bot: pre = old pos, post = new pos.
    """

    def __init__(self, spawn: Pos, dzs: set[Pos], n_bots: int, max_t: int = 500):
        self.spawn = spawn
        self.dzs = dzs
        self.n_bots = n_bots
        self.max_t = max_t
        # pre_pos[bot_id][t] = position before move at t
        # post_pos[bot_id][t] = position after move at t
        self.pre: dict[int, dict[int, Pos]] = {b: {} for b in range(n_bots)}
        self.post: dict[int, dict[int, Pos]] = {b: {} for b in range(n_bots)}

        # Initialize all bots at spawn
        for b in range(n_bots):
            for t in range(max_t):
                self.pre[b][t] = spawn
                self.post[b][t] = spawn

    def set_path(self, bot_id: int, path: list[Pos], start_t: int):
        """Record a bot's path. path[0] = pos at start_t, path[1] = pos at start_t+1."""
        for i, pos in enumerate(path):
            t = start_t + i
            if t >= self.max_t:
                break
            if i == 0:
                self.pre[bot_id][t] = pos
                self.post[bot_id][t] = pos
            else:
                self.pre[bot_id][t] = path[i-1] if path[i-1] != pos else pos
                self.post[bot_id][t] = pos
            # Pre-position at t is post-position at t-1 if we haven't set it
            # Actually: pre at t = where bot IS before processing at t
            # = post at t-1 (result of previous round)
            # For path: bot is at path[i] after round start_t+i
            # So pre[start_t+i] = path[i-1] (where it was), post[start_t+i] = path[i]
            self.pre[bot_id][t] = path[i-1] if i > 0 else pos
            self.post[bot_id][t] = pos

    def set_stay(self, bot_id: int, pos: Pos, from_t: int, to_t: int):
        """Bot stays at pos from from_t to to_t."""
        for t in range(max(0, from_t), min(to_t + 1, self.max_t)):
            self.pre[bot_id][t] = pos
            self.post[bot_id][t] = pos

    def is_free(self, pos: Pos, t: int, bot_id: int) -> bool:
        """Can bot_id move TO pos at time t?

        Sequential sim: bot_id processes in ID order.
        - Bots J < bot_id: already moved → check post[J][t]
        - Bots J > bot_id: not moved yet → check pre[J][t]
        Spawn and DZs are shared (overlap allowed).
        """
        if pos == self.spawn or pos in self.dzs:
            return True
        if t < 0 or t >= self.max_t:
            return False

        for j in range(self.n_bots):
            if j == bot_id:
                continue
            if j < bot_id:
                if self.post[j].get(t, self.spawn) == pos:
                    return False
            else:
                if self.pre[j].get(t, self.spawn) == pos:
                    return False
        return True

    def would_swap(self, pos_from: Pos, pos_to: Pos, t: int, bot_id: int) -> bool:
        """Check if moving from pos_from to pos_to at t would swap with another bot."""
        for j in range(self.n_bots):
            if j == bot_id:
                continue
            # Swap: bot_id goes from A->B, bot J goes from B->A at same time
            j_pre = self.pre[j].get(t, self.spawn)
            j_post = self.post[j].get(t, self.spawn)
            if j_pre == pos_to and j_post == pos_from:
                return True
        return False


def bfs_timed(grid: Grid, occ: SeqOccupancy, start: Pos, end: Pos,
              start_t: int, bot_id: int, max_t: int = 500) -> list[Pos] | None:
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
            if not occ.is_free(nb, nt, bot_id):
                continue
            if nb != pos and occ.would_swap(pos, nb, nt, bot_id):
                continue
            state = (nb, nt)
            if state not in parent:
                parent[state] = (pos, t)
                q.append(state)
    return None


class Planner:
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

        self._shelves: dict[str, list[tuple[Pos, Pos]]] = {}
        for item_type, positions in self.gm.shelf_map.items():
            entries = []
            for shelf in positions:
                for pp in pickup_positions(self.grid, shelf):
                    entries.append((shelf, pp))
            self._shelves[item_type] = entries

    def _find_shelf(self, item_type: str, near: Pos, occ: SeqOccupancy,
                    est_t: int, bot_id: int) -> tuple[Pos, Pos] | None:
        entries = self._shelves.get(item_type, [])
        scored = []
        for shelf, pp in entries:
            d = self.dist.distance(near, pp)
            if d is not None:
                scored.append((d, shelf, pp))
        scored.sort()
        for d, shelf, pp in scored:
            at = est_t + d
            if (occ.is_free(pp, at, bot_id) and
                occ.is_free(pp, at + 1, bot_id) and
                occ.is_free(pp, at + 2, bot_id)):
                return (shelf, pp)
        return (scored[0][1], scored[0][2]) if scored else None

    def plan(self) -> dict[int, dict[int, str]]:
        occ = SeqOccupancy(self.spawn, self.dzs, self.n_bots, self.max_t)
        actions: dict[int, dict[int, str]] = {}

        item_queue: list[str] = []
        for order in self.oq:
            item_queue.extend(order.items_required)

        bot_items: dict[int, list[str]] = {i: [] for i in range(self.n_bots)}
        for i, item in enumerate(item_queue):
            bot_items[(i // 3) % self.n_bots].append(item)

        bot_trips: dict[int, list[list[str]]] = {}
        for bid, items in bot_items.items():
            trips = []
            while items:
                trips.append(items[:3])
                items = items[3:]
            bot_trips[bid] = trips

        print(f"Items: {len(item_queue)}, Bots: {self.n_bots}", flush=True)

        total_items = 0

        for bot_id in range(self.n_bots):
            trips = bot_trips.get(bot_id, [])
            if not trips:
                continue

            cur = self.spawn
            t = bot_id  # stagger

            for trip in trips:
                if t >= self.max_t - 10:
                    break

                trip_ok = True
                for item_type in trip:
                    # Try multiple shelves if first is blocked at actual arrival
                    placed = False
                    entries = self._shelves.get(item_type, [])
                    scored = [(self.dist.distance(cur, pp) or 999, s, pp) for s, pp in entries]
                    scored.sort()

                    for _, shelf, pp in scored[:5]:
                        path = bfs_timed(self.grid, occ, cur, pp, t, bot_id, self.max_t)
                        if path is None:
                            continue

                        arrive = t + len(path) - 1
                        pick_t = arrive + 1

                        # Verify cell is free at arrival AND pickup AND post-pickup
                        if not (occ.is_free(pp, arrive, bot_id) and
                                occ.is_free(pp, pick_t, bot_id) and
                                occ.is_free(pp, pick_t + 1, bot_id)):
                            continue

                        # Commit this path
                        occ.set_path(bot_id, path, t)
                        for j in range(1, len(path)):
                            self._set_move(actions, t + j, bot_id, path[j-1], path[j])
                        occ.set_stay(bot_id, pp, pick_t, min(pick_t + 100, self.max_t))
                        self._set_raw(actions, pick_t, bot_id, f"pick_up:{item_type}")
                        t = pick_t + 1
                        cur = pp
                        placed = True
                        break

                    if not placed:
                        trip_ok = False
                        break

                if not trip_ok:
                    occ.set_stay(bot_id, cur, t, self.max_t)
                    break

                dz = min(self.dz_list, key=lambda z: self.dist.distance(cur, z) or 999)
                path = bfs_timed(self.grid, occ, cur, dz, t, bot_id, self.max_t)
                if path is None:
                    for alt in self.dz_list:
                        path = bfs_timed(self.grid, occ, cur, alt, t, bot_id, self.max_t)
                        if path:
                            dz = alt
                            break
                if path is None:
                    occ.set_stay(bot_id, cur, t, self.max_t)
                    break

                occ.set_path(bot_id, path, t)
                for j in range(1, len(path)):
                    self._set_move(actions, t + j, bot_id, path[j-1], path[j])

                arrive = t + len(path) - 1
                drop_t = arrive + 1
                self._set_raw(actions, drop_t, bot_id, "drop_off")
                occ.set_stay(bot_id, dz, drop_t, drop_t + 2)

                t = drop_t + 1
                cur = dz
                total_items += len(trip)

                # Return to spawn
                ret = bfs_timed(self.grid, occ, dz, self.spawn, t, bot_id, self.max_t)
                if ret and len(ret) > 1:
                    occ.set_path(bot_id, ret, t)
                    for j in range(1, len(ret)):
                        self._set_move(actions, t + j, bot_id, ret[j-1], ret[j])
                    t = t + len(ret) - 1
                    cur = self.spawn

                occ.set_stay(bot_id, cur, t, min(t + 100, self.max_t))

            occ.set_stay(bot_id, cur, t, self.max_t)
            print(f"Bot {bot_id:2d}: ends@{t}, items={total_items}", flush=True)

        print(f"\nPlanned: {total_items} items", flush=True)
        return actions

    def _set_move(self, actions, r, bot_id, from_pos, to_pos):
        if r < 0 or r >= self.max_t:
            return
        dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        move = {(1,0): "move_right", (-1,0): "move_left",
                (0,1): "move_down", (0,-1): "move_up"}.get((dx, dy), "wait")
        self._set_raw(actions, r, bot_id, move)

    def _set_raw(self, actions, r, bot_id, action):
        if r < 0 or r >= self.max_t:
            return
        actions.setdefault(r, {})[bot_id] = action

    def validate(self, plan: dict) -> dict:
        strategy = ScriptedStrategy(plan)
        sim = Simulator.from_recon_file(self.recon_path)
        return sim.run(strategy)


    def plan_iterative(self, passes: int = 5) -> dict[int, dict[int, str]]:
        """Run plan() multiple times. Each pass uses previous pass's
        full bot traces as initial occupancy."""
        actions = self.plan()

        for p in range(2, passes + 1):
            # Build new occ pre-seeded with ALL positions from previous pass
            occ = SeqOccupancy(self.spawn, self.dzs, self.n_bots, self.max_t)
            for bid in range(self.n_bots):
                pp = self.spawn
                for r in range(self.max_t):
                    act = actions.get(r, {}).get(bid)
                    prev = pp
                    if act:
                        if act == 'move_right': pp = (pp[0]+1, pp[1])
                        elif act == 'move_left': pp = (pp[0]-1, pp[1])
                        elif act == 'move_down': pp = (pp[0], pp[1]+1)
                        elif act == 'move_up': pp = (pp[0], pp[1]-1)
                    occ.pre[bid][r] = prev
                    occ.post[bid][r] = pp
                occ.set_stay(bid, pp, r, self.max_t)

            # Re-plan all bots with full knowledge
            actions2: dict[int, dict[int, str]] = {}
            total = 0

            item_queue = []
            for order in self.oq:
                item_queue.extend(order.items_required)
            bot_items = {i: [] for i in range(self.n_bots)}
            for i, item in enumerate(item_queue):
                bot_items[(i // 3) % self.n_bots].append(item)
            bot_trips = {}
            for bid, items in bot_items.items():
                trips = []
                while items:
                    trips.append(items[:3])
                    items = items[3:]
                bot_trips[bid] = trips

            for bot_id in range(self.n_bots):
                trips = bot_trips.get(bot_id, [])
                if not trips:
                    continue
                cur = self.spawn
                t = bot_id

                for trip in trips:
                    if t >= self.max_t - 10:
                        break
                    trip_ok = True
                    for item_type in trip:
                        si = self._find_shelf(item_type, cur, occ, t, bot_id)
                        if si is None:
                            trip_ok = False; break
                        shelf, pp = si
                        path = bfs_timed(self.grid, occ, cur, pp, t, bot_id, self.max_t)
                        if path is None:
                            trip_ok = False; break
                        occ.set_path(bot_id, path, t)
                        for j in range(1, len(path)):
                            self._set_move(actions2, t+j, bot_id, path[j-1], path[j])
                        arrive = t + len(path) - 1
                        pick_t = arrive + 1
                        occ.set_stay(bot_id, pp, pick_t, min(pick_t+100, self.max_t))
                        self._set_raw(actions2, pick_t, bot_id, f"pick_up:{item_type}")
                        t = pick_t + 1; cur = pp

                    if not trip_ok:
                        occ.set_stay(bot_id, cur, t, self.max_t); break

                    dz = min(self.dz_list, key=lambda z: self.dist.distance(cur, z) or 999)
                    path = bfs_timed(self.grid, occ, cur, dz, t, bot_id, self.max_t)
                    if path is None:
                        for alt in self.dz_list:
                            path = bfs_timed(self.grid, occ, cur, alt, t, bot_id, self.max_t)
                            if path: dz = alt; break
                    if path is None:
                        occ.set_stay(bot_id, cur, t, self.max_t); break

                    occ.set_path(bot_id, path, t)
                    for j in range(1, len(path)):
                        self._set_move(actions2, t+j, bot_id, path[j-1], path[j])
                    arrive = t + len(path) - 1
                    drop_t = arrive + 1
                    self._set_raw(actions2, drop_t, bot_id, "drop_off")
                    occ.set_stay(bot_id, dz, drop_t, drop_t+2)
                    t = drop_t + 1; cur = dz; total += len(trip)

                    ret = bfs_timed(self.grid, occ, dz, self.spawn, t, bot_id, self.max_t)
                    if ret and len(ret) > 1:
                        occ.set_path(bot_id, ret, t)
                        for j in range(1, len(ret)):
                            self._set_move(actions2, t+j, bot_id, ret[j-1], ret[j])
                        t = t + len(ret) - 1; cur = self.spawn
                    occ.set_stay(bot_id, cur, t, min(t+100, self.max_t))

                occ.set_stay(bot_id, cur, t, self.max_t)

            print(f"Pass {p}: {total} items", flush=True)
            actions = actions2

        return actions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()

    t0 = time.time()
    planner = Planner(args.recon)
    plan = planner.plan_iterative(args.passes)
    result = planner.validate(plan)
    print(f"Sim: score={result['score']} items={result['items_delivered']} "
          f"orders={result['orders_completed']}")
    print(f"Time: {time.time()-t0:.1f}s")
