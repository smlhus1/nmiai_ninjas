"""Planner v7: sim-in-the-loop with priority-based collision resolution.

Runs the ACTUAL simulator step by step. Each round:
1. For each bot, determine target (next pickup or dropoff)
2. BFS from ACTUAL position to target, avoiding other bots
3. Lower-ID bots have priority (move first in sim)
4. Higher-ID bots yield if blocked

This is like V2+PIBT but with pre-computed trip assignments
instead of reactive OrderSolver decisions.

Usage:
    py -m solver.planner_v7 --recon logs/74001e7f_2026-03-17_recon.json
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

from solver.grid import GameMap, Pos, pickup_positions
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue
from Simulering.offline.simulator import Simulator


@dataclass
class BotState:
    trip_idx: int = 0          # which trip we're on
    pick_idx: int = 0          # which item in current trip
    phase: str = "pickup"      # pickup / deliver / idle
    target: Pos | None = None


class SimLoopPlanner:
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

    def _best_shelf(self, item_type: str, near: Pos) -> tuple[Pos, Pos] | None:
        best, best_d = None, 9999
        for shelf in self.gm.shelf_map.get(item_type, []):
            for pp in pickup_positions(self.grid, shelf):
                d = self.dist.distance(near, pp)
                if d is not None and d < best_d:
                    best_d, best = d, (shelf, pp)
        return best

    def _nearest_dz(self, pos: Pos) -> Pos:
        return min(self.dz_list, key=lambda z: self.dist.distance(pos, z) or 999)

    def _make_trips(self, items: list[str]) -> list[list[tuple[str, Pos, Pos]]]:
        remaining = list(items)
        trips = []
        while remaining:
            trip = []
            cur = self.spawn
            for _ in range(min(3, len(remaining))):
                bi, bd = -1, 9999
                for i, it in enumerate(remaining):
                    r = self._best_shelf(it, cur)
                    if r:
                        d = self.dist.distance(cur, r[1])
                        if d is not None and d < bd:
                            bi, bd = i, d
                if bi < 0:
                    break
                it = remaining.pop(bi)
                r = self._best_shelf(it, cur)
                trip.append((it, r[0], r[1]))
                cur = r[1]
            if trip:
                trips.append(trip)
        return trips

    def run(self) -> dict:
        sim = Simulator.from_recon_file(self.recon_path)
        state = sim.reset()

        # Build trip queue per bot
        all_trips = []
        for order in self.oq:
            all_trips.extend(self._make_trips(list(order.items_required)))

        bot_trips = {i: [] for i in range(self.n_bots)}
        for i, trip in enumerate(all_trips):
            bot_trips[i % self.n_bots].append(trip)

        # Bot states
        bs = {i: BotState() for i in range(self.n_bots)}

        # Initialize targets
        for bid in range(self.n_bots):
            self._advance(bid, bs[bid], bot_trips[bid], self.spawn)

        for rnd in range(sim.max_rounds):
            sd = state.to_dict()
            bots = {b["id"]: b for b in sd["bots"]}
            items_map = sd["items"]

            # Compute planned next positions (lower ID first)
            next_pos: dict[int, Pos] = {}
            actions = []

            for bid in range(self.n_bots):
                bot = bots[bid]
                pos = tuple(bot["position"])
                inv = bot["inventory"]
                st = bs[bid]

                # Check if just completed pickup (inventory grew)
                if st.phase == "pickup" and st.target:
                    trips = bot_trips[bid]
                    if st.trip_idx < len(trips):
                        trip = trips[st.trip_idx]
                        if st.pick_idx < len(trip):
                            item_type, shelf, pp = trip[st.pick_idx]
                            # Adjacent to shelf? Try pickup
                            if abs(pos[0]-shelf[0]) + abs(pos[1]-shelf[1]) == 1:
                                item_id = next(
                                    (i["id"] for i in items_map
                                     if i["type"] == item_type and tuple(i["position"]) == shelf),
                                    None)
                                if item_id:
                                    actions.append({"bot": bid, "action": "pick_up", "item_id": item_id})
                                    next_pos[bid] = pos
                                    st.pick_idx += 1
                                    if st.pick_idx >= len(trip):
                                        st.phase = "deliver"
                                        st.target = self._nearest_dz(pos)
                                    else:
                                        st.target = trip[st.pick_idx][2]
                                    continue

                # Deliver phase: at dropoff?
                if st.phase == "deliver" and pos in self.dzs:
                    actions.append({"bot": bid, "action": "drop_off"})
                    next_pos[bid] = pos
                    # After drop, check if inventory empty
                    st.trip_idx += 1
                    st.pick_idx = 0
                    self._advance(bid, st, bot_trips[bid], pos)
                    continue

                # Navigate toward target
                target = st.target
                if target is None or pos == target:
                    actions.append({"bot": bid, "action": "wait"})
                    next_pos[bid] = pos
                    continue

                # BFS avoiding ALL other bots' current positions
                all_bot_positions = {tuple(bots[b]["position"]) for b in bots if b != bid}
                path = self._bfs_avoid(pos, target, all_bot_positions)

                if len(path) >= 2:
                    nxt = path[1]
                    # Don't move into a cell occupied by another bot
                    if nxt not in all_bot_positions or nxt == target:
                        dx, dy = nxt[0]-pos[0], nxt[1]-pos[1]
                        move = {(0,-1):"move_up",(0,1):"move_down",
                                (-1,0):"move_left",(1,0):"move_right"}.get((dx,dy),"wait")
                        actions.append({"bot": bid, "action": move})
                        next_pos[bid] = nxt
                    else:
                        actions.append({"bot": bid, "action": "wait"})
                        next_pos[bid] = pos
                else:
                    actions.append({"bot": bid, "action": "wait"})
                    next_pos[bid] = pos

            state, done = sim.step(actions)

            # Post-step: update phases based on actual inventory
            nd = state.to_dict()
            for bot in nd["bots"]:
                bid = bot["id"]
                st = bs[bid]
                if st.phase == "deliver" and len(bot["inventory"]) == 0:
                    st.trip_idx += 1
                    st.pick_idx = 0
                    self._advance(bid, st, bot_trips[bid], tuple(bot["position"]))

            if done:
                break
            if rnd % 100 == 0:
                tc = {}
                for s in bs.values():
                    tc[s.phase] = tc.get(s.phase, 0) + 1
                print(f"R{rnd}: score={sim._score} {tc}", flush=True)

        result = {"score": sim._score, "orders": sim._orders_completed,
                  "items": sim._items_delivered, "rounds": sim._round}
        print(f"\nFinal: score={result['score']} orders={result['orders']} "
              f"items={result['items']} rounds={result['rounds']}", flush=True)
        return result

    def _advance(self, bid: int, st: BotState, trips: list, cur: Pos):
        """Set next target for bot."""
        if st.trip_idx >= len(trips):
            st.phase = "idle"
            st.target = None
            return
        trip = trips[st.trip_idx]
        if st.pick_idx < len(trip):
            st.phase = "pickup"
            st.target = trip[st.pick_idx][2]  # pickup position
        else:
            st.phase = "deliver"
            st.target = self._nearest_dz(cur)

    def _bfs_avoid(self, start: Pos, end: Pos, avoid: set[Pos]) -> list[Pos]:
        """BFS avoiding claimed positions."""
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
            for nb in self.grid.neighbors(pos, respect_one_way=False):
                if nb not in parent and (nb not in avoid or nb == end):
                    parent[nb] = pos
                    q.append(nb)
        # Fallback: ignore avoidance
        return self._bfs_plain(start, end)

    def _bfs_plain(self, start: Pos, end: Pos) -> list[Pos]:
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
            for nb in self.grid.neighbors(pos, respect_one_way=False):
                if nb not in parent:
                    parent[nb] = pos
                    q.append(nb)
        return [start]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    t0 = time.time()
    planner = SimLoopPlanner(args.recon)
    result = planner.run()
    print(f"Time: {time.time()-t0:.1f}s")
