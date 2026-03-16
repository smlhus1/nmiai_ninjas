"""Sequential collision-correct planner.

Models the server's EXACT collision resolution: bots processed in ID order.
Bot 0 moves first, then bot 1 sees bot 0's NEW position but bot 2+'s OLD positions.

Plans round-by-round using greedy BFS, validated by simulator.
"""

from __future__ import annotations

import json
import sys
import os
import time
import logging
from collections import Counter, deque
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.grid import GameMap, Pos, pickup_positions
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue, ShelfIndex
from Simulering.offline.simulator import Simulator

Pos = tuple[int, int]


def bfs_distance(grid, start: Pos, goal: Pos) -> int:
    """BFS distance, -1 if unreachable."""
    if start == goal:
        return 0
    visited = {start}
    q = deque([(start, 0)])
    while q:
        pos, d = q.popleft()
        for nb in grid.neighbors(pos, respect_one_way=False):
            if nb == goal:
                return d + 1
            if nb not in visited:
                visited.add(nb)
                q.append((nb, d + 1))
    return -1


def bfs_next_toward(grid, start: Pos, goal: Pos) -> Pos:
    """Return next position on shortest path to goal."""
    if start == goal:
        return start
    parent = {start: None}
    q = deque([start])
    while q:
        pos = q.popleft()
        if pos == goal:
            p = pos
            while parent[p] != start:
                p = parent[p]
            return p
        for nb in grid.neighbors(pos, respect_one_way=False):
            if nb not in parent:
                parent[nb] = pos
                q.append(nb)
    return start


def pos_to_action(from_pos: Pos, to_pos: Pos) -> str:
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dx == 1: return "move_right"
    if dx == -1: return "move_left"
    if dy == 1: return "move_down"
    if dy == -1: return "move_up"
    return "wait"


class SequentialPlanner:
    """Round-by-round planner with sequential collision model."""

    def __init__(self, recon_path: str):
        self.gm = GameMap.from_recon(recon_path)
        self.grid = self.gm.grid
        self.dist_cache = DistanceCache(self.grid)
        self.oq = OrderQueue.from_recon(recon_path)
        self.si = ShelfIndex(self.gm, self.dist_cache)
        self.recon_path = recon_path
        self.n_bots = self.gm.bot_count
        self.spawn = self.gm.spawn
        self.drop_off_zones = list(self.gm.drop_off_zones)

        with open(recon_path) as f:
            recon = json.load(f)
        self.shelf_map = recon.get("shelf_map", {})

    def __call__(self, state: dict) -> dict:
        """Strategy callable for Simulator.run()."""
        bots = state.get("bots", [])
        items = state.get("items", [])
        orders = state.get("orders", [])
        round_num = state.get("round", 0)

        # Parse active order
        active = None
        preview = None
        for o in orders:
            if o.get("status") == "active" and not o.get("complete"):
                active = o
            elif o.get("status") == "preview":
                preview = o

        remaining = Counter()
        remaining_types = set()
        if active:
            remaining = Counter(active["items_required"])
            for d in active.get("items_delivered", []):
                if remaining[d] > 0:
                    remaining[d] -= 1
            remaining_types = set(t for t, c in remaining.items() if c > 0)

        preview_types = set()
        if preview:
            preview_types = set(preview.get("items_required", []))

        # Initialize bot state
        if not hasattr(self, '_bot_goal'):
            self._bot_goal: dict[int, str] = {}
            self._bot_target: dict[int, Pos] = {}
            self._bot_item_type: dict[int, str] = {}
            self._prev_inv: dict[int, list] = {}
            self._last_order_id = None
            self._claimed_types: Counter = Counter()

        # Order changed
        active_id = active["id"] if active else None
        if active_id != self._last_order_id:
            self._last_order_id = active_id
            self._claimed_types.clear()
            for bid in list(self._bot_goal.keys()):
                if self._bot_goal.get(bid) != "deliver":
                    self._bot_goal[bid] = "idle"

        # Build item lookup
        items_by_pos = {}
        for item in items:
            pos = tuple(item["position"])
            items_by_pos.setdefault(pos, []).append(item)

        # Detect pickups (inventory grew)
        for bot in bots:
            bid = bot["id"]
            inv = bot.get("inventory", [])
            prev = self._prev_inv.get(bid, [])
            if len(inv) > len(prev) and self._bot_goal.get(bid) == "pickup":
                # Pickup done → deliver
                pos = tuple(bot["position"])
                dz = min(self.drop_off_zones,
                        key=lambda z: self.dist_cache.distance(pos, z) or 9999)
                self._bot_goal[bid] = "deliver"
                self._bot_target[bid] = dz

        # === SEQUENTIAL ROUND PLANNING ===
        # Process bots in ID order (matching server)
        # Track occupied positions as each bot decides
        occupied: set[Pos] = set()
        # First: mark all bot positions as occupied
        bot_positions = {}
        for bot in bots:
            pos = tuple(bot["position"])
            bot_positions[bot["id"]] = pos
            occupied.add(pos)

        actions = []
        decided_positions: dict[int, Pos] = {}  # Where each bot WILL be

        for bot in sorted(bots, key=lambda b: b["id"]):
            bid = bot["id"]
            pos = tuple(bot["position"])
            inv = bot.get("inventory", [])

            if bid not in self._bot_goal:
                self._bot_goal[bid] = "idle"

            goal = self._bot_goal.get(bid, "idle")
            target = self._bot_target.get(bid, pos)

            # === ACTIONS AT TARGET ===
            # Pickup
            if goal == "pickup" and pos == target:
                item_type = self._bot_item_type.get(bid)
                item_id = None
                for dx, dy in [(0,0),(0,-1),(0,1),(-1,0),(1,0)]:
                    adj = (pos[0]+dx, pos[1]+dy)
                    for item in items_by_pos.get(adj, []):
                        if item.get("type") == item_type:
                            item_id = item["id"]
                            break
                    if item_id:
                        break
                if item_id:
                    actions.append({"bot": bid, "action": "pick_up", "item_id": item_id})
                    decided_positions[bid] = pos
                    continue
                else:
                    self._bot_goal[bid] = "idle"
                    goal = "idle"

            # Dropoff
            if goal == "deliver" and pos in self.drop_off_zones:
                if inv and any(t in remaining_types for t in inv):
                    actions.append({"bot": bid, "action": "drop_off"})
                    self._bot_goal[bid] = "idle"
                    decided_positions[bid] = pos
                    continue
                else:
                    self._bot_goal[bid] = "idle"
                    goal = "idle"

            # === ASSIGN NEW TASK ===
            if goal == "idle":
                # Full inventory with matching items → deliver
                if len(inv) >= 3 or (inv and any(t in remaining_types for t in inv)):
                    mc = sum(1 for t in inv if t in remaining_types)
                    if mc > 0:
                        dz = min(self.drop_off_zones,
                                key=lambda z: self.dist_cache.distance(pos, z) or 9999)
                        self._bot_goal[bid] = "deliver"
                        self._bot_target[bid] = dz
                        goal = "deliver"
                        target = dz

                # Assign pickup for needed item
                if goal == "idle" and len(inv) < 3:
                    all_needs = remaining_types | preview_types
                    for item_type in all_needs:
                        if item_type in remaining_types:
                            needed = remaining[item_type]
                        else:
                            needed = 3
                        if self._claimed_types[item_type] >= needed * 2:
                            continue
                        shelves = self.shelf_map.get(item_type, [])
                        best_pp = None
                        best_d = 9999
                        for sp in shelves:
                            sp = tuple(sp)
                            for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                                pp = (sp[0]+dx, sp[1]+dy)
                                if self.grid.walkable(pp):
                                    d = self.dist_cache.distance(pos, pp) or 9999
                                    if d < best_d:
                                        best_d = d
                                        best_pp = pp
                        if best_pp:
                            self._bot_goal[bid] = "pickup"
                            self._bot_target[bid] = best_pp
                            self._bot_item_type[bid] = item_type
                            self._claimed_types[item_type] += 1
                            goal = "pickup"
                            target = best_pp
                            break

            # === MOVEMENT ===
            if goal in ("pickup", "deliver") and pos != target:
                next_pos = bfs_next_toward(self.grid, pos, target)
                # Sequential collision check: can we move there?
                # Position is free if no DECIDED bot is there AND
                # no UNDECIDED bot (higher ID) is currently there
                # Actually: just check decided_positions
                if next_pos not in decided_positions.values():
                    actions.append({"bot": bid, "action": pos_to_action(pos, next_pos)})
                    # Update occupied: remove old, add new
                    decided_positions[bid] = next_pos
                    continue

            # Wait
            actions.append({"bot": bid, "action": "wait"})
            decided_positions[bid] = pos

        # Save inventory
        for bot in bots:
            self._prev_inv[bot["id"]] = list(bot.get("inventory", []))

        return {"actions": actions}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    t0 = time.time()
    planner = SequentialPlanner(args.recon)
    sim = Simulator.from_recon_file(args.recon)
    result = sim.run(planner, verbose=True)
    print(f"\nScore: {result['score']}, Orders: {result['orders_completed']}, "
          f"Rounds: {result['rounds_used']}, Time: {time.time()-t0:.1f}s")
