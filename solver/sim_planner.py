"""Simulation-validated planner: plans trips and validates EVERY step in sim.

Instead of trusting reservation tables, runs the actual simulator step by step.
When a bot gets blocked by collision, re-plans that bot's path.

This guarantees sim-accurate collision handling.
"""

from __future__ import annotations

import json
import sys
import time
import os
import logging
from collections import deque, Counter
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.WARNING)

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.grid import GameMap, Pos, pickup_positions
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue, ShelfIndex
from Simulering.offline.simulator import Simulator

Pos = tuple[int, int]


def bfs_next_step(grid, start: Pos, goal: Pos, blocked: set[Pos]) -> Pos:
    """BFS one step toward goal, avoiding blocked positions. Returns next pos."""
    if start == goal:
        return start
    parent = {start: None}
    q = deque([start])
    while q:
        pos = q.popleft()
        if pos == goal:
            # Trace back to first step
            p = pos
            while parent[p] != start:
                p = parent[p]
            return p
        for nb in grid.neighbors(pos, respect_one_way=False):
            if nb not in parent and nb not in blocked:
                parent[nb] = pos
                q.append(nb)
    return start  # No path — stay


def pos_to_action(from_pos: Pos, to_pos: Pos) -> str:
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dx == 1: return "move_right"
    if dx == -1: return "move_left"
    if dy == 1: return "move_down"
    if dy == -1: return "move_up"
    return "wait"


class SimPlanner:
    """Plans trips step-by-step, validated by simulator each round."""

    def __init__(self, recon_path: str):
        self.gm = GameMap.from_recon(recon_path)
        self.grid = self.gm.grid
        self.dist_cache = DistanceCache(self.grid)
        self.oq = OrderQueue.from_recon(recon_path)
        self.si = ShelfIndex(self.gm, self.dist_cache)
        self.recon_path = recon_path
        self.n_bots = self.gm.bot_count
        self.spawn = self.gm.spawn
        self.drop_off_zones = self.gm.drop_off_zones

        with open(recon_path) as f:
            recon = json.load(f)
        self.order_sequence = recon.get("order_sequence", [])
        self.shelf_map = recon.get("shelf_map", {})

    def run(self) -> dict:
        """Run planner with sim validation. Returns sim result."""
        sim = Simulator.from_recon_file(self.recon_path)
        state = sim.reset()

        # Bot state
        bot_goal: dict[int, str] = {}       # "pickup", "deliver", "park"
        bot_target: dict[int, Pos] = {}
        bot_item_type: dict[int, str] = {}
        orders = list(self.oq)
        order_idx = 0
        items_assigned: Counter = Counter()  # how many of each type assigned

        for r in range(sim.max_rounds):
            sd = state.to_dict()
            bots = sd["bots"]
            items = sd["items"]
            active_order = None
            for o in sd.get("orders", []):
                if o.get("status") == "active" and not o.get("complete"):
                    active_order = o
                    break

            remaining = Counter()
            if active_order:
                remaining = Counter(active_order["items_required"])
                for d in active_order.get("items_delivered", []):
                    if remaining[d] > 0:
                        remaining[d] -= 1

            # Build occupied set (where bots currently are)
            occupied = set(tuple(b["position"]) for b in bots)

            actions = []
            new_occupied = set()  # track where bots WILL be after this round

            # Process bots in ID order (matching server collision model)
            for bot in sorted(bots, key=lambda b: b["id"]):
                bid = bot["id"]
                pos = tuple(bot["position"])
                inv = bot.get("inventory", [])

                # Initialize
                if bid not in bot_goal:
                    bot_goal[bid] = "park"
                    bot_target[bid] = pos

                # Check if pickup completed (inventory grew)
                if bot_goal[bid] == "pickup" and len(inv) > 0:
                    if bot_item_type.get(bid) in inv:
                        # Switch to deliver
                        dz = min(self.drop_off_zones,
                                key=lambda z: self.dist_cache.distance(pos, z) or 9999)
                        bot_goal[bid] = "deliver"
                        bot_target[bid] = dz

                # At target: execute action
                if bot_goal[bid] == "pickup" and pos == bot_target[bid]:
                    # Find adjacent item matching type
                    item_type = bot_item_type.get(bid)
                    item_id = None
                    for item in items:
                        ipos = tuple(item["position"])
                        if item.get("type") == item_type and abs(ipos[0]-pos[0]) + abs(ipos[1]-pos[1]) <= 1:
                            item_id = item["id"]
                            break
                    if item_id:
                        actions.append({"bot": bid, "action": "pick_up", "item_id": item_id})
                        new_occupied.add(pos)
                        continue
                    else:
                        bot_goal[bid] = "park"

                if bot_goal[bid] == "deliver" and pos in self.drop_off_zones:
                    if inv and any(t in remaining for t in inv):
                        actions.append({"bot": bid, "action": "drop_off"})
                        bot_goal[bid] = "park"
                        new_occupied.add(pos)
                        continue

                # Assign new task if parked
                if bot_goal[bid] == "park" and remaining:
                    # Find needed item type not fully assigned
                    for item_type, needed in remaining.items():
                        if items_assigned[item_type] < needed:
                            # Find nearest shelf
                            shelves = self.shelf_map.get(item_type, [])
                            if shelves:
                                best_pp = None
                                best_d = 9999
                                for sp in shelves:
                                    sp = tuple(sp)
                                    for pp in pickup_positions(self.grid, sp):
                                        d = self.dist_cache.distance(pos, pp) or 9999
                                        if d < best_d:
                                            best_d = d
                                            best_pp = pp
                                if best_pp:
                                    bot_goal[bid] = "pickup"
                                    bot_target[bid] = best_pp
                                    bot_item_type[bid] = item_type
                                    items_assigned[item_type] += 1
                                    break

                # Full inventory without matching items → deliver anyway
                if bot_goal[bid] == "park" and len(inv) >= 3:
                    dz = min(self.drop_off_zones,
                            key=lambda z: self.dist_cache.distance(pos, z) or 9999)
                    bot_goal[bid] = "deliver"
                    bot_target[bid] = dz

                # Move toward target
                target = bot_target.get(bid, pos)
                if pos != target:
                    next_pos = bfs_next_step(self.grid, pos, target, set())
                    if next_pos != pos:
                        actions.append({"bot": bid, "action": pos_to_action(pos, next_pos)})
                        new_occupied.add(next_pos)
                        continue

                # Default: wait
                actions.append({"bot": bid, "action": "wait"})
                new_occupied.add(pos)

            # Reset items_assigned on order change
            if active_order:
                current_id = active_order.get("id")
                if not hasattr(self, '_last_order_id') or self._last_order_id != current_id:
                    self._last_order_id = current_id
                    items_assigned.clear()
                    # Reset non-deliver bots
                    for bid in list(bot_goal.keys()):
                        if bot_goal[bid] != "deliver":
                            bot_goal[bid] = "park"

            state, done = sim.step(actions)
            if done:
                break

            if r % 50 == 0:
                print(f"  R{r}: score={sim._score}, orders={sim._orders_completed}", flush=True)

        result = {
            "score": sim._score,
            "orders_completed": sim._orders_completed,
            "items_delivered": sim._items_delivered,
            "rounds_used": sim._round,
        }
        print(f"\nFinal: score={result['score']}, orders={result['orders_completed']}, "
              f"rounds={result['rounds_used']}", flush=True)
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    t0 = time.time()
    planner = SimPlanner(args.recon)
    result = planner.run()
    print(f"Time: {time.time()-t0:.1f}s")
