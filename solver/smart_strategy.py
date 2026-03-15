"""Smart strategy: round-by-round decisions using known order sequence.

Unlike BotAdapter (reactive), this knows ALL future orders.
Unlike planner_v2 (pre-computed), this decides EACH ROUND based on actual state.

Uses simple collision avoidance: lower ID moves first, others avoid.
Matches sim's collision model exactly because it reads sim state directly.
"""

from __future__ import annotations

import json
import sys
import os
import time
from collections import Counter, deque
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

Pos = tuple[int, int]

MOVES = {
    "move_up": (0, -1), "move_down": (0, 1),
    "move_left": (-1, 0), "move_right": (1, 0),
}


class SmartStrategy:
    """Knows full order sequence. Decides per-round. Simple PIBT."""

    def __init__(self, orders: list[dict], shelf_map: dict[str, list],
                 drop_off_zones: list, grid_walls: set[Pos], grid_w: int, grid_h: int,
                 shelves: set[Pos]):
        self.orders = orders
        self.shelf_map = shelf_map  # type -> [[x,y], ...]
        self.drop_off_zones = [tuple(z) for z in drop_off_zones]
        self.walls = grid_walls
        self.shelves = shelves
        self.obstacles = grid_walls | shelves
        self.w = grid_w
        self.h = grid_h

        # Bot state
        self.bot_targets: dict[int, Pos | None] = {}
        self.bot_goals: dict[int, str] = {}  # "pickup", "deliver", "idle"
        self.bot_item_type: dict[int, str | None] = {}
        self.claimed_types: Counter = Counter()
        self.active_order_id: str | None = None

    def __call__(self, state: dict) -> dict:
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

        # Detect order change
        active_id = active["id"] if active else None
        if active_id != self.active_order_id:
            self.active_order_id = active_id
            self.claimed_types.clear()
            # Reclaim items in bot inventories
            for bot in bots:
                for item in bot.get("inventory", []):
                    self.claimed_types[item] += 1

        # Items remaining for active order
        remaining = Counter()
        if active:
            remaining = Counter(active["items_required"])
            for d in active.get("items_delivered", []):
                if remaining[d] > 0:
                    remaining[d] -= 1

        # Preview remaining
        preview_remaining = Counter()
        if preview:
            preview_remaining = Counter(preview["items_required"])

        # Build item lookup
        items_by_type: dict[str, list] = {}
        for item in items:
            items_by_type.setdefault(item["type"], []).append(item)

        bot_positions = {b["id"]: tuple(b["position"]) for b in bots}
        occupied = set(bot_positions.values())

        # Assign goals to bots
        for bot in bots:
            bid = bot["id"]
            pos = tuple(bot["position"])
            inv = bot.get("inventory", [])

            if bid not in self.bot_goals:
                self.bot_goals[bid] = "idle"
                self.bot_targets[bid] = None
                self.bot_item_type[bid] = None

            # If at target and goal is pickup → do pickup
            if self.bot_goals[bid] == "pickup" and pos == self.bot_targets[bid]:
                pass  # handled in action generation

            # If at target and goal is deliver → do deliver
            if self.bot_goals[bid] == "deliver" and pos in self.drop_off_zones:
                pass  # handled in action generation

            # Assign idle bots
            if self.bot_goals[bid] == "idle" or self.bot_targets[bid] is None:
                # Has matching items for active? → deliver
                if inv and active:
                    matching = [i for i in inv if remaining.get(i, 0) > 0]
                    if matching:
                        dz = min(self.drop_off_zones, key=lambda z: abs(z[0]-pos[0])+abs(z[1]-pos[1]))
                        self.bot_goals[bid] = "deliver"
                        self.bot_targets[bid] = dz
                        continue

                # Find unclaimed item to pick
                assigned = False
                for order_items, is_preview in [(remaining, False), (preview_remaining, True)]:
                    if assigned:
                        break
                    for item_type, needed in order_items.items():
                        if needed <= self.claimed_types.get(item_type, 0):
                            continue
                        # Find nearest shelf with this item type
                        best_pos = None
                        best_d = 9999
                        for shelf_pos_list in [self.shelf_map.get(item_type, [])]:
                            for sp in shelf_pos_list:
                                sp = tuple(sp)
                                # Find adjacent walkable cell
                                for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                                    adj = (sp[0]+dx, sp[1]+dy)
                                    if self._walkable(adj):
                                        d = abs(adj[0]-pos[0]) + abs(adj[1]-pos[1])
                                        if d < best_d:
                                            best_d = d
                                            best_pos = adj

                        if best_pos:
                            self.bot_goals[bid] = "pickup"
                            self.bot_targets[bid] = best_pos
                            self.bot_item_type[bid] = item_type
                            self.claimed_types[item_type] += 1
                            assigned = True
                            break

        # Generate actions — process in ID order (matching sim's collision model)
        actions = []
        # Track where bots have moved TO (claimed cells this round)
        claimed: set[Pos] = set()

        for bot in sorted(bots, key=lambda b: b["id"]):
            bid = bot["id"]
            pos = tuple(bot["position"])
            inv = bot.get("inventory", [])
            goal = self.bot_goals.get(bid, "idle")
            target = self.bot_targets.get(bid)

            # Pickup action
            if goal == "pickup" and target == pos:
                item_type = self.bot_item_type.get(bid)
                item_id = None
                if item_type:
                    for item in items_by_type.get(item_type, []):
                        ipos = tuple(item["position"])
                        if abs(ipos[0]-pos[0]) + abs(ipos[1]-pos[1]) == 1:
                            item_id = item["id"]
                            break
                if item_id:
                    actions.append({"bot": bid, "action": "pick_up", "item_id": item_id})
                    self.bot_goals[bid] = "idle"
                    self.bot_targets[bid] = None
                    self.bot_item_type[bid] = None
                    claimed.add(pos)
                    continue
                else:
                    self.bot_goals[bid] = "idle"
                    self.bot_targets[bid] = None

            # Dropoff action
            if goal == "deliver" and pos in self.drop_off_zones:
                actions.append({"bot": bid, "action": "drop_off"})
                self.bot_goals[bid] = "idle"
                self.bot_targets[bid] = None
                claimed.add(pos)
                continue

            # Movement toward target
            if target and target != pos:
                best_move = self._best_step(pos, target, occupied, claimed)
                if best_move:
                    dx = best_move[0] - pos[0]
                    dy = best_move[1] - pos[1]
                    act = {(1,0): "move_right", (-1,0): "move_left",
                           (0,1): "move_down", (0,-1): "move_up"}.get((dx,dy), "wait")
                    actions.append({"bot": bid, "action": act})
                    # Update occupied: this bot LEAVES old pos, ENTERS new pos
                    occupied.discard(pos)
                    occupied.add(best_move)
                    claimed.add(best_move)
                    continue

            actions.append({"bot": bid, "action": "wait"})

        return {"actions": actions}

    def _best_step(self, pos: Pos, target: Pos, occupied: set[Pos], moved_to: set[Pos]) -> Pos | None:
        """Find best single step toward target avoiding collisions.

        Matches sim collision model: check if target cell is occupied by
        any other bot's CURRENT position. Sim processes in ID order and
        updates positions, so lower-ID bots that already moved free their cells.
        """
        candidates = []
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            np = (pos[0]+dx, pos[1]+dy)
            if not self._walkable(np):
                continue
            if np in moved_to:  # cell already claimed by higher-priority bot this round
                continue
            d = abs(target[0]-np[0]) + abs(target[1]-np[1])
            candidates.append((d, np))

        candidates.sort()
        return candidates[0][1] if candidates else None

    def _walkable(self, pos: Pos) -> bool:
        return (0 <= pos[0] < self.w and 0 <= pos[1] < self.h
                and pos not in self.obstacles)


def run(recon_path: str) -> int:
    """Run smart strategy through sim."""
    import logging; logging.basicConfig(level=logging.CRITICAL)
    from Simulering.offline.simulator import Simulator

    with open(recon_path) as f:
        recon = json.load(f)

    walls = set(tuple(w) for w in recon["walls"])
    shelf_map = recon["shelf_map"]
    all_shelves = set()
    for positions in shelf_map.values():
        for p in positions:
            all_shelves.add(tuple(p))
    grid_size = recon["grid_size"]
    drop_off_zones = recon["drop_off_zones"]
    orders = recon["order_sequence"]

    strategy = SmartStrategy(
        orders=orders,
        shelf_map=shelf_map,
        drop_off_zones=drop_off_zones,
        grid_walls=walls,
        grid_w=grid_size[0],
        grid_h=grid_size[1],
        shelves=all_shelves,
    )

    sim = Simulator.from_recon_file(recon_path)
    result = sim.run(strategy)
    return result["score"], result["orders_completed"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    t0 = time.time()
    score, orders = run(args.recon)
    print(f"Score: {score}, Orders: {orders}, Time: {time.time()-t0:.1f}s")
