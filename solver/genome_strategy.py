"""Genome-driven strategy: executes a genome using PIBT navigation.

Reads decisions from genome (which bot picks which item from which shelf).
Uses real PIBT for collision-free movement. Runs in validated sim.

Key fixes from smart_strategy v1:
- One-way aisles properly initialized (136 rules)
- Idle bots park FAR from corridors (y=1/y=9, not y=15/y=16)
- ESCAPE priority for bots blocking drop-off
- Proper pickup adjacency check
"""

from __future__ import annotations

import json
import sys
import os
import time
import logging
from collections import Counter
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.genome import Genome, ItemAssignment, generate_genome

Pos = tuple[int, int]

# Parking spots: far from drop-off corridors, spread across map
PARKING_SPOTS = [
    (4, 1), (8, 1), (12, 1), (16, 1), (20, 1), (24, 1),
    (4, 9), (8, 9), (12, 9), (16, 9), (20, 9), (24, 9),
    (4, 5), (8, 5), (12, 5), (16, 5), (20, 5), (24, 5),
    (4, 3), (8, 3),
]


class GenomeStrategy:
    """Executes a genome in the simulator using PIBT for movement."""

    def __init__(self, genome: Genome, shelf_map: dict[str, list],
                 drop_off_zones: list[Pos], grid_walls: set[Pos],
                 grid_w: int, grid_h: int, shelves: set[Pos]):
        self.genome = genome
        self.shelf_map = shelf_map
        self.drop_off_zones = [tuple(z) for z in drop_off_zones]
        self.walls = grid_walls
        self.shelves = shelves
        self.obstacles = grid_walls | shelves
        self.w = grid_w
        self.h = grid_h

        # Initialize PIBT + PathEngine ONCE
        from bot.models import Grid as BotGrid
        from bot.engine.pathfinding import PathEngine
        from bot.engine.pibt import PIBTResolver

        self._grid = BotGrid(grid_w, grid_h, frozenset(self.obstacles))
        self._pe = PathEngine()
        self._pe.set_grid(self._grid, drop_off=self.drop_off_zones[0])
        # Force one-way detection
        self._pe._one_way = self._pe._detect_one_way_aisles(self._grid, self.drop_off_zones[0])
        self._pe._one_way_enabled = True

        # Guidance graph for congestion-aware routing
        from bot.engine.guidance import GuidanceGraph
        self._guidance = GuidanceGraph(
            self._grid,
            one_way=self._pe._one_way,
            alpha=2.0, beta=3.0, decay=0.7, update_interval=3,
        )

        self._pibt = PIBTResolver(
            self._grid, self._pe.distance, self._pe.corridors,
            one_way=self._pe._one_way,
            guidance_fn=self._guidance.guided_distance,
        )

        # Bot state
        self.bot_goal: dict[int, str] = {}  # "pickup", "deliver", "park"
        self.bot_target: dict[int, Pos] = {}
        self.bot_item_type: dict[int, str | None] = {}
        self.active_order_id: str | None = None
        self.genome_order_idx: int = 0
        self.genome_item_idx: int = 0
        self.assigned_bots: set[int] = set()  # bots with active assignment

    def __call__(self, state: dict) -> dict:
        bots = state.get("bots", [])
        items = state.get("items", [])
        orders = state.get("orders", [])
        round_num = state.get("round", 0)

        # Parse orders
        active = None
        for o in orders:
            if o.get("status") == "active" and not o.get("complete"):
                active = o
                break

        active_id = active["id"] if active else None

        # Order changed — sync genome index with active order
        if active_id != self.active_order_id:
            if self.active_order_id is not None:
                self.genome_order_idx += 1
            self.active_order_id = active_id
            self.genome_item_idx = 0
            self.assigned_bots.clear()
            # Reset ALL bot goals — force re-assignment for new order
            for bid in list(self.bot_goal.keys()):
                if self.bot_goal[bid] != "deliver":  # let deliverers finish
                    self.bot_goal[bid] = "park"
                    self.bot_target[bid] = PARKING_SPOTS[bid % len(PARKING_SPOTS)]
                    self.bot_item_type[bid] = None

        # Items remaining for active order
        remaining = Counter()
        if active:
            remaining = Counter(active["items_required"])
            for d in active.get("items_delivered", []):
                if remaining[d] > 0:
                    remaining[d] -= 1

        # Build item lookup
        items_by_pos: dict[Pos, list] = {}
        for item in items:
            pos = tuple(item["position"])
            items_by_pos.setdefault(pos, []).append(item)

        # Update guidance graph with traffic data
        bot_positions_map = {b["id"]: tuple(b["position"]) for b in bots}
        self._guidance.on_round(bot_positions_map, round_num)

        # Get current genome order
        genome_order = None
        if self.genome_order_idx < len(self.genome.orders):
            genome_order = self.genome.orders[self.genome_order_idx]

        # === ASSIGN GOALS ===
        for bot in bots:
            bid = bot["id"]
            pos = tuple(bot["position"])
            inv = bot.get("inventory", [])

            if bid not in self.bot_goal:
                self.bot_goal[bid] = "park"
                self.bot_target[bid] = PARKING_SPOTS[bid % len(PARKING_SPOTS)]
                self.bot_item_type[bid] = None

            # Already has goal — check completion
            goal = self.bot_goal[bid]
            target = self.bot_target[bid]

            # Pickup completed (inventory grew)
            if goal == "pickup" and inv and self.bot_item_type[bid]:
                if self.bot_item_type[bid] in inv:
                    # Picked up — now deliver
                    dz = min(self.drop_off_zones, key=lambda z: self._pe.distance(pos, z) or 9999)
                    self.bot_goal[bid] = "deliver"
                    self.bot_target[bid] = dz
                    self.bot_item_type[bid] = None
                    continue

            # Delivery completed (at drop-off with matching items → sim handles it)
            if goal == "deliver" and pos in self.drop_off_zones:
                # Will send drop_off action below
                pass

            # Idle/parked bot — assign next item from genome
            if goal == "park" and bid not in self.assigned_bots:
                if genome_order and self.genome_item_idx < len(genome_order.assignments):
                    assignment = genome_order.assignments[self.genome_item_idx]
                    item_type = assignment.item_type
                    shelf_idx = assignment.shelf_index
                    shelves_list = self.shelf_map.get(item_type, [])
                    if shelves_list:
                        shelf_pos = tuple(shelves_list[shelf_idx % len(shelves_list)])
                        pickup_pos = self._find_pickup_pos(shelf_pos)
                        if pickup_pos:
                            self.bot_goal[bid] = "pickup"
                            self.bot_target[bid] = pickup_pos
                            self.bot_item_type[bid] = item_type
                            self.assigned_bots.add(bid)
                            self.genome_item_idx += 1

        # === GENERATE ACTIONS ===
        # Separate immediate actions vs movement
        immediate: dict[int, dict] = {}
        movement_bots: dict[int, Pos] = {}
        movement_targets: dict[int, Pos] = {}

        for bot in bots:
            bid = bot["id"]
            pos = tuple(bot["position"])
            inv = bot.get("inventory", [])
            goal = self.bot_goal.get(bid, "park")
            target = self.bot_target.get(bid, pos)

            # Pickup: at target + adjacent to shelf with matching item
            if goal == "pickup" and pos == target:
                item_type = self.bot_item_type.get(bid)
                item_id = self._find_item_id(pos, item_type, items_by_pos)
                if item_id:
                    immediate[bid] = {"bot": bid, "action": "pick_up", "item_id": item_id}
                    continue
                # Item not found — clear and park
                self.bot_goal[bid] = "park"
                self.bot_target[bid] = PARKING_SPOTS[bid % len(PARKING_SPOTS)]

            # Dropoff: at drop-off zone with matching items
            if goal == "deliver" and pos in self.drop_off_zones:
                if inv and remaining:
                    matching = [i for i in inv if remaining.get(i, 0) > 0]
                    if matching:
                        immediate[bid] = {"bot": bid, "action": "drop_off"}
                        self.bot_goal[bid] = "park"
                        self.bot_target[bid] = PARKING_SPOTS[bid % len(PARKING_SPOTS)]
                        self.assigned_bots.discard(bid)
                        continue
                # At drop-off but NO matching items → ESCAPE!
                # Must leave immediately so deliverers can use the zone
                movement_bots[bid] = pos
                escape = PARKING_SPOTS[bid % len(PARKING_SPOTS)]
                movement_targets[bid] = escape
                continue

            # Movement
            movement_bots[bid] = pos
            movement_targets[bid] = target if target else pos

        # TRICK 1: Add stationary bots (pickup/dropoff) to PIBT as obstacles
        # Without this, PIBT sends bots INTO occupied pickup/dropoff positions
        for bid, act in immediate.items():
            bot = [b for b in bots if b["id"] == bid][0]
            pos = tuple(bot["position"])
            if bid not in movement_bots:
                movement_bots[bid] = pos
                movement_targets[bid] = pos  # stay in place

        # TRICK 2: PIBT urgency with ESCAPE priority
        urgency: dict[int, int] = {}
        idle_bots: set[int] = set()
        for bid in movement_bots:
            pos = movement_bots[bid]
            g = self.bot_goal.get(bid, "park")

            # ESCAPE: bot ON drop-off that is NOT delivering → absolute highest priority
            if pos in self.drop_off_zones and bid not in immediate:
                urgency[bid] = -1  # ESCAPE — push everyone
            elif g == "deliver":
                urgency[bid] = 0
            elif g == "pickup":
                urgency[bid] = 1
            else:
                urgency[bid] = 3
                idle_bots.add(bid)

        # Resolve movement with PIBT
        resolved = {}
        if movement_bots:
            resolved = self._pibt.resolve(
                movement_bots, movement_targets,
                tiebreak_offset=round_num,
                urgency=urgency,
                idle_bots=idle_bots,
            )

        # Build actions
        actions = []
        for bot in bots:
            bid = bot["id"]
            pos = tuple(bot["position"])

            if bid in immediate:
                actions.append(immediate[bid])
            elif bid in resolved:
                new_pos = resolved[bid]
                if new_pos == pos:
                    actions.append({"bot": bid, "action": "wait"})
                else:
                    dx = new_pos[0] - pos[0]
                    dy = new_pos[1] - pos[1]
                    act = {(1,0): "move_right", (-1,0): "move_left",
                           (0,1): "move_down", (0,-1): "move_up"}.get((dx,dy), "wait")
                    actions.append({"bot": bid, "action": act})
            else:
                actions.append({"bot": bid, "action": "wait"})

        return {"actions": actions}

    def _find_pickup_pos(self, shelf_pos: Pos) -> Pos | None:
        """Find walkable cell adjacent to shelf."""
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            adj = (shelf_pos[0] + dx, shelf_pos[1] + dy)
            if 0 <= adj[0] < self.w and 0 <= adj[1] < self.h and adj not in self.obstacles:
                return adj
        return None

    def _find_item_id(self, pos: Pos, item_type: str | None, items_by_pos: dict) -> str | None:
        """Find item_id adjacent to pos matching type."""
        if not item_type:
            return None
        for dx, dy in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            adj = (pos[0] + dx, pos[1] + dy)
            for item in items_by_pos.get(adj, []):
                if item["type"] == item_type:
                    return item["id"]
        return None


def run_genome(recon_path: str, genome: Genome) -> tuple[int, int]:
    """Run a genome through the validated sim. Returns (score, orders)."""
    with open(recon_path) as f:
        recon = json.load(f)

    walls = set(tuple(w) for w in recon["walls"])
    shelves = set()
    for ps in recon["shelf_map"].values():
        for p in ps:
            shelves.add(tuple(p))

    strategy = GenomeStrategy(
        genome=genome,
        shelf_map=recon["shelf_map"],
        drop_off_zones=recon["drop_off_zones"],
        grid_walls=walls,
        grid_w=recon["grid_size"][0],
        grid_h=recon["grid_size"][1],
        shelves=shelves,
    )

    from Simulering.offline.simulator import Simulator
    sim = Simulator.from_recon_file(recon_path)
    result = sim.run(strategy)
    return result["score"], result["orders_completed"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", required=True)
    args = parser.parse_args()

    with open(args.recon) as f:
        recon = json.load(f)

    genome = generate_genome(
        recon["order_sequence"],
        recon["shelf_map"],
        n_bots=recon.get("bot_count", 20),
    )

    t0 = time.time()
    score, orders = run_genome(args.recon, genome)
    print(f"Score: {score}, Orders: {orders}, Time: {time.time()-t0:.1f}s")
