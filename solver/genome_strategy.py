"""Genome-driven strategy: executes a genome using PIBT navigation.

Reads decisions from genome (which bot picks which item from which shelf).
Uses real PIBT for collision-free movement. Runs in validated sim.

V2: Ported ALL execution tricks from BotAdapter/V2TaskPlanner:
- Spawn scatter (BFS dispersal from spawn stacking)
- Delivery scheduling (match count → deliver)
- Queue management (idle bots wait on queue row)
- Zone affinity (bots prefer their zone's shelves)
- Sprint + pipeline teams (active order sprint, preview pipeline)
- Multi-bot simultaneous assignment (not sequential)
- Inventory-aware decisions (full inv → deliver or queue)
- Demand scoring (future order lookahead)
"""

from __future__ import annotations

import json
import math
import sys
import os
import time
import logging
from collections import Counter, deque
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

from solver.genome import Genome, ItemAssignment, generate_genome

Pos = tuple[int, int]

# Zone definitions for nightmare map (3 zones matching 3 drop-off positions)
NIGHTMARE_ZONES = [
    (1, 9),    # Left zone: shelves x=3,5,7,9
    (10, 18),  # Center zone: shelves x=11,13,15,17
    (19, 28),  # Right zone: shelves x=19,21,23,25
]


def _bot_zone(bot_id: int, n_bots: int) -> int:
    """Assign bot to zone by ID. Spread evenly across 3 zones."""
    zone_size = n_bots // 3
    remainder = n_bots % 3
    if bot_id < zone_size + (1 if remainder >= 1 else 0):
        return 0
    elif bot_id < 2 * zone_size + (1 if remainder >= 1 else 0) + (1 if remainder >= 2 else 0):
        return 1
    else:
        return 2


class GenomeStrategy:
    """Executes a genome in the simulator using PIBT for movement.

    V2: Full execution layer ported from BotAdapter/V2TaskPlanner.
    Genome controls WHICH item from WHICH shelf for each bot.
    Execution layer handles HOW: scatter, delivery, queuing, zones.
    """

    def __init__(self, genome: Genome, shelf_map: dict[str, list],
                 drop_off_zones: list[Pos], grid_walls: set[Pos],
                 grid_w: int, grid_h: int, shelves: set[Pos],
                 order_sequence: list[dict] | None = None):
        self.genome = genome
        self.shelf_map = shelf_map
        self.drop_off_zones = [tuple(z) for z in drop_off_zones]
        self.walls = grid_walls
        self.shelves = shelves
        self.obstacles = grid_walls | shelves
        self.w = grid_w
        self.h = grid_h
        self.order_sequence = order_sequence or []

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
            alpha=genome.guidance_alpha,
            beta=genome.guidance_beta,
            decay=genome.guidance_decay,
            update_interval=3,
        )

        # Store genome routing params
        self._dropoff_load_factor = genome.dropoff_load_factor
        self._max_deliverers_override = genome.max_deliverers
        self._sprint_team_size = genome.sprint_team_size
        self._max_per_zone = genome.max_deliverers_per_zone
        self._preposition_rounds = genome.preposition_rounds

        self._pibt = PIBTResolver(
            self._grid, self._pe.distance, self._pe.corridors,
            one_way=self._pe._one_way,
            guidance_fn=self._guidance.guided_distance,
        )

        # Cross-corridors for queue management
        self._cross_corridors = self._find_cross_corridors()
        self._queue_y = self._cross_corridors[-2] if len(self._cross_corridors) >= 2 else self._cross_corridors[0]
        self._drive_y = self._cross_corridors[-1] if self._cross_corridors else grid_h - 1

        # Pre-compute queue positions (on queue row, NOT on one-way aisles)
        one_way = self._pe._one_way
        self._queue_positions: list[Pos] = []
        for x in range(3, grid_w - 1):
            pos = (x, self._queue_y)
            if pos not in self.obstacles and pos not in one_way:
                self._queue_positions.append(pos)

        # Pre-compute safe idle spots: prefer non-one-way, then non-corridor
        # If no non-one-way spots exist (nightmare), use corridor spots far from drop-off
        self._idle_spots: list[Pos] = []
        corridor_ys = set(self._cross_corridors)
        drop_off_set = set(self.drop_off_zones)
        for y in range(grid_h):
            for x in range(grid_w):
                pos = (x, y)
                if pos in self.obstacles or pos in drop_off_set:
                    continue
                if pos not in one_way:
                    self._idle_spots.append(pos)

        # If no safe spots, use corridor positions far from drop-off
        if not self._idle_spots:
            for y in sorted(self._cross_corridors):
                for x in range(grid_w):
                    pos = (x, y)
                    if pos in self.obstacles or pos in drop_off_set:
                        continue
                    # Prefer top corridors (y=1, y=9) over bottom (y=15, y=16)
                    self._idle_spots.append(pos)

        # Sort: top of map first (far from drop-off at y=16)
        self._idle_spots.sort(key=lambda p: (p[1], p[0]))

        # Bot state
        self.bot_goal: dict[int, str] = {}  # "pickup", "deliver", "park"
        self.bot_target: dict[int, Pos] = {}
        self.bot_item_type: dict[int, str | None] = {}
        self.active_order_id: str | None = None
        self.genome_order_idx: int = 0
        self.genome_item_idx: int = 0
        self.assigned_bots: set[int] = set()
        self._prev_inventory: dict[int, list[str]] = {}
        self._prev_pos: dict[int, Pos] = {}
        self._stuck_count: dict[int, int] = {}

        # No scatter — bots go directly to work targets from spawn.
        # PIBT handles spawn stacking naturally (low ID moves first).

        # Item tracking: claimed items (no two bots target same item)
        self._claimed_items: set[str] = set()

        # Demand scoring from order sequence
        self._current_order_index = 0

        # Build item-type → shelf positions lookup
        self._shelf_positions: dict[str, list[Pos]] = {}
        for item_type, positions in shelf_map.items():
            self._shelf_positions[item_type] = [tuple(p) for p in positions]

    def _find_cross_corridors(self) -> list[int]:
        """Find cross-corridor y-values (rows where >=60% of cells are walkable)."""
        cross_ys = []
        for y in range(self.h):
            walkable = sum(1 for x in range(self.w) if (x, y) not in self.obstacles)
            if walkable >= self.w * 0.6:
                cross_ys.append(y)
        return sorted(cross_ys)

    def _compute_scatter_targets(self, spawn: Pos, bot_ids: list[int]) -> dict[int, Pos]:
        """BFS scatter targets from spawn. Lower IDs get closer targets."""
        visited = set()
        queue = deque([(spawn, 0)])
        visited.add(spawn)
        positions: list[Pos] = []

        while queue and len(positions) < len(bot_ids):
            pos, dist = queue.popleft()
            if dist > 0:
                positions.append(pos)
            for neighbor in self._pe._directed_neighbors(pos):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        sorted_ids = sorted(bot_ids)
        targets: dict[int, Pos] = {}
        for i, bot_id in enumerate(sorted_ids):
            if i < len(positions):
                targets[bot_id] = positions[i]
            else:
                targets[bot_id] = spawn
        return targets

    def _build_demand_score(self, active_order: dict | None, preview_items: list[str]) -> Counter:
        """Count how many upcoming orders need each item type."""
        demand: Counter = Counter()
        if active_order:
            for t in set(active_order.get("items_required", [])):
                demand[t] += 3
        if preview_items:
            for t in set(preview_items):
                demand[t] += 2
        if self.order_sequence:
            start = self._current_order_index + 2
            for i in range(start, min(start + 8, len(self.order_sequence))):
                order = self.order_sequence[i]
                for t in set(order.get("items_required", [])):
                    demand[t] += 1
        return demand

    def _find_pickup_pos(self, shelf_pos: Pos) -> Pos | None:
        """Find walkable cell adjacent to shelf."""
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            adj = (shelf_pos[0] + dx, shelf_pos[1] + dy)
            if 0 <= adj[0] < self.w and 0 <= adj[1] < self.h and adj not in self.obstacles:
                return adj
        return None

    def _best_pickup_pos(self, bot_pos: Pos, shelf_pos: Pos) -> Pos | None:
        """Find closest walkable cell adjacent to shelf from bot's perspective."""
        best = None
        best_d = 9999
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            adj = (shelf_pos[0] + dx, shelf_pos[1] + dy)
            if 0 <= adj[0] < self.w and 0 <= adj[1] < self.h and adj not in self.obstacles:
                d = self._pe.distance(bot_pos, adj) or 9999
                if d < best_d:
                    best_d = d
                    best = adj
        return best

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

    def _best_drop_off(self, pos: Pos, bot_positions: dict[int, Pos]) -> Pos:
        """Find best drop-off zone — nearest, but load-balanced across zones.

        Counts how many OTHER bots are already heading to each zone to
        spread deliveries across all 3 drop-offs.
        """
        # Count current deliverers per zone
        zone_load: Counter = Counter()
        for bid, goal in self.bot_goal.items():
            if goal == "deliver":
                target = self.bot_target.get(bid)
                if target in self.drop_off_zones:
                    zone_load[target] += 1

        best_dz = self.drop_off_zones[0]
        best_score = 99999
        for dz in self.drop_off_zones:
            d = self._pe.distance(pos, dz) or 9999
            load = zone_load.get(dz, 0)
            score = d + self._dropoff_load_factor * load  # Penalize overloaded zones
            if score < best_score:
                best_score = score
                best_dz = dz

        return best_dz

    def _queue_pos(self, bot_pos: Pos, occupied: set[Pos]) -> Pos:
        """Find nearest unoccupied queue position."""
        best = bot_pos
        best_d = 9999
        for pos in self._queue_positions:
            if pos in occupied:
                continue
            d = self._pe.distance(bot_pos, pos) or 9999
            if d < best_d:
                best_d = d
                best = pos
        return best

    def _safe_idle_pos(self, bot_pos: Pos) -> Pos:
        """Return a safe idle position — never on one-way aisle or corridor."""
        one_way = self._pe._one_way
        corridor_ys = set(self._cross_corridors)
        # If current position is safe, stay
        if bot_pos not in one_way and bot_pos[1] not in corridor_ys:
            return bot_pos
        # Find nearest safe idle spot
        best_pos = bot_pos
        best_d = 9999
        for pos in self._idle_spots:
            d = self._pe.distance(bot_pos, pos) or 9999
            if d < best_d:
                best_d = d
                best_pos = pos
                if d <= 2:
                    break  # Good enough
        return best_pos

    def _sync_order_index(self, active_order: dict | None):
        """Sync current_order_index with active order."""
        if not self.order_sequence or not active_order:
            return
        active_id = active_order.get("id", "")
        for i, order in enumerate(self.order_sequence):
            if order.get("id") == active_id:
                self._current_order_index = i
                return
        # Fallback: match by items
        active_items = sorted(active_order.get("items_required", []))
        for i, order in enumerate(self.order_sequence):
            if sorted(order.get("items_required", [])) == active_items:
                self._current_order_index = i
                return

    def __call__(self, state: dict) -> dict:
        bots = state.get("bots", [])
        items = state.get("items", [])
        orders = state.get("orders", [])
        round_num = state.get("round", 0)
        n_bots = len(bots)

        # Parse orders
        active = None
        preview = None
        for o in orders:
            if o.get("status") == "active" and not o.get("complete"):
                active = o
            elif o.get("status") == "preview":
                preview = o

        active_id = active["id"] if active else None

        # Items remaining for active order
        remaining = Counter()
        remaining_types: set[str] = set()
        remaining_list: list[str] = []
        if active:
            remaining = Counter(active["items_required"])
            for d in active.get("items_delivered", []):
                if remaining[d] > 0:
                    remaining[d] -= 1
            remaining_types = set(t for t, c in remaining.items() if c > 0)
            remaining_list = list(remaining.elements())

        # Preview items
        preview_types: set[str] = set()
        preview_list: list[str] = []
        if preview:
            preview_types = set(preview.get("items_required", []))
            preview_list = list(preview.get("items_required", []))

        # Sync order index
        self._sync_order_index(active)

        # Order changed — advance genome index and RESET all bot assignments
        if active_id != self.active_order_id:
            if self.active_order_id is not None:
                self.genome_order_idx += 1
            self.active_order_id = active_id
            self.genome_item_idx = 0
            self.assigned_bots.clear()
            self._claimed_items.clear()
            # Reset ALL non-deliver bots to force re-assignment for new order
            for bid in list(self.bot_goal.keys()):
                if self.bot_goal[bid] != "deliver":
                    self.bot_goal[bid] = "park"
                    self.bot_item_type[bid] = None

        # Build item lookup
        items_by_pos: dict[Pos, list] = {}
        items_by_id: dict[str, dict] = {}
        for item in items:
            pos = tuple(item["position"])
            items_by_pos.setdefault(pos, []).append(item)
            items_by_id[item["id"]] = item

        # Build position maps
        bot_positions = {b["id"]: tuple(b["position"]) for b in bots}

        # Update guidance graph
        self._guidance.on_round(bot_positions, round_num)

        # Build demand score
        demand = self._build_demand_score(active, preview_list)

        # === STUCK DETECTION ===
        for bot in bots:
            bid = bot["id"]
            pos = tuple(bot["position"])
            if pos == self._prev_pos.get(bid):
                self._stuck_count[bid] = self._stuck_count.get(bid, 0) + 1
            else:
                self._stuck_count[bid] = 0
            self._prev_pos[bid] = pos

        # Detect inventory changes (pickup completed)
        for bot in bots:
            bid = bot["id"]
            inv = bot.get("inventory", [])
            prev_inv = self._prev_inventory.get(bid, [])
            if len(inv) > len(prev_inv) and self.bot_goal.get(bid) == "pickup":
                # Pickup completed → deliver
                pos = tuple(bot["position"])
                dz = self._best_drop_off(pos, bot_positions)
                self.bot_goal[bid] = "deliver"
                self.bot_target[bid] = dz
                self.bot_item_type[bid] = None

        # No scatter phase — bots go directly to unique work targets.
        # PIBT processes by ID order (low ID first), so they naturally unstick.

        # Occupied queue tracking
        occupied_queue: set[Pos] = set()
        for bot in bots:
            if tuple(bot["position"])[1] == self._queue_y:
                occupied_queue.add(tuple(bot["position"]))

        # Count matching items per bot for active order
        def match_count(bot_inv: list[str]) -> int:
            if not remaining_list:
                return 0
            temp = list(remaining_list)
            count = 0
            for inv_item in bot_inv:
                if inv_item in temp:
                    count += 1
                    temp.remove(inv_item)
            return count

        # Track filling types (what bots are already picking)
        filling_types: list[str] = []
        for bid, goal in self.bot_goal.items():
            if goal == "pickup" and self.bot_item_type.get(bid):
                filling_types.append(self.bot_item_type[bid])

        # All inventory types
        all_inventory: list[str] = []
        for bot in bots:
            all_inventory.extend(bot.get("inventory", []))

        # === INITIALIZE UNTRACKED BOTS ===
        for bot in bots:
            bid = bot["id"]
            if bid not in self.bot_goal:
                self.bot_goal[bid] = "park"
                self.bot_target[bid] = self._safe_idle_pos(tuple(bot["position"]))
                self.bot_item_type[bid] = None

        # === PHASE 1: DELIVERY SCHEDULING ===
        # Bots with matching inventory → deliver (load-balanced across zones)
        if remaining_types:
            candidates = []
            for bot in bots:
                bid = bot["id"]
                inv = bot.get("inventory", [])
                mc = match_count(inv)
                if mc <= 0:
                    continue
                pos = tuple(bot["position"])
                d_to_drop = self._pe.distance(pos, self._best_drop_off(pos, bot_positions)) or 9999
                candidates.append((-mc, d_to_drop, bid, inv))

            candidates.sort()

            n_zones = len(self.drop_off_zones)
            if self._max_deliverers_override > 0:
                max_deliverers = self._max_deliverers_override
            else:
                high_match = sum(1 for _, _, _, inv in candidates if match_count(inv) >= 2)
                max_deliverers = n_zones * (3 if high_match >= 4 else 1)

            delivering = 0
            for _, _, bid, inv in candidates:
                if delivering >= max_deliverers:
                    break
                if self.bot_goal.get(bid) == "deliver":
                    delivering += 1
                    continue
                pos = bot_positions[bid]
                self.bot_goal[bid] = "deliver"
                self.bot_target[bid] = self._best_drop_off(pos, bot_positions)
                self.bot_item_type[bid] = None
                self.assigned_bots.add(bid)
                delivering += 1

        # === PHASE 2: Validate existing tasks ===
        item_ids_set = set(items_by_id.keys())
        for bot in bots:
            bid = bot["id"]
            goal = self.bot_goal.get(bid, "park")
            inv = bot.get("inventory", [])
            pos = tuple(bot["position"])

            # Deliverers with no active matches → clear
            if goal == "deliver":
                if not any(t in remaining_types for t in inv):
                    if not inv:
                        self.bot_goal[bid] = "park"
                        self.bot_target[bid] = self._safe_idle_pos(pos)
                    else:
                        self.bot_goal[bid] = "park"
                        self.bot_target[bid] = self._safe_idle_pos(pos)
                    self.assigned_bots.discard(bid)

            # Stuck pickup bots → re-route to queue (deliver bots keep their drop-off target)
            if self._stuck_count.get(bid, 0) > 15 and goal == "pickup":
                self.bot_goal[bid] = "park"
                self.bot_target[bid] = self._safe_idle_pos(pos)
                self.bot_item_type[bid] = None
                self.assigned_bots.discard(bid)
                self._stuck_count[bid] = 0

        # === PHASE 3: RE-EVALUATE ALL BOTS ===
        genome_order = None
        if self.genome_order_idx < len(self.genome.orders):
            genome_order = self.genome.orders[self.genome_order_idx]

        # --- Validate and reassign all bots ---
        for bot in bots:
            bid = bot["id"]
            pos = tuple(bot["position"])
            inv = bot.get("inventory", [])
            goal = self.bot_goal.get(bid, "park")

            # --- DELIVER bots: validate ---
            if goal == "deliver":
                active_mc = match_count(inv)
                if active_mc > 0:
                    continue  # Valid delivery
                if pos in self.drop_off_zones:
                    continue  # Will escape in action generation
                preview_mc = sum(1 for t in inv if t in preview_types) if preview_types else 0
                if preview_mc > 0 and len(inv) >= 2:
                    continue  # Pre-positioned for auto-delivery
                self.bot_goal[bid] = "park"
                self.bot_item_type[bid] = None
                self.assigned_bots.discard(bid)

            # --- PICKUP bots: validate ---
            if goal == "pickup":
                item_type = self.bot_item_type.get(bid)
                if item_type and demand.get(item_type, 0) > 0:
                    continue  # Still useful
                self.bot_goal[bid] = "park"
                self.bot_item_type[bid] = None
                self.assigned_bots.discard(bid)

            # --- PARK bots: assign new work ---
            if self.bot_goal.get(bid) != "park":
                continue

            # Full inventory → deliver or park
            if len(inv) >= 3:
                active_mc = match_count(inv)
                preview_mc = sum(1 for t in inv if t in preview_types) if preview_types else 0
                if active_mc > 0 or preview_mc > 0:
                    dz = self._best_drop_off(pos, bot_positions)
                    self.bot_goal[bid] = "deliver"
                    self.bot_target[bid] = dz
                else:
                    self.bot_target[bid] = self._safe_idle_pos(pos)
                continue

            # --- Assignment ---
            assigned = False

            # Active order items first, then preview
            if remaining_types:
                target_types = remaining_types
                target_list = remaining_list
            elif preview_types:
                target_types = preview_types
                target_list = preview_list
            else:
                target_types = set()
                target_list = []

            # Early game: combined
            if round_num <= 20 and preview_types and remaining_types:
                target_types = remaining_types | preview_types
                target_list = remaining_list + preview_list

            # Try genome assignment first
            if not assigned and genome_order and self.genome_item_idx < len(genome_order.assignments):
                assignment = genome_order.assignments[self.genome_item_idx]
                item_type = assignment.item_type
                shelf_idx = assignment.shelf_index

                if demand.get(item_type, 0) > 0:
                    shelves_list = self._shelf_positions.get(item_type, [])
                    if shelves_list:
                        shelf_pos = shelves_list[shelf_idx % len(shelves_list)]
                        pickup_pos = self._best_pickup_pos(pos, shelf_pos)
                        if pickup_pos:
                            self.bot_goal[bid] = "pickup"
                            self.bot_target[bid] = pickup_pos
                            self.bot_item_type[bid] = item_type
                            self.assigned_bots.add(bid)
                            self.genome_item_idx += 1
                            filling_types.append(item_type)
                            assigned = True

                if not assigned:
                    self.genome_item_idx += 1

            # Fallback: targeted pickup
            if not assigned and target_types:
                assigned = self._assign_targeted_pickup(
                    bid, pos, inv, items, target_types, target_list,
                    filling_types, demand, n_bots
                )

            # Fallback: fill with demand items
            if not assigned:
                assigned = self._assign_fill_pickup(
                    bid, pos, inv, items, all_inventory, filling_types,
                    remaining_types, demand, n_bots
                )

            # Nothing to do → park safely
            if not assigned:
                self.bot_target[bid] = self._safe_idle_pos(pos)

        # === PHASE 4: Drive lane cleanup ===
        for bot in bots:
            bid = bot["id"]
            pos = tuple(bot["position"])
            if pos[1] != self._drive_y:
                continue
            if self.bot_goal.get(bid) == "park" and self.bot_target.get(bid) == pos:
                self.bot_target[bid] = self._safe_idle_pos(pos)

        # === GENERATE ACTIONS ===
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
                # Item not found — clear and queue
                self.bot_goal[bid] = "park"
                self.bot_target[bid] = self._safe_idle_pos(pos)

            # Dropoff: at drop-off zone with matching items
            if goal == "deliver" and pos in self.drop_off_zones:
                if inv and remaining:
                    matching = [i for i in inv if remaining.get(i, 0) > 0]
                    if matching:
                        immediate[bid] = {"bot": bid, "action": "drop_off"}
                        self.bot_goal[bid] = "park"
                        self.bot_target[bid] = self._safe_idle_pos(pos)
                        self.assigned_bots.discard(bid)
                        continue
                # At drop-off but NO matching items → ESCAPE!
                movement_bots[bid] = pos
                movement_targets[bid] = self._safe_idle_pos(pos)
                continue

            # Movement
            movement_bots[bid] = pos
            movement_targets[bid] = target if target else pos

        # Add stationary bots to PIBT
        for bid, act in immediate.items():
            pos = bot_positions[bid]
            if bid not in movement_bots:
                movement_bots[bid] = pos
                movement_targets[bid] = pos

        # PIBT urgency
        urgency: dict[int, int] = {}
        idle_bots: set[int] = set()
        for bid in movement_bots:
            pos = movement_bots[bid]
            g = self.bot_goal.get(bid, "park")

            if pos in self.drop_off_zones and bid not in immediate:
                urgency[bid] = -1  # ESCAPE
            elif g == "deliver":
                urgency[bid] = 0
            elif g == "pickup":
                urgency[bid] = 1
            # No scatter state needed — bots go directly to work
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

        # Save inventory snapshots
        for bot in bots:
            self._prev_inventory[bot["id"]] = list(bot.get("inventory", []))

        return {"actions": actions}

    def _assign_targeted_pickup(
        self, bid: int, bot_pos: Pos, bot_inv: list[str],
        items: list[dict], target_types: set[str], target_list: list[str],
        filling_types: list[str], demand: Counter, n_bots: int,
    ) -> bool:
        """Pick nearest item matching target_types. Zone-aware."""
        # Type budget: 3x overbooking
        type_budget = Counter(target_list)
        for t in type_budget:
            type_budget[t] *= 3
        for t in filling_types:
            if t in type_budget and type_budget[t] > 0:
                type_budget[t] -= 1

        # Zone preference
        zone_x_range = None
        if n_bots >= 20:
            zone_id = _bot_zone(bid, n_bots)
            zone_x_range = NIGHTMARE_ZONES[zone_id]

        best_item = None
        best_score = (9999, 9999)

        for item in items:
            if item["id"] in self._claimed_items:
                continue
            if item["type"] not in target_types:
                continue
            if type_budget.get(item["type"], 0) <= 0:
                continue

            item_pos = tuple(item["position"])
            pp = self._best_pickup_pos(bot_pos, item_pos)
            if pp is None:
                continue
            d = self._pe.distance(bot_pos, pp) or 9999
            if d >= 9999:
                continue

            zone_penalty = 0
            if zone_x_range:
                ix = item_pos[0]
                if ix < zone_x_range[0] or ix > zone_x_range[1]:
                    zone_penalty = 10

            demand_bonus = min(demand.get(item["type"], 0), 3)
            score = (d + zone_penalty - demand_bonus, d)
            if score < best_score:
                best_score = score
                best_item = item
                best_pp = pp

        if best_item:
            self.bot_goal[bid] = "pickup"
            self.bot_target[bid] = best_pp
            self.bot_item_type[bid] = best_item["type"]
            self.assigned_bots.add(bid)
            self._claimed_items.add(best_item["id"])
            filling_types.append(best_item["type"])
            return True
        return False

    def _assign_fill_pickup(
        self, bid: int, bot_pos: Pos, bot_inv: list[str],
        items: list[dict], all_inventory: list[str], filling_types: list[str],
        active_need: set[str], demand: Counter, n_bots: int,
    ) -> bool:
        """Assign fill pickup — ONLY items with demand > 0 (needed by upcoming orders)."""
        covered = Counter(all_inventory + filling_types)

        # Zone preference
        zone_x_range = None
        if n_bots >= 20:
            zone_id = _bot_zone(bid, n_bots)
            zone_x_range = NIGHTMARE_ZONES[zone_id]

        best_item = None
        best_score = (9999, 9999)

        for item in items:
            if item["id"] in self._claimed_items:
                continue
            # ONLY pick items that are needed by upcoming orders
            if demand.get(item["type"], 0) <= 0:
                continue

            item_pos = tuple(item["position"])
            if zone_x_range:
                ix = item_pos[0]
                if ix < zone_x_range[0] or ix > zone_x_range[1]:
                    continue

            pp = self._best_pickup_pos(bot_pos, item_pos)
            if pp is None:
                continue
            d = self._pe.distance(bot_pos, pp) or 9999
            if d >= 9999:
                continue

            is_active = item["type"] in active_need
            if is_active:
                priority = 0
            else:
                priority = 1 + covered.get(item["type"], 0)

            demand_bonus = min(demand.get(item["type"], 0), 2)
            score = (priority, d - demand_bonus)
            if score < best_score:
                best_score = score
                best_item = item
                best_pp = pp

        if best_item:
            self.bot_goal[bid] = "pickup"
            self.bot_target[bid] = best_pp
            self.bot_item_type[bid] = best_item["type"]
            self.assigned_bots.add(bid)
            self._claimed_items.add(best_item["id"])
            filling_types.append(best_item["type"])
            return True
        return False


def run_genome(recon_path: str, genome: Genome) -> tuple[int, int, int]:
    """Run a genome through the validated sim. Returns (score, orders, rounds_used)."""
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
        order_sequence=recon.get("order_sequence", []),
    )

    from Simulering.offline.simulator import Simulator
    sim = Simulator.from_recon_file(recon_path)

    # Run manually to capture mid-game score (for velocity optimization)
    state = sim.reset()
    mid_score = 0
    for r in range(sim.max_rounds):
        state_dict = state.to_dict()
        response = strategy(state_dict)
        actions = response.get("actions", [])
        state, game_over = sim.step(actions)
        if r == 249:  # Score at round 250 (halfway)
            mid_score = sim._score
        if game_over:
            break

    return sim._score, sim._orders_completed, sim._round, mid_score


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
    score, orders, rounds, mid_score = run_genome(args.recon, genome)
    print(f"Score: {score}, Orders: {orders}, Rounds: {rounds}, Mid250: {mid_score}, Time: {time.time()-t0:.1f}s")
