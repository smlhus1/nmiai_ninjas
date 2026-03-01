"""
Offline optimal planner for Grocery Bot.
Takes recon data, brute-forces optimal pickup sequences with cross-order pipelining.

Supports:
- Single bot (Easy): exact optimal via permutation enumeration
- Multi bot (Medium-Expert): zone-based assignment + per-bot optimal ordering

Usage:
    from offline.planner import OfflinePlanner
    planner = OfflinePlanner(grid, drop_off, spawn_positions)
    plan = planner.plan_full_game(order_sequence, item_type_positions)
    actions = planner.to_action_sequence(plan)
"""
from __future__ import annotations
from itertools import permutations, combinations
from typing import Optional
from .grid import Grid, Pos, move_action


class ItemTarget:
    """An item to pick up."""
    __slots__ = ("item_type", "shelf_pos", "pickup_pos", "item_id")

    def __init__(self, item_type: str, shelf_pos: Pos,
                 pickup_pos: Pos, item_id: str = ""):
        self.item_type = item_type
        self.shelf_pos = shelf_pos
        self.pickup_pos = pickup_pos
        self.item_id = item_id

    def __repr__(self):
        return f"Item({self.item_type}@{self.shelf_pos})"


class Batch:
    """A group of 1-3 items to pick up in one trip."""
    __slots__ = ("items", "cost", "start_pos", "end_pos")

    def __init__(self, items: list[ItemTarget], cost: int,
                 start_pos: Pos, end_pos: Pos):
        self.items = items
        self.cost = cost  # total rounds: navigate + pick + deliver
        self.start_pos = start_pos
        self.end_pos = end_pos

    def __repr__(self):
        types = [i.item_type for i in self.items]
        return f"Batch({types}, cost={self.cost})"


class OrderPlan:
    """Planned execution for one order."""
    __slots__ = ("order_id", "batches", "total_cost",
                 "cross_order_items")

    def __init__(self, order_id: str, batches: list[Batch],
                 total_cost: int, cross_order_items: list[ItemTarget] = None):
        self.order_id = order_id
        self.batches = batches
        self.total_cost = total_cost
        self.cross_order_items = cross_order_items or []

    def __repr__(self):
        return f"OrderPlan({self.order_id}, batches={len(self.batches)}, cost={self.total_cost})"


class OfflinePlanner:
    """
    Brute-force optimal planner.
    
    For Easy (1 bot, 3-4 items/order):
      - 3! = 6 permutations per order
      - 4 items → 4! = 24 permutations with partition into batches of 3
      - Total: ~500 evaluations for entire game. Runs in <10ms.
    
    For Medium+ (3+ bots):
      - Assign items to bots via zone proximity
      - Per-bot: same brute-force as Easy
      - Coordinate drop-off timing (1-round gaps)
    """

    def __init__(self, grid: Grid, drop_off: Pos,
                 spawn_positions: list[Pos], inventory_cap: int = 3):
        self.grid = grid
        self.drop_off = drop_off
        self.spawns = spawn_positions
        self.cap = inventory_cap
        # Precompute BFS from drop-off (most queried destination)
        self.grid.bfs_from(drop_off)

    # ------------------------------------------------------------------
    # Core: compute cost of a specific pickup sequence
    # ------------------------------------------------------------------

    def _sequence_cost(self, bot_pos: Pos, items: list[ItemTarget],
                       deliver: bool = True) -> int:
        """
        Cost (in rounds) to pick up items in order, then deliver.
        Each pickup = navigate to pickup_pos + 1 action.
        Deliver = navigate to drop_off + 1 action.
        """
        cost = 0
        pos = bot_pos
        for item in items:
            dist = self.grid.distance(pos, item.pickup_pos)
            if dist >= 9999:
                return 9999  # unreachable
            cost += dist + 1  # +1 for pick_up action
            pos = item.pickup_pos
        if deliver:
            dist = self.grid.distance(pos, self.drop_off)
            if dist >= 9999:
                return 9999
            cost += dist + 1  # +1 for drop_off action
        return cost

    # ------------------------------------------------------------------
    # Brute-force: best ordering for a single batch (≤3 items)
    # ------------------------------------------------------------------

    def _best_batch_order(self, bot_pos: Pos,
                          items: list[ItemTarget]) -> tuple[list[ItemTarget], int]:
        """Find optimal pickup order within a batch. Returns (ordered_items, cost)."""
        assert len(items) <= self.cap
        best_cost = 9999
        best_order = items
        for perm in permutations(items):
            cost = self._sequence_cost(bot_pos, list(perm))
            if cost < best_cost:
                best_cost = cost
                best_order = list(perm)
        return best_order, best_cost

    # ------------------------------------------------------------------
    # Partition items into batches of ≤cap, find best partition
    # ------------------------------------------------------------------

    def _all_partitions(self, items: list[ItemTarget],
                        cap: int) -> list[list[list[ItemTarget]]]:
        """
        Enumerate all ways to partition items into groups of ≤cap.
        For 3 items, cap 3: 1 partition (all together)
        For 4 items, cap 3: 4 partitions (which 3 go first)
        For 5 items, cap 3: C(5,3) = 10 partitions × C(2,2) = 10
        For 6 items, cap 3: C(6,3) = 20 partitions
        """
        n = len(items)
        if n <= cap:
            return [[items]]

        result = []
        # Choose first batch (≤cap items), recurse on remainder
        min_first = max(1, n - cap * ((n - 1) // cap))
        max_first = min(cap, n)

        for size in range(min_first, max_first + 1):
            for combo in combinations(range(n), size):
                first = [items[i] for i in combo]
                rest = [items[i] for i in range(n) if i not in combo]
                if not rest:
                    result.append([first])
                else:
                    for sub_partition in self._all_partitions(rest, cap):
                        result.append([first] + sub_partition)

        # Deduplicate (order of batches matters for travel cost)
        # Keep all — the travel between batches means order matters
        return result

    def _best_partition(self, bot_pos: Pos,
                        items: list[ItemTarget]) -> tuple[list[Batch], int]:
        """
        Find optimal partition of items into batches and ordering within each.
        Returns (list of Batch, total cost).
        """
        if len(items) <= self.cap:
            ordered, cost = self._best_batch_order(bot_pos, items)
            batch = Batch(ordered, cost, bot_pos, self.drop_off)
            return [batch], cost

        partitions = self._all_partitions(items, self.cap)
        best_cost = 9999
        best_batches = []

        for partition in partitions:
            # For each partition, try all orderings of batches
            for batch_perm in permutations(range(len(partition))):
                total_cost = 0
                batches = []
                pos = bot_pos

                for bi in batch_perm:
                    batch_items = partition[bi]
                    ordered, cost = self._best_batch_order(pos, batch_items)
                    batches.append(Batch(ordered, cost, pos, self.drop_off))
                    total_cost += cost
                    pos = self.drop_off  # after delivery, bot is at drop-off

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_batches = batches

        return best_batches, best_cost

    # ------------------------------------------------------------------
    # Resolve item types to concrete shelf positions
    # ------------------------------------------------------------------

    def _resolve_items(self, item_types: list[str],
                       type_positions: dict[str, list[Pos]],
                       bot_pos: Pos,
                       used_positions: Optional[set[Pos]] = None
                       ) -> list[ItemTarget]:
        """
        Map item type names to concrete ItemTargets with best shelf positions.
        Handles duplicate types (e.g., 2x yogurt) by assigning different shelves.
        """
        used = set(used_positions) if used_positions else set()
        targets = []
        for itype in item_types:
            positions = type_positions.get(itype, [])
            best_pos = None
            best_dist = 9999
            for shelf_pos in positions:
                if shelf_pos in used:
                    continue
                pickup = self.grid.best_pickup_pos(bot_pos, shelf_pos)
                if pickup is None:
                    continue
                dist = self.grid.distance(bot_pos, pickup)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = shelf_pos

            if best_pos is None:
                # Fallback: allow reuse (items respawn)
                for shelf_pos in positions:
                    pickup = self.grid.best_pickup_pos(bot_pos, shelf_pos)
                    if pickup:
                        best_pos = shelf_pos
                        break

            if best_pos:
                pickup = self.grid.best_pickup_pos(bot_pos, best_pos)
                targets.append(ItemTarget(itype, best_pos, pickup))
                used.add(best_pos)

        return targets

    # ------------------------------------------------------------------
    # Plan single order (with optional cross-order pipelining)
    # ------------------------------------------------------------------

    def plan_order(self, bot_pos: Pos, order_items: list[str],
                   type_positions: dict[str, list[Pos]],
                   preview_items: Optional[list[str]] = None
                   ) -> OrderPlan:
        """
        Plan optimal execution for one order.
        
        If preview_items provided and the last batch has spare capacity,
        pack preview items for auto-delivery on order transition.
        """
        items = self._resolve_items(order_items, type_positions, bot_pos)

        if not items:
            return OrderPlan("?", [], 0)

        batches, total_cost = self._best_partition(bot_pos, items)
        cross_order = []

        # --- Cross-order pipelining ---
        if preview_items and batches:
            last_batch = batches[-1]
            free_slots = self.cap - len(last_batch.items)

            if free_slots > 0:
                # Find preview items that can be packed in last batch
                # Position: start from last pickup pos of last batch
                last_pickup = last_batch.items[-1].pickup_pos
                preview_targets = self._resolve_items(
                    preview_items[:free_slots], type_positions, last_pickup
                )

                if preview_targets:
                    # Recalculate last batch with added preview items
                    combined = last_batch.items + preview_targets
                    new_ordered, new_cost = self._best_batch_order(
                        last_batch.start_pos, combined
                    )
                    old_cost = last_batch.cost
                    extra_cost = new_cost - old_cost

                    # Only add if marginal cost is reasonable
                    # (preview items save an entire future trip: ~12 rounds)
                    trip_savings = 12 * len(preview_targets)
                    if extra_cost < trip_savings:
                        batches[-1] = Batch(new_ordered, new_cost,
                                            last_batch.start_pos, self.drop_off)
                        total_cost = sum(b.cost for b in batches)
                        cross_order = preview_targets

        return OrderPlan("?", batches, total_cost, cross_order)

    # ------------------------------------------------------------------
    # Plan full game (all orders in sequence)
    # ------------------------------------------------------------------

    def plan_full_game(self, order_sequence: list[dict],
                       type_positions: dict[str, list[Pos]],
                       bot_index: int = 0
                       ) -> list[OrderPlan]:
        """
        Plan all orders for a single bot.
        
        order_sequence: list of {"id": ..., "items_required": [...]}
        type_positions: {"milk": [(2,1), (4,3)], ...}
        
        Returns list of OrderPlan. Stops when 300 rounds exceeded.
        """
        # Convert position lists to tuple lists
        type_pos = {t: [tuple(p) if isinstance(p, list) else p for p in ps]
                    for t, ps in type_positions.items()}

        pos = self.spawns[bot_index] if bot_index < len(self.spawns) else self.spawns[0]
        total_rounds = 0
        plans = []

        for i, order in enumerate(order_sequence):
            # Preview = next order (if exists)
            preview = order_sequence[i + 1] if i + 1 < len(order_sequence) else None
            preview_items = preview["items_required"] if preview else None

            # Account for cross-order items already delivered
            remaining_items = list(order["items_required"])
            if plans and plans[-1].cross_order_items:
                for xo in plans[-1].cross_order_items:
                    if xo.item_type in remaining_items:
                        remaining_items.remove(xo.item_type)

            if not remaining_items:
                # Order already complete from cross-order!
                plans.append(OrderPlan(order["id"], [], 0))
                continue

            plan = self.plan_order(pos, remaining_items, type_pos, preview_items)
            plan.order_id = order["id"]

            if total_rounds + plan.total_cost > 300:
                # Can we at least partially complete?
                # Try smaller batches until we fit
                for num_batches in range(len(plan.batches), 0, -1):
                    partial_cost = sum(b.cost for b in plan.batches[:num_batches])
                    if total_rounds + partial_cost <= 300:
                        plan.batches = plan.batches[:num_batches]
                        plan.total_cost = partial_cost
                        plan.cross_order_items = []  # no cross-order in endgame
                        plans.append(plan)
                        total_rounds += partial_cost
                        break
                break

            plans.append(plan)
            total_rounds += plan.total_cost
            pos = self.drop_off  # after delivery, bot is at drop-off

        # Summary
        total_items = sum(
            len(b.items) for p in plans for b in p.batches
        )
        total_orders = sum(1 for p in plans if p.batches)
        cross_items = sum(len(p.cross_order_items) for p in plans)

        print(f"\n=== OFFLINE PLAN SUMMARY ===")
        print(f"  Orders planned: {total_orders}")
        print(f"  Total items: {total_items}")
        print(f"  Cross-order items: {cross_items}")
        print(f"  Estimated rounds: {total_rounds}/300")
        est_score = total_items + total_orders * 5
        print(f"  Estimated score: {est_score} "
              f"({total_items} items + {total_orders}×5 bonus)")
        print()

        return plans

    # ------------------------------------------------------------------
    # Convert plan to action sequence
    # ------------------------------------------------------------------

    def to_action_sequence(self, plans: list[OrderPlan],
                           bot_id: int = 0) -> list[dict]:
        """
        Convert OrderPlans to a flat list of per-round actions.
        Each action: {"bot": 0, "action": "move_up"} or
                     {"bot": 0, "action": "pick_up", "item_id": "..."}
        
        Note: item_ids must be filled in at replay time since they
        change between runs. Use item_type + shelf_pos to match.
        """
        actions = []
        pos = self.spawns[bot_id] if bot_id < len(self.spawns) else self.spawns[0]

        for plan in plans:
            for batch in plan.batches:
                for item in batch.items:
                    # Navigate to pickup position
                    path = self.grid.find_path(pos, item.pickup_pos)
                    for step in path:
                        actions.append({
                            "bot": bot_id,
                            "action": move_action(pos, step),
                        })
                        pos = step

                    # Pick up (item_id resolved at runtime)
                    actions.append({
                        "bot": bot_id,
                        "action": "pick_up",
                        "item_type": item.item_type,
                        "shelf_pos": list(item.shelf_pos),
                        # item_id filled at replay time
                    })

                # Navigate to drop-off
                path = self.grid.find_path(pos, self.drop_off)
                for step in path:
                    actions.append({
                        "bot": bot_id,
                        "action": move_action(pos, step),
                    })
                    pos = step

                # Drop off
                actions.append({
                    "bot": bot_id,
                    "action": "drop_off",
                })

        print(f"Generated {len(actions)} actions for bot {bot_id}")
        return actions

    # ------------------------------------------------------------------
    # Multi-bot: zone-based assignment
    # ------------------------------------------------------------------

    def assign_zones(self, num_bots: int) -> dict[int, set[Pos]]:
        """
        Assign shelf positions to bots via Voronoi on spawn positions.
        Returns {bot_id: set of shelf positions in zone}.
        """
        zones: dict[int, set[Pos]] = {i: set() for i in range(num_bots)}

        for shelf in self.grid.shelves:
            best_bot = 0
            best_dist = 9999
            for i in range(num_bots):
                spawn = self.spawns[i] if i < len(self.spawns) else self.spawns[0]
                d = self.grid.distance(spawn, shelf)
                if d < best_dist:
                    best_dist = d
                    best_bot = i
            zones[best_bot].add(shelf)

        return zones

    def plan_multi_bot(self, order_sequence: list[dict],
                       type_positions: dict[str, list[Pos]],
                       num_bots: int) -> dict[int, list[dict]]:
        """
        Plan for multiple bots. Returns {bot_id: action_sequence}.
        
        Strategy:
        1. Assign zones via Voronoi
        2. For each order, assign items to bots by zone
        3. Per-bot: brute-force optimal ordering
        4. Coordinate drop-off timing (bot 0 first, then 1, etc.)
        """
        zones = self.assign_zones(num_bots)
        type_pos = {t: [tuple(p) if isinstance(p, list) else p for p in ps]
                    for t, ps in type_positions.items()}

        # For now: bot 0 gets the optimal single-bot plan,
        # other bots assist with items in their zones
        # TODO: proper multi-bot VRP
        all_actions: dict[int, list[dict]] = {}

        # Simple approach: round-robin item assignment
        for bot_id in range(num_bots):
            spawn = self.spawns[bot_id] if bot_id < len(self.spawns) else self.spawns[0]

            # Filter items by zone preference
            zone_type_pos: dict[str, list[Pos]] = {}
            for itype, positions in type_pos.items():
                zone_positions = [p for p in positions if p in zones[bot_id]]
                # Include non-zone positions as fallback
                other_positions = [p for p in positions if p not in zones[bot_id]]
                zone_type_pos[itype] = zone_positions + other_positions

            # Assign every Nth order to this bot
            bot_orders = order_sequence[bot_id::num_bots]

            sub_planner = OfflinePlanner(
                self.grid, self.drop_off, [spawn], self.cap
            )
            plans = sub_planner.plan_full_game(bot_orders, zone_type_pos, 0)
            all_actions[bot_id] = sub_planner.to_action_sequence(plans, bot_id)

        return all_actions


# ------------------------------------------------------------------
# Convenience: plan from analyzer output
# ------------------------------------------------------------------

def plan_from_analysis(analysis: dict, grid: Grid) -> list[dict]:
    """
    Take output from analyzer.analyze() and produce action sequence.
    Returns list of per-round actions ready for replay.
    """
    drop_off = tuple(analysis["drop_off"])
    # Infer spawn from first bot position (round 0 not in analysis, use heuristic)
    # For now: assume spawn is at drop_off (conservative)
    spawn = drop_off

    type_positions = {
        t: [tuple(p) for p in ps]
        for t, ps in analysis["type_shelf_positions"].items()
    }

    order_seq = [
        {"id": o["id"], "items_required": o["items"]}
        for o in analysis["order_sequence"]
    ]

    planner = OfflinePlanner(grid, drop_off, [spawn])
    plans = planner.plan_full_game(order_seq, type_positions)
    return planner.to_action_sequence(plans)
