"""
OrderSolver: min-makespan assignment for active orders.

Given an active order + bot states, finds the bot-to-item assignment that
minimizes **makespan** (round when ALL items are delivered).

Key innovation over v1 (Hungarian min-sum):
- Models drop-off sequencing explicitly (bots queue, only 1 delivers/round)
- Optimizes for ORDER COMPLETION speed, not item throughput
- Brute-force partitioning for <= 6 items, greedy+2opt for larger
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from itertools import product, permutations
from typing import Optional

from bot.models import Bot, Item, Order, Pos
from bot.engine.world_model import WorldModel
from bot.strategy.task import Task, TaskType, Route, RouteStop

logger = logging.getLogger(__name__)


@dataclass
class BotPlan:
    """What a single bot should do for the current order."""
    bot_id: int
    pickups: list[tuple[Item, Pos]]  # (item, pickup_pos) in pickup order
    deliver: bool = True
    estimated_arrival: int = 0  # Round bot arrives at drop-off
    delivery_round: int = 0    # Round bot actually delivers (after queue)
    # Which pickups are for preview order (index into pickups list)
    preview_indices: list[int] = field(default_factory=list)


@dataclass
class OrderPlan:
    """Complete plan for fulfilling an order."""
    order_id: str
    bot_plans: dict[int, BotPlan]  # bot_id -> plan
    makespan: int = 0              # Rounds until order complete
    idle_bots: list[int] = field(default_factory=list)
    preview_order_id: str | None = None  # Preview order ID for PRE_PICK tasks


class OrderSolver:
    """Solves optimal bot-to-item assignment for min-makespan order completion."""

    def __init__(self) -> None:
        # Per-solve caches (cleared each call)
        self._pickup_cache: dict[tuple[Pos, Pos], Pos | None] = {}
        self._dist_cache: dict[tuple[Pos, Pos], int] = {}
        self._solve_candidates: dict[str, list[Item]] = {}  # type -> shelf items

    def _cached_pickup(self, world: WorldModel, from_pos: Pos, shelf_pos: Pos) -> Pos | None:
        key = (from_pos, shelf_pos)
        if key not in self._pickup_cache:
            self._pickup_cache[key] = world.best_pickup_position(from_pos, shelf_pos)
        return self._pickup_cache[key]

    def _cached_dist(self, world: WorldModel, a: Pos, b: Pos) -> int:
        key = (a, b)
        if key not in self._dist_cache:
            self._dist_cache[key] = world.distance(a, b)
        return self._dist_cache[key]

    def solve(
        self,
        world: WorldModel,
        bots: list[Bot],
        order: Order,
        claimed_items: set[str],
        current_round: int,
        in_transit_types: list[str] | None = None,
        preview_order: Order | None = None,
    ) -> OrderPlan:
        """
        Find assignment that completes order fastest.

        When preview_order is given, merges active + preview into one queue
        (conveyor belt model). Active items get priority, preview items fill
        remaining capacity. All items are KNOWN — no speculation.

        Steps:
        1. Identify needed items (order remaining minus bot inventories AND in-transit)
        2. Find candidate items on map
        3. Enumerate partitions (assign items to bots)
        4. Score each partition by makespan (with drop-off sequencing)
        5. Return best
        """
        drop_off = world.state.drop_off  # Primary; per-bot uses world.nearest_drop_off()

        # Clear per-solve caches
        self._pickup_cache.clear()
        self._dist_cache.clear()
        self._solve_candidates.clear()

        # Step 1: What items do bots already carry that match this order?
        remaining = list(order.items_remaining)

        # Deduct items already in-transit from other bots
        if in_transit_types:
            temp = list(remaining)
            for t in in_transit_types:
                if t in temp:
                    temp.remove(t)
            remaining = temp

        # Track boundary between active and preview items
        active_count = len(remaining)

        # Append preview items to the queue (conveyor belt)
        preview_remaining: list[str] = []
        if preview_order and not world.is_endgame():
            preview_remaining = list(preview_order.items_remaining)
            # Deduct preview items already in bot inventories (PRE_PICK bots)
            for bot in bots:
                temp_prev = list(preview_remaining)
                for inv in bot.inventory:
                    if inv in temp_prev:
                        temp_prev.remove(inv)
                preview_remaining = temp_prev
            # Also deduct in-transit preview items
            if in_transit_types:
                temp_prev = list(preview_remaining)
                for t in in_transit_types:
                    if t in temp_prev:
                        temp_prev.remove(t)
                preview_remaining = temp_prev
        bots_with_matching = {}  # bot_id -> list of matching inv types
        for bot in bots:
            matching = []
            temp_remaining = list(remaining)
            for inv in bot.inventory:
                if inv in temp_remaining:
                    matching.append(inv)
                    temp_remaining.remove(inv)
            if matching:
                bots_with_matching[bot.id] = matching
                remaining = temp_remaining

        if not remaining and not preview_remaining:
            # All items already in inventories — just deliver
            return self._plan_delivery_only(world, bots, order, bots_with_matching, drop_off)

        # Merge active + preview into unified queue
        if preview_remaining:
            logger.debug("Conveyor belt R%d: active=%s, preview=%s",
                         current_round, remaining, preview_remaining)
        unified_remaining = remaining + preview_remaining

        if not unified_remaining and not remaining:
            return self._plan_delivery_only(world, bots, order, bots_with_matching, drop_off)

        # Step 2: Find candidate items on map for each needed type
        # Store item with shelf position — pickup side chosen per-bot in _evaluate_partition
        needed_types = Counter(unified_remaining)
        candidates: dict[str, list[Item]] = {}  # type -> [item]
        for item_type, count in needed_types.items():
            type_candidates = []
            for item in world.items_of_type(item_type):
                if item.id in claimed_items:
                    continue
                # Verify at least one bot can reach this item
                reachable = any(
                    self._cached_pickup(world, b.position, item.position) is not None
                    for b in bots
                )
                if not reachable:
                    continue
                type_candidates.append(item)
            # Sort by total trip cost: nearest bot distance + distance to drop-off
            # Zone-aware: prefer items in the same zone as the bot
            def _item_cost(it: Item) -> float:
                best_cost = 9999.0
                for b in bots:
                    pp = self._cached_pickup(world, b.position, it.position)
                    if pp is None:
                        continue
                    cost = self._cached_dist(world, b.position, pp) + self._cached_dist(world, pp, drop_off) * 0.5
                    # Zone penalty: out-of-zone items cost more for this bot
                    bz = world.bot_zone(b.id)
                    if bz and not world.item_in_zone(it.position, bz):
                        cost += 20
                    best_cost = min(best_cost, cost)
                return best_cost
            type_candidates.sort(key=_item_cost)
            # Keep best candidates per type (more than needed for partition diversity)
            candidates[item_type] = type_candidates[:max(3, count + 2)]

        # Store all candidates for per-bot shelf resolution
        self._solve_candidates = candidates

        # Check if we can even fulfill the order
        for item_type, count in needed_types.items():
            if len(candidates.get(item_type, [])) < count:
                # Not enough items on map — partial plan
                logger.debug("OrderSolver: can't fulfill %s (need %d, have %d)",
                             item_type, count, len(candidates.get(item_type, [])))

        # Build flat list of items to assign — active items first, then preview
        # For types needing multiple, create camp_ items at best shelf
        # (items are infinite — can pick from same shelf repeatedly)
        active_types_needed = Counter(remaining)
        preview_types_needed = Counter(preview_remaining)

        items_to_assign: list[Item] = []
        is_preview_item: list[bool] = []  # parallel array tracking active vs preview

        def _add_items_for(types_needed: Counter, is_preview: bool):
            for item_type, count in types_needed.items():
                avail = candidates.get(item_type, [])
                if not avail:
                    continue
                best_item = avail[0]
                for k in range(count):
                    if k == 0:
                        items_to_assign.append(best_item)
                    else:
                        camp = Item(
                            id=f"camp_{item_type}_{k}{'_p' if is_preview else ''}",
                            type=item_type,
                            position=best_item.position,
                        )
                        items_to_assign.append(camp)
                    is_preview_item.append(is_preview)

        _add_items_for(active_types_needed, False)
        _add_items_for(preview_types_needed, True)

        # Sort active items: FARTHEST from drop-off FIRST.
        # This ensures far items start traveling early while near items
        # deliver quickly. Last remaining item should be the NEAREST shelf.
        if active_count > 1:
            active_items = items_to_assign[:active_count]
            active_preview = is_preview_item[:active_count]
            # Sort by distance to nearest drop-off (descending = farthest first)
            def _dist_to_drop(item):
                pp = self._cached_pickup(world, drop_off, item.position)
                if pp is None:
                    return 0
                return self._cached_dist(world, pp, drop_off)
            sorted_pairs = sorted(zip(active_items, active_preview),
                                  key=lambda x: -_dist_to_drop(x[0]))
            for i, (item, is_prev) in enumerate(sorted_pairs):
                items_to_assign[i] = item
                is_preview_item[i] = is_prev

        if not items_to_assign:
            return OrderPlan(
                order_id=order.id,
                bot_plans={},
                idle_bots=[b.id for b in bots],
            )

        # Filter bots that can actually pick (have capacity)
        pickable_bots = [b for b in bots if len(b.inventory) < 3]
        if not pickable_bots:
            # All bots full — those with matching items deliver
            return self._plan_delivery_only(world, bots, order, bots_with_matching, drop_off)

        # Step 3+4: Enumerate partitions and find best makespan
        n_items = len(items_to_assign)
        n_bots = len(pickable_bots)

        if n_items <= 9 and n_bots <= 5:
            best_plan = self._brute_force_partition(
                world, pickable_bots, items_to_assign, order,
                bots_with_matching, drop_off,
                is_preview=is_preview_item if any(is_preview_item) else None,
            )
        else:
            best_plan = self._greedy_partition(
                world, pickable_bots, items_to_assign, order,
                bots_with_matching, drop_off,
                is_preview=is_preview_item if any(is_preview_item) else None,
            )

        # Mark idle bots and preview order ID
        assigned_ids = set(best_plan.bot_plans.keys())
        best_plan.idle_bots = [b.id for b in bots if b.id not in assigned_ids]
        if preview_order:
            best_plan.preview_order_id = preview_order.id

        return best_plan

    def _brute_force_partition(
        self,
        world: WorldModel,
        bots: list[Bot],
        items: list[Item],
        order: Order,
        bots_with_matching: dict[int, list[str]],
        drop_off: Pos,
        is_preview: list[bool] | None = None,
    ) -> OrderPlan:
        """Enumerate all possible item-to-bot assignments, pick min makespan."""
        import time as _time
        n_items = len(items)
        n_bots = len(bots)
        bot_capacities = {b.id: 3 - len(b.inventory) for b in bots}

        best_plan: Optional[OrderPlan] = None
        best_makespan = 99999
        t_start = _time.perf_counter()
        evaluated = 0

        for assignment in product(range(n_bots), repeat=n_items):
            # Time-bound: return best so far after 300ms
            if evaluated & 0xFF == 0 and (_time.perf_counter() - t_start) > 0.3:
                logger.debug("Brute-force time-bounded after %d evals (%.0fms)",
                             evaluated, (_time.perf_counter() - t_start) * 1000)
                break

            bot_items: dict[int, list[Item]] = {b.id: [] for b in bots}
            bot_preview_indices: dict[int, list[int]] = {b.id: [] for b in bots}
            valid = True
            for item_idx, bot_idx in enumerate(assignment):
                bot = bots[bot_idx]
                if is_preview and is_preview[item_idx]:
                    bot_preview_indices[bot.id].append(len(bot_items[bot.id]))
                bot_items[bot.id].append(items[item_idx])
                if len(bot_items[bot.id]) > bot_capacities[bot.id]:
                    valid = False
                    break
            if not valid:
                continue
            evaluated += 1

            plan = self._evaluate_partition(
                world, bots, bot_items, order, bots_with_matching, drop_off,
                bot_preview_indices=bot_preview_indices,
            )
            if plan.makespan < best_makespan:
                best_makespan = plan.makespan
                best_plan = plan

        if best_plan is None:
            return self._greedy_partition(
                world, bots, items, order, bots_with_matching, drop_off,
                is_preview=is_preview,
            )
        return best_plan

    def _greedy_partition(
        self,
        world: WorldModel,
        bots: list[Bot],
        items: list[Item],
        order: Order,
        bots_with_matching: dict[int, list[str]],
        drop_off: Pos,
        is_preview: list[bool] | None = None,
    ) -> OrderPlan:
        """Greedy assignment: each item goes to nearest bot, favoring batching.

        For 10+ bots: prefers assigning items to bots that already have items
        (reducing total deliverers and drop-off congestion).
        """
        bot_items: dict[int, list[Item]] = {b.id: [] for b in bots}
        bot_preview_indices: dict[int, list[int]] = {b.id: [] for b in bots}
        bot_capacities = {b.id: 3 - len(b.inventory) for b in bots}

        for i, item in enumerate(items):
            item_is_preview = is_preview and is_preview[i]
            best_bot_id = None
            best_cost = 99999
            for bot in bots:
                if len(bot_items[bot.id]) >= bot_capacities[bot.id]:
                    continue
                if bot_items[bot.id]:
                    last_item = bot_items[bot.id][-1]
                    last_pickup = self._cached_pickup(world, bot.position, last_item.position)
                    last_pos = last_pickup if last_pickup else bot.position
                else:
                    last_pos = bot.position
                pickup = self._cached_pickup(world, last_pos, item.position)
                if pickup is None:
                    continue
                cost = self._cached_dist(world, last_pos, pickup)
                if cost < best_cost:
                    best_cost = cost
                    best_bot_id = bot.id
            if best_bot_id is not None:
                if item_is_preview:
                    bot_preview_indices[best_bot_id].append(len(bot_items[best_bot_id]))
                bot_items[best_bot_id].append(item)

        return self._evaluate_partition(
            world, bots, bot_items, order, bots_with_matching, drop_off,
            bot_preview_indices=bot_preview_indices,
        )

    def _evaluate_partition(
        self,
        world: WorldModel,
        bots: list[Bot],
        bot_items: dict[int, list[Item]],
        order: Order,
        bots_with_matching: dict[int, list[str]],
        drop_off: Pos,
        bot_preview_indices: dict[int, list[int]] | None = None,
    ) -> OrderPlan:
        """
        Compute makespan for a specific item-to-bot partition.

        Key fixes over original:
        1. Per-bot pickup position: chooses best shelf side FOR EACH bot
        2. Active vs preview makespan: only active-item bots count for makespan
        3. Drop-off queue: only active-item bots enter the queue
        """
        bot_plans: dict[int, BotPlan] = {}
        active_arrivals: list[tuple[int, int]] = []  # (arrival, bot_id) for active items
        preview_arrivals: list[tuple[int, int]] = []  # for preview-only bots

        for bot in bots:
            items_for_bot = bot_items.get(bot.id, [])
            has_matching_inv = bot.id in bots_with_matching

            if not items_for_bot and not has_matching_inv:
                continue  # Bot not needed

            preview_idx_set = set(bot_preview_indices.get(bot.id, [])) if bot_preview_indices else set()

            # Determine if this bot carries any active items
            has_active_items = bool(items_for_bot) and any(
                i not in preview_idx_set for i in range(len(items_for_bot))
            )
            bot_contributes_active = has_active_items or has_matching_inv

            if items_for_bot:
                # TSP with per-bot pickup positions (unified route)
                ordered_items = self._optimal_tsp_for_bot(
                    world, bot.position, drop_off, items_for_bot,
                )
                # Remap preview indices after TSP reordering
                if preview_idx_set:
                    orig_preview = {id(items_for_bot[i]) for i in preview_idx_set}
                    reordered_preview = [j for j, (item, _) in enumerate(ordered_items) if id(item) in orig_preview]
                else:
                    reordered_preview = []
            else:
                ordered_items = []
                reordered_preview = []

            # Compute travel time using per-bot pickup positions
            if ordered_items:
                pos = bot.position
                travel = 0
                for _, pickup_pos in ordered_items:
                    travel += self._cached_dist(world, pos, pickup_pos)
                    travel += 1  # pick_up action
                    pos = pickup_pos
                travel += self._cached_dist(world, pos, drop_off)
            elif has_matching_inv:
                travel = self._cached_dist(world, bot.position, drop_off)
            else:
                continue

            arrival = travel + 1 + 1  # +1 drop_off, +1 PIBT buffer

            plan = BotPlan(
                bot_id=bot.id,
                pickups=ordered_items,
                estimated_arrival=arrival,
                preview_indices=reordered_preview,
            )
            bot_plans[bot.id] = plan

            if bot_contributes_active:
                active_arrivals.append((arrival, bot.id))
            else:
                preview_arrivals.append((arrival, bot.id))

        # Drop-off sequencing: only active bots enter the queue
        # Preview-only bots don't need to deliver (auto-delivery handles them)
        active_arrivals.sort()
        prev_delivery = 0
        for arrival, bot_id in active_arrivals:
            delivery_round = max(arrival, prev_delivery + 1)
            bot_plans[bot_id].delivery_round = delivery_round
            prev_delivery = delivery_round

        # Preview bots get delivery_round after active bots (but don't affect makespan)
        for arrival, bot_id in preview_arrivals:
            delivery_round = max(arrival, prev_delivery + 1)
            bot_plans[bot_id].delivery_round = delivery_round
            prev_delivery = delivery_round

        # Makespan = when active order is COMPLETE (not when all bots finish)
        if active_arrivals:
            active_bot_ids = {bid for _, bid in active_arrivals}
            active_makespan = max(
                bot_plans[bid].delivery_round for bid in active_bot_ids
            )
        else:
            active_makespan = 0

        return OrderPlan(
            order_id=order.id,
            bot_plans=bot_plans,
            makespan=active_makespan,
        )

    def _assign_preview_items(
        self,
        world: WorldModel,
        bots: list[Bot],
        preview_items: list[Item],
        plan: OrderPlan,
        drop_off: Pos,
    ) -> None:
        """Assign preview items to bots with leftover capacity.

        Only bots that are NOT already picking active items get preview work.
        This prevents preview pickups from delaying active order completion.
        """
        # Find bots with remaining capacity that aren't doing active pickups
        bot_capacities = {}
        for bot in bots:
            existing = plan.bot_plans.get(bot.id)
            active_pickups = 0
            if existing:
                active_pickups = len(existing.pickups) - len(existing.preview_indices)
            if active_pickups > 0:
                continue  # Don't add preview to bots with active pickups
            used = len(bot.inventory) + (len(existing.pickups) if existing else 0)
            remaining_cap = 3 - used
            if remaining_cap > 0:
                bot_capacities[bot.id] = remaining_cap

        if not bot_capacities:
            return

        # Greedy: assign each preview item to nearest bot with capacity
        for item in preview_items:
            best_bot_id = None
            best_cost = 99999
            for bot in bots:
                cap = bot_capacities.get(bot.id, 0)
                if cap <= 0:
                    continue
                existing = plan.bot_plans.get(bot.id)
                if existing and existing.pickups:
                    last_pos = existing.pickups[-1][1]  # last pickup position
                else:
                    last_pos = bot.position
                pickup = world.best_pickup_position(last_pos, item.position)
                if pickup is None:
                    continue
                cost = world.distance(last_pos, pickup)
                if cost < best_cost:
                    best_cost = cost
                    best_bot_id = bot.id
            if best_bot_id is None:
                continue

            bot_capacities[best_bot_id] -= 1
            bot_obj = next(b for b in bots if b.id == best_bot_id)
            pickup_pos = world.best_pickup_position(
                plan.bot_plans[best_bot_id].pickups[-1][1] if (best_bot_id in plan.bot_plans and plan.bot_plans[best_bot_id].pickups) else bot_obj.position,
                item.position,
            )
            if pickup_pos is None:
                continue

            if best_bot_id not in plan.bot_plans:
                plan.bot_plans[best_bot_id] = BotPlan(
                    bot_id=best_bot_id,
                    pickups=[(item, pickup_pos)],
                    deliver=True,
                    preview_indices=[0],
                )
            else:
                bp = plan.bot_plans[best_bot_id]
                bp.preview_indices.append(len(bp.pickups))
                bp.pickups.append((item, pickup_pos))

    def _resolve_items_for_bot(
        self, world: WorldModel, bot_pos: Pos, items: list[Item],
        drop_off: Pos,
    ) -> list[Item]:
        """Replace items with best shelf of same type for this bot.

        Uses round-trip cost (bot -> pickup -> drop_off) to pick the shelf
        that minimizes total travel, not just pickup distance. This prevents
        sending bots to nearby shelves that are far from drop_off.
        """
        if not self._solve_candidates:
            return items
        resolved = []
        changed = False
        for item in items:
            cands = self._solve_candidates.get(item.type)
            if not cands or len(cands) <= 1:
                resolved.append(item)
                continue
            best = item
            best_cost = 9999
            for c in cands:
                pp = self._cached_pickup(world, bot_pos, c.position)
                if pp is None:
                    continue
                # Round-trip: bot -> pickup + pickup -> drop_off
                cost = (self._cached_dist(world, bot_pos, pp)
                        + self._cached_dist(world, pp, drop_off))
                if cost < best_cost:
                    best_cost = cost
                    best = c
            if best is not item:
                changed = True
                if item.id.startswith("camp_"):
                    resolved.append(Item(id=item.id, type=item.type, position=best.position))
                else:
                    resolved.append(best)
            else:
                resolved.append(item)
        return resolved if changed else items

    def _plan_delivery_only(
        self,
        world: WorldModel,
        bots: list[Bot],
        order: Order,
        bots_with_matching: dict[int, list[str]],
        drop_off: Pos,
    ) -> OrderPlan:
        """All needed items are in inventories — just schedule deliveries."""
        bot_plans: dict[int, BotPlan] = {}
        arrivals: list[tuple[int, int]] = []

        for bot in bots:
            if bot.id not in bots_with_matching:
                continue
            travel = world.distance(bot.position, drop_off) + 1  # +1 for drop_off
            plan = BotPlan(
                bot_id=bot.id,
                pickups=[],
                estimated_arrival=travel,
            )
            bot_plans[bot.id] = plan
            arrivals.append((travel, bot.id))

        arrivals.sort()
        prev_delivery = 0
        for arrival, bot_id in arrivals:
            delivery_round = max(arrival, prev_delivery + 1)
            bot_plans[bot_id].delivery_round = delivery_round
            prev_delivery = delivery_round

        makespan = max((p.delivery_round for p in bot_plans.values()), default=0)
        idle = [b.id for b in bots if b.id not in bot_plans]

        return OrderPlan(
            order_id=order.id,
            bot_plans=bot_plans,
            makespan=makespan,
            idle_bots=idle,
        )

    def _optimal_tsp_for_bot(
        self,
        world: WorldModel,
        start: Pos,
        drop_off: Pos,
        items: list[Item],
    ) -> list[tuple[Item, Pos]]:
        """Order items by TSP, choosing best pickup side FOR THIS BOT.

        Each item's pickup position is chosen based on where the bot will
        be coming FROM, not a global "nearest bot" heuristic.
        """
        if len(items) <= 1:
            item = items[0]
            pp = self._cached_pickup(world, start, item.position)
            if pp is None:
                pp = item.position  # fallback
            return [(item, pp)]

        if len(items) <= 6:
            best_cost = float('inf')
            best_result = None
            for perm in permutations(range(len(items))):
                pos = start
                cost = 0
                result = []
                for idx in perm:
                    item = items[idx]
                    pp = self._cached_pickup(world, pos, item.position)
                    if pp is None:
                        cost = float('inf')
                        break
                    cost += self._cached_dist(world, pos, pp) + 1
                    result.append((item, pp))
                    pos = pp
                cost += self._cached_dist(world, pos, drop_off)
                if cost < best_cost:
                    best_cost = cost
                    best_result = list(result)
            return best_result if best_result else [(items[0], items[0].position)]
        else:
            # NN heuristic for large counts
            remaining = list(items)
            ordered = []
            pos = start
            while remaining:
                best_idx = -1
                best_cost = float('inf')
                best_pp = None
                for i, item in enumerate(remaining):
                    pp = self._cached_pickup(world, pos, item.position)
                    if pp is None:
                        continue
                    c = self._cached_dist(world, pos, pp)
                    if c < best_cost:
                        best_cost = c
                        best_idx = i
                        best_pp = pp
                if best_idx < 0:
                    break
                ordered.append((remaining.pop(best_idx), best_pp))
                pos = best_pp
            return ordered
