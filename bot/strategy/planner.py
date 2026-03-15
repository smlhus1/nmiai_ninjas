"""
TaskPlanner: assigns tasks to bots each round.

This is the strategic brain. It decides WHAT each bot should do.
The ActionResolver decides HOW (pathing, collision avoidance).

Key principles:
- Global optimization: consider all bots together, not greedily per-bot
- Sticky assignments: don't reassign unless the task is invalid or done
- No double-booking: two bots should not go for the same item
- Prioritize order completion over random item pickup
- Endgame mode: when active order can't be completed, optimize items/round
- Preview pre-staging: idle bots pre-pick items for upcoming orders
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from bot.models import GameState, Bot, Item, Order, OrderStatus
from bot.engine.world_model import WorldModel
from bot.strategy.task import Task, TaskType, BotAssignment
from bot.strategy.planner_assignment import AssignmentMixin
from bot.strategy.planner_validation import ValidationMixin
from bot.strategy.planner_decisions import DecisionsMixin

logger = logging.getLogger(__name__)


class TaskPlanner(AssignmentMixin, ValidationMixin, DecisionsMixin):
    """
    Stateful planner — takes world model + current assignments,
    returns updated assignments.
    """

    def __init__(self) -> None:
        self._prev_inventory: dict[int, tuple[str, ...]] = {}
        self._stuck_deliver_rounds: dict[int, int] = {}
        self._stuck_pick_rounds: dict[int, int] = {}
        self._blacklisted_items: dict[str, int] = {}
        self._last_active_order_id: str | None = None
        self._gate_rounds_held: int = 0
        self._gate_order_id: str | None = None

    def blacklist_item(self, item_id: str, expiry_round: int) -> None:
        """Register an item as blacklisted until expiry_round (e.g. after stuck clear)."""
        self._blacklisted_items[item_id] = expiry_round

    def _count_preview_items_in_inventories(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment] | None = None,
    ) -> tuple[Counter[str], Counter[str]]:
        """(have_per_type, need_per_type) for preview order.

        Counts inventory items on bots that are PRE_PICKing or have no active-order task
        (to avoid counting active-order items as preview-ready).
        Also counts items on bots with DELIVER tasks that have preview matches (staging for auto-delivery).
        """
        preview_orders = world.state.preview_orders
        if not preview_orders:
            return (Counter(), Counter())
        preview = preview_orders[0]
        need = Counter(preview.items_remaining)
        active = world.state.active_orders
        active_types = set(active[0].items_remaining) if active else set()
        have: Counter[str] = Counter()
        for bot in world.state.bots:
            a = assignments.get(bot.id) if assignments else None
            is_preview_bot = (
                a is not None and a.task is not None
                and a.task.task_type == TaskType.PRE_PICK
            )
            for inv in bot.inventory:
                if inv not in need:
                    continue
                if is_preview_bot:
                    have[inv] += 1
                elif inv not in active_types:
                    have[inv] += 1
        return (have, need)

    def _preview_ready_in_inventories(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment] | None = None,
    ) -> bool:
        """True if all preview-order items are already in bot inventories."""
        have, need = self._count_preview_items_in_inventories(world, assignments)
        if not need:
            return True
        for t, c in need.items():
            if have.get(t, 0) < c:
                return False
        return True

    def maintain(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        *,
        skip_route_abort: bool = False,
        skip_time_check: bool = False,
    ) -> dict[int, BotAssignment]:
        """
        Task lifecycle management only: advance routes, invalidate stale tasks,
        expire blacklists, update inventory snapshots. No new assignments.
        Used by ReplayPlanner to keep task state healthy without overriding plan.
        """
        state = world.state
        self._advance_routes(world, assignments, skip_route_abort=skip_route_abort)
        self._invalidate_stale(world, assignments, skip_time_check=skip_time_check)
        expired = [iid for iid, exp_round in self._blacklisted_items.items()
                   if state.round >= exp_round]
        for iid in expired:
            del self._blacklisted_items[iid]
        for bot in state.bots:
            self._prev_inventory[bot.id] = bot.inventory
        return assignments

    def plan(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> dict[int, BotAssignment]:
        """
        Main planning entry point. Mutates and returns assignments dict.
        Called once per round by the Coordinator.
        """
        state = world.state

        # Step 0: Advance routes (before invalidation)
        self._advance_routes(world, assignments)

        # Step 1: Invalidate stale tasks
        self._invalidate_stale(world, assignments)

        # Step 1.5: Expire old blacklist entries
        expired = [iid for iid, exp_round in self._blacklisted_items.items()
                   if state.round >= exp_round]
        for iid in expired:
            del self._blacklisted_items[iid]

        # Step 2: Track what's already claimed (PICK_UP + PRE_PICK + route items + blacklisted)
        claimed_items: set[str] = set(self._blacklisted_items.keys())
        for a in assignments.values():
            # Claim all items in active routes
            if a.route:
                for stop in a.route.stops[a.route_step:]:
                    claimed_items.add(stop.item_id)
            elif (
                a.task
                and a.task.task_type in (TaskType.PICK_UP, TaskType.PRE_PICK)
                and a.task.item_id
            ):
                claimed_items.add(a.task.item_id)

        # Completion gate: don't rush last active item until preview is ready (auto-delivery pipeline)
        gate_min_rounds = (
            getattr(self._config, "gate_min_rounds_remaining", 40)
            if getattr(self, "_config", None) else 40
        )
        gate_max_delay = (
            getattr(self._config, "gate_max_delay", 6)
            if getattr(self, "_config", None) else 6
        )
        active_list = state.active_orders
        preview_ready = self._preview_ready_in_inventories(world, assignments)
        active_remaining = len(active_list[0].items_remaining) if active_list else 0
        gate_closed = False
        if active_list and state.preview_orders and len(state.bots) >= 2:
            if self._gate_order_id != active_list[0].id:
                self._gate_rounds_held = 0
                self._gate_order_id = active_list[0].id
            preview_items_needed = sum(self._count_preview_items_in_inventories(world, assignments)[1].values())
            total_capacity = len(state.bots) * 3
            if preview_items_needed <= total_capacity and world.rounds_remaining > gate_min_rounds:
                gate_closed = (
                    active_remaining == 1
                    and not preview_ready
                    and self._gate_rounds_held < gate_max_delay
                )
                if gate_closed:
                    self._gate_rounds_held += 1
                    logger.debug("Completion gate closed: preview not ready, holding active (rounds_held=%d)",
                                 self._gate_rounds_held)
                elif active_remaining <= 2 and preview_ready:
                    self._gate_rounds_held = 0

        # RUSH: active order has 1 item remaining -> nearest IDLE bot takes it (skip when gate closed)
        if not gate_closed and state.active_orders and len(state.bots) >= 2:
            active = state.active_orders[0]
            remaining = active.items_remaining
            if len(remaining) == 1:
                needed_type = remaining[0]
                best_bid, best_dist, best_item, best_pp = None, 9999, None, None
                for bot in state.bots:
                    if len(bot.inventory) >= 3:
                        continue
                    # Don't steal bots that are delivering or already assigned
                    a = assignments.get(bot.id)
                    if a and a.task and a.task.task_type in (TaskType.DELIVER, TaskType.PICK_UP):
                        continue
                    for item in world.items_of_type(needed_type):
                        if item.id in claimed_items:
                            continue
                        pp = world.best_pickup_position(bot.position, item.position)
                        if pp and (d := world.distance(bot.position, pp)) < best_dist:
                            best_bid, best_dist, best_item, best_pp = bot.id, d, item, pp
                if best_bid is not None:
                    a = assignments[best_bid]
                    a.clear()
                    a.task = Task(
                        task_type=TaskType.PICK_UP, target_pos=best_pp,
                        item_id=best_item.id, item_type=best_item.type,
                        item_pos=best_item.position, order_id=active.id,
                    )
                    a.path = None
                    claimed_items.add(best_item.id)
                    logger.debug("Bot %d RUSH: last item %s for order %s",
                                 best_bid, best_item.type, active.id)

        # Endgame check: if active order can't be completed, switch strategy
        if world.is_endgame() and not world.can_complete_active_order():
            self._plan_endgame(world, assignments, claimed_items)
            # Save inventory snapshots (normally done at end of plan())
            for bot in state.bots:
                self._prev_inventory[bot.id] = bot.inventory
            return assignments

        # Step 3: Rush preview holders when active order is almost done
        rounds_to_complete = self._estimate_rounds_to_order_completion(world, assignments)
        self._rush_preview_holders(world, assignments, rounds_to_complete)

        # Step 4: Assign tasks to unassigned bots
        unassigned = sorted(
            bot_id for bot_id, a in assignments.items() if not a.has_task
        )

        # Pipeline: reserve 1 bot for preview when active has <=3 items and >=2 on active
        unassigned = self._reserve_one_bot_for_preview(
            world, state, unassigned, assignments, claimed_items
        )

        # Phase A: Active order picking via Hungarian (or greedy fallback)
        self._assign_active_tasks(
            world, state, unassigned, assignments, claimed_items
        )

        # Rebuild unassigned after active assignment
        unassigned = sorted(
            bot_id for bot_id, a in assignments.items() if not a.has_task
        )

        # Phase B: Preview pre-picking for idle bots
        self._assign_preview_tasks(world, state, unassigned, assignments, claimed_items)

        # Phase C: Remaining bots get fallback tasks
        unassigned = sorted(
            bot_id for bot_id, a in assignments.items() if not a.has_task
        )
        for bot_id in unassigned:
            bot = state.get_bot(bot_id)
            if bot is None:
                continue
            task = self._find_fallback_task(world, bot, claimed_items, assignments)
            if task:
                assignments[bot_id].task = task
                assignments[bot_id].path = None
                if task.item_id:
                    claimed_items.add(task.item_id)


        # Completion gate enforcement: if gate is closed, prevent order-completing deliveries
        if gate_closed and active_list:
            active_order = active_list[0]
            remaining_types = Counter(active_order.items_remaining)
            for bot_id, a in assignments.items():
                if not (a.task and a.task.task_type == TaskType.DELIVER):
                    continue
                bot = state.get_bot(bot_id)
                if bot is None:
                    continue
                test_remaining = dict(remaining_types)
                for inv in bot.inventory:
                    if inv in test_remaining and test_remaining[inv] > 0:
                        test_remaining[inv] -= 1
                would_complete = all(v <= 0 for v in test_remaining.values())
                if would_complete:
                    staging = self._nearest_dropoff_adjacent(bot, world)
                    a.task = Task(
                        task_type=TaskType.IDLE,
                        target_pos=staging or bot.position,
                    )
                    a.path = None
                    logger.debug("Gate: Bot %d held back from completing order (preview not ready)", bot_id)

        # Overflow deliverers: send to pick instead of staging (active waiting)
        self._reassign_overflow_deliverers_to_pick(world, assignments, claimed_items)

        # Save inventory snapshots for stuck detection
        for bot in state.bots:
            self._prev_inventory[bot.id] = bot.inventory

        return assignments

    def _reassign_overflow_deliverers_to_pick(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        claimed_items: set[str],
    ) -> None:
        """
        Bots that would be sent to staging (drop-off queue full) get a PICK_UP task
        instead so they stay productive instead of waiting.
        """
        state = world.state
        # Token-style: limit concurrent drop-off to reduce queue (2 slots balance throughput vs congestion)
        max_slots = min(2, max(len(world.dropoff_adjacent_positions()), 1))
        active = state.active_orders
        if not active:
            return
        remaining_types = set(active[0].items_remaining)
        need = Counter(active[0].items_remaining)

        deliverers: list[tuple[int, int, int, bool]] = []
        bots_by_id: dict[int, Bot] = {}
        for bot_id, assignment in assignments.items():
            if assignment.task and assignment.task.task_type == TaskType.DELIVER:
                bot = state.get_bot(bot_id)
                if bot:
                    has_match = any(inv in remaining_types for inv in bot.inventory)
                    if has_match:
                        d = world.distance(bot.position, world.nearest_drop_off(bot.position, bot_id))
                        matching_count = sum(1 for inv in bot.inventory if inv in remaining_types)
                        bots_by_id[bot_id] = bot
                        deliverers.append((d, bot_id, matching_count, False))

        if len(deliverers) <= max_slots:
            return

        have: Counter[str] = Counter()
        for bot_id, _, _, _ in deliverers:
            bot = bots_by_id.get(bot_id)
            if bot:
                for inv in bot.inventory:
                    if inv in need:
                        have[inv] += 1
        updated: list[tuple[int, int, int, bool]] = []
        for (d, bot_id, matching_count, _) in deliverers:
            bot = bots_by_id.get(bot_id)
            completes = False
            if bot and need:
                for inv in bot.inventory:
                    if inv in need and have[inv] - sum(1 for i in bot.inventory if i == inv) < need[inv]:
                        completes = True
                        break
            updated.append((d, bot_id, matching_count, completes))
        deliverers = updated
        deliverers.sort(key=lambda x: (not x[3], -x[2], x[0]))

        from bot.strategy.route_builder import build_routes
        preview_orders = state.preview_orders
        preview_types = set(preview_orders[0].items_remaining) if preview_orders else set()
        for i, (_, bot_id, _, _) in enumerate(deliverers):
            if i < max_slots:
                continue
            bot = state.get_bot(bot_id)
            if not bot or len(bot.inventory) >= 3:
                continue
            # Bot has preview items but is overflow deliverer — keep as DELIVER
            # so _schedule_dropoff can manage staging; items auto-deliver on transition
            if preview_types and any(inv in preview_types for inv in bot.inventory):
                continue
            assignment = assignments[bot_id]
            assignment.route = None
            assignment.route_step = 0
            assignment.path = None
            # Assign multi-item route or single next pickup
            routes = build_routes(bot, world, active[0], claimed_items)
            route_assigned = False
            for route in routes:
                if len(route.stops) >= 1 and route.total_cost <= world.rounds_remaining:
                    first_stop = route.stops[0]
                    assignment.task = Task(
                        task_type=TaskType.PICK_UP,
                        target_pos=first_stop.pickup_pos,
                        item_id=first_stop.item_id,
                        item_type=first_stop.item_type,
                        item_pos=first_stop.item_pos,
                        order_id=route.order_id,
                    )
                    if len(route.stops) > 1:
                        assignment.route = route
                        assignment.route_step = 0
                        for stop in route.stops:
                            claimed_items.add(stop.item_id)
                    else:
                        claimed_items.add(first_stop.item_id)
                    route_assigned = True
                    logger.debug("Bot %d overflow deliverer -> pick (%d stops)", bot_id, len(route.stops))
                    break
            if not route_assigned:
                next_task = self._find_next_pickup(bot, world, assignments, bot_id)
                if next_task:
                    assignment.task = next_task
                    if next_task.item_id:
                        claimed_items.add(next_task.item_id)
                    logger.debug("Bot %d overflow deliverer -> single pick", bot_id)
                # else leave as DELIVER; coordinator will send to staging

    def _item_picked_up(self, bot_id: int, item_type: str, bot_inventory: tuple) -> bool:
        """Check if bot's inventory gained an item of the expected type since last round."""
        prev_inv = self._prev_inventory.get(bot_id, ())
        prev_count = Counter(prev_inv).get(item_type, 0)
        curr_count = Counter(bot_inventory).get(item_type, 0)
        return curr_count > prev_count

    def _estimate_rounds_to_order_completion(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> int:
        """
        Estimate rounds until the active order is fully delivered.
        Based on: bots with matching inventory (distance to drop-off), bots on
        active route (remaining route work + distance to drop-off). Used for
        pipeline coordination (preview reserve, auto-delivery rush).
        """
        active_list = world.state.active_orders
        if not active_list:
            return 0
        order = active_list[0]
        remaining_types = set(order.items_remaining)
        if not remaining_types:
            return 0

        drop_off = world.state.drop_off  # Used as fallback; per-bot uses nearest_drop_off
        rounds_per_item = (
            getattr(self._config, "rounds_per_item_estimate", 6)
            if getattr(self, "_config", None)
            else 6
        )
        etas: list[int] = []

        for bot_id, bot in [(b.id, b) for b in world.state.bots]:
            if not bot.inventory and (bot_id not in assignments or not assignments[bot_id].route):
                continue
            # Bot with matching inventory: rounds to reach drop-off
            if bot.inventory and any(inv in remaining_types for inv in bot.inventory):
                etas.append(world.distance(bot.position, world.nearest_drop_off(bot.position, bot_id)))
            # Bot on route for this order: rough rounds to finish route + deliver
            a = assignments.get(bot_id)
            if a and a.route and a.route.order_id == order.id:
                remaining_stops = len(a.route.stops) - a.route_step
                route_work = remaining_stops * rounds_per_item
                etas.append(route_work + world.distance(bot.position, drop_off))

        return max(etas, default=0)

    def _advance_routes(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        skip_route_abort: bool = False,
    ) -> None:
        """Advance route progress for bots that have picked up their current stop."""
        state = world.state
        current_item_ids = {item.id for item in state.items}

        for bot_id, assignment in assignments.items():
            if not assignment.route:
                continue

            bot = state.get_bot(bot_id)
            if bot is None:
                continue

            current_stop = assignment.current_route_stop
            if current_stop is None:
                # All stops completed -> switch to DELIVER
                bot = state.get_bot(bot_id)
                deliver_target = world.nearest_drop_off(bot.position, bot_id) if bot else state.drop_off
                assignment.task = Task(
                    task_type=TaskType.DELIVER,
                    target_pos=deliver_target,
                )
                assignment.route = None
                assignment.route_step = 0
                assignment.path = None
                continue

            if current_stop.item_id.startswith("camp_"):
                picked_up = self._item_picked_up(
                    bot_id, current_stop.item_type, bot.inventory
                )
            else:
                picked_up = (
                    current_stop.item_id not in current_item_ids
                    or self._item_picked_up(bot_id, current_stop.item_type, bot.inventory)
                )

            if picked_up:
                assignment.route_step += 1
                next_stop = assignment.current_route_stop
                if next_stop is None:
                    # Last item picked -> DELIVER
                    bot = state.get_bot(bot_id)
                    deliver_target = world.nearest_drop_off(bot.position, bot_id) if bot else state.drop_off
                    assignment.task = Task(
                        task_type=TaskType.DELIVER,
                        target_pos=deliver_target,
                    )
                    assignment.route = None
                    assignment.route_step = 0
                    assignment.path = None
                else:
                    # Update task to next stop
                    assignment.task = Task(
                        task_type=TaskType.PICK_UP,
                        target_pos=next_stop.pickup_pos,
                        item_id=next_stop.item_id,
                        item_type=next_stop.item_type,
                        item_pos=next_stop.item_pos,
                        order_id=assignment.route.order_id,
                    )
                    assignment.path = None
                    logger.debug("Bot %d route advanced to step %d: %s",
                                 bot_id, assignment.route_step, next_stop.item_type)
