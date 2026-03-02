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

from bot.models import GameState
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
        # Track previous inventory per bot to detect stuck drop-offs and pick success
        self._prev_inventory: dict[int, tuple[str, ...]] = {}
        self._stuck_deliver_rounds: dict[int, int] = {}
        self._stuck_pick_rounds: dict[int, int] = {}
        # Blacklisted items: item_id -> round when blacklist expires
        self._blacklisted_items: dict[str, int] = {}

    def maintain(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> dict[int, BotAssignment]:
        """
        Task lifecycle management only: advance routes, invalidate stale tasks,
        expire blacklists, update inventory snapshots. No new assignments.
        Used by ReplayPlanner to keep task state healthy without overriding plan.
        """
        state = world.state
        self._advance_routes(world, assignments)
        self._invalidate_stale(world, assignments)
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

        # Endgame check: if active order can't be completed, switch strategy
        if world.is_endgame() and not world.can_complete_active_order():
            self._plan_endgame(world, assignments, claimed_items)
            # Save inventory snapshots (normally done at end of plan())
            for bot in state.bots:
                self._prev_inventory[bot.id] = bot.inventory
            return assignments

        # Step 3: Rush preview holders when active order is almost done
        # With 4+ bots, auto-delivery (position-independent) handles cascades well
        # and rushing wastes bot capacity. With fewer bots, rush helps coordination.
        if len(state.bots) < 4:
            self._rush_preview_holders(world, assignments)

        # Step 4: Assign tasks to unassigned bots
        unassigned = sorted(
            bot_id for bot_id, a in assignments.items() if not a.has_task
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
        # Compute active-order type coverage to prevent over-picking
        # (only with 4+ bots — with fewer bots, multi-item routes handle this)
        covered_types: set[str] = set()
        if state.active_orders and len(state.bots) >= 4:
            active_budget = Counter(state.active_orders[0].items_remaining)
            for bot_obj in state.bots:
                for inv_item in bot_obj.inventory:
                    if active_budget[inv_item] > 0:
                        active_budget[inv_item] -= 1
            for a in assignments.values():
                if a.task and a.task.task_type == TaskType.PICK_UP and a.task.item_type:
                    if active_budget[a.task.item_type] > 0:
                        active_budget[a.task.item_type] -= 1
                if a.route:
                    for stop in a.route.stops[a.route_step:]:
                        if active_budget[stop.item_type] > 0:
                            active_budget[stop.item_type] -= 1
            active_budget = +active_budget
            if state.active_orders:
                for t in state.active_orders[0].items_remaining:
                    if active_budget.get(t, 0) == 0:
                        covered_types.add(t)

        unassigned = sorted(
            bot_id for bot_id, a in assignments.items() if not a.has_task
        )
        for bot_id in unassigned:
            bot = state.get_bot(bot_id)
            if bot is None:
                continue
            task = self._find_fallback_task(
                world, bot, claimed_items, assignments, covered_types
            )
            if task:
                assignments[bot_id].task = task
                assignments[bot_id].path = None
                if task.item_id:
                    claimed_items.add(task.item_id)

        # Save inventory snapshots for stuck detection
        for bot in state.bots:
            self._prev_inventory[bot.id] = bot.inventory

        return assignments

    def _item_picked_up(self, bot_id: int, item_type: str, bot_inventory: tuple) -> bool:
        """Check if bot's inventory gained an item of the expected type since last round."""
        prev_inv = self._prev_inventory.get(bot_id, ())
        return bot_inventory.count(item_type) > prev_inv.count(item_type)

    def _advance_routes(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
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
                assignment.task = Task(
                    task_type=TaskType.DELIVER,
                    target_pos=state.drop_off,
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
                    assignment.task = Task(
                        task_type=TaskType.DELIVER,
                        target_pos=state.drop_off,
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
