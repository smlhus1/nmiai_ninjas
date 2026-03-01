"""
Task assignment logic for TaskPlanner.

Methods for assigning active-order tasks (via Hungarian or greedy fallback),
preview pre-picking, and rushing preview holders to drop-off.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from bot.models import GameState, Bot, Order
from bot.engine.world_model import WorldModel
from bot.strategy.task import Task, TaskType, BotAssignment

if TYPE_CHECKING:
    from bot.strategy.planner import TaskPlanner

logger = logging.getLogger(__name__)


class AssignmentMixin:

    def _assign_active_tasks(
        self: TaskPlanner,
        world: WorldModel,
        state: GameState,
        unassigned: list[int],
        assignments: dict[int, BotAssignment],
        claimed_items: set[str],
    ) -> None:
        """Assign bots to active order items using Hungarian if available, else greedy."""
        try:
            from bot.strategy.hungarian import solve_assignment

            # Collect preview item IDs for cost matrix
            preview_item_ids: set[str] = set()
            for order in state.preview_orders:
                for item_type in order.items_remaining:
                    for item in world.items_of_type(item_type):
                        preview_item_ids.add(item.id)

            bots = [state.get_bot(bid) for bid in unassigned]
            bots = [b for b in bots if b is not None]

            if bots:
                result = solve_assignment(
                    bots, world, assignments, claimed_items, preview_item_ids
                )
                for bot_id, (task, route) in result.items():
                    assignments[bot_id].task = task
                    assignments[bot_id].route = route
                    assignments[bot_id].route_step = 0
                    assignments[bot_id].path = None
                    # Claim ALL items in the route, not just the first
                    if route:
                        claimed_items.update(route.item_ids)
                    elif task.item_id:
                        claimed_items.add(task.item_id)
                    logger.debug("Bot %d assigned (hungarian): %s route=%s", bot_id, task,
                                 f"{len(route.stops)}-stop" if route else "none")
        except ImportError:
            # scipy not installed — fall back to greedy
            logger.debug("scipy not available, using greedy assignment")
            for bot_id in unassigned:
                bot = state.get_bot(bot_id)
                if bot is None:
                    continue
                task = self._find_best_task(world, bot, claimed_items, assignments)
                if task:
                    assignments[bot_id].task = task
                    assignments[bot_id].path = None
                    if task.item_id:
                        claimed_items.add(task.item_id)
                    logger.debug("Bot %d assigned (greedy): %s", bot_id, task)

    def _assign_preview_tasks(
        self: TaskPlanner,
        world: WorldModel,
        state: GameState,
        unassigned: list[int],
        assignments: dict[int, BotAssignment],
        claimed_items: set[str],
    ) -> None:
        """Assign idle bots to pre-pick items for preview orders."""
        preview_orders = state.preview_orders
        if not preview_orders:
            return

        preview = preview_orders[0]

        for bot_id in unassigned:
            bot = state.get_bot(bot_id)
            if bot is None:
                continue

            # Don't pre-pick if inventory is full (leave 1 slot open)
            if len(bot.inventory) >= 2:
                continue

            task = self._find_preview_task(world, bot, preview, claimed_items)
            if task:
                assignments[bot_id].task = task
                assignments[bot_id].path = None
                if task.item_id:
                    claimed_items.add(task.item_id)
                logger.debug("Bot %d assigned preview: %s", bot_id, task)

    def _find_preview_task(
        self: TaskPlanner,
        world: WorldModel,
        bot: Bot,
        preview_order: Order,
        claimed_items: set[str],
    ) -> Optional[Task]:
        """Find nearest preview-order item to pre-pick."""
        from collections import Counter

        best_task: Optional[Task] = None
        best_dist = 9999

        preview_budget = Counter(preview_order.items_remaining)
        # Subtract only this bot's own inventory
        for inv_item in bot.inventory:
            if preview_budget[inv_item] > 0:
                preview_budget[inv_item] -= 1

        for item_type in preview_order.items_remaining:
            # Skip types we already have enough of
            if preview_budget.get(item_type, 0) <= 0:
                continue
            for item in world.items_of_type(item_type):
                if item.id in claimed_items:
                    continue

                pickup_pos = world.best_pickup_position(bot.position, item.position)
                if pickup_pos is None:
                    continue

                d = world.distance(bot.position, pickup_pos)
                if d < best_dist:
                    best_dist = d
                    best_task = Task(
                        task_type=TaskType.PRE_PICK,
                        target_pos=pickup_pos,
                        item_id=item.id,
                        item_type=item.type,
                        item_pos=item.position,
                        order_id=preview_order.id,
                    )

        return best_task

    def _rush_preview_holders(
        self: TaskPlanner,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> None:
        """
        When active order is almost done (<=2 items remaining), send bots
        holding preview items to drop-off for auto-delivery on order transition.
        """
        active = world.state.active_orders
        if not active:
            return

        remaining = active[0].items_remaining
        if len(remaining) > 2:
            return

        preview_orders = world.state.preview_orders
        if not preview_orders:
            return

        preview_types = set(preview_orders[0].items_remaining)

        for bot_id, assignment in assignments.items():
            if assignment.task and assignment.task.task_type == TaskType.PRE_PICK:
                bot = world.state.get_bot(bot_id)
                if bot is None:
                    continue
                # If bot is carrying items that match preview order, rush to drop-off
                if any(inv in preview_types for inv in bot.inventory):
                    assignment.task = Task(
                        task_type=TaskType.DELIVER,
                        target_pos=world.state.drop_off,
                    )
                    assignment.path = None
                    logger.debug("Bot %d rushing preview items to drop-off", bot_id)
