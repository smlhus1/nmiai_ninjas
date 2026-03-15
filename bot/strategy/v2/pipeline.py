"""
PipelineManager: preview pre-picking and auto-delivery staging.

Manages idle bots (not needed for active order) by assigning them to
pre-pick preview order items and staging near drop-off for auto-delivery.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from bot.models import Bot, Order, Pos
from bot.engine.world_model import WorldModel
from bot.strategy.task import Task, TaskType, BotAssignment

logger = logging.getLogger(__name__)


class PipelineManager:
    """Manages preview pre-picking pipeline for idle bots."""

    def assign_idle_bots(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        idle_bot_ids: list[int],
        claimed_items: set[str],
        estimated_active_completion: int = 999,
        future_orders: list[dict] | None = None,
    ) -> None:
        """
        Assign idle bots to preview pre-picking or staging.

        Priority:
        1. Pre-pick preview items (bots not needed for active order)
        2. Pre-pick future order items (N+2 from recon, if available)
        3. Stage near drop-off for auto-delivery when order is almost done
        """
        state = world.state
        preview_orders = state.preview_orders
        if not idle_bot_ids:
            return
        if not preview_orders and not future_orders:
            return

        # Don't start new pre-picks in endgame — bots risk getting stuck
        # with full non-matching inventory and can't contribute
        if world.is_endgame():
            return

        preview = preview_orders[0] if preview_orders else None
        drop_off = state.drop_off  # Used as generic reference; per-bot uses nearest

        if preview:
            # Count what we already have for preview
            have, need = self._count_preview_items(world, assignments, preview)

            # Stage bots with preview items near drop-off when active is almost done
            self._stage_preview_holders(
                world, assignments, preview, drop_off,
                estimated_active_completion,
            )

            # Assign remaining idle bots to pre-pick preview items
            for bot_id in idle_bot_ids:
                if assignments[bot_id].has_task:
                    continue  # Already staged or has task

                bot = state.get_bot(bot_id)
                if bot is None or len(bot.inventory) >= 3:
                    continue

                task = self._find_preview_task(
                    world, bot, preview, claimed_items, have, need,
                )
                if task:
                    assignments[bot_id].task = task
                    assignments[bot_id].path = None
                    if task.item_id:
                        claimed_items.add(task.item_id)
                    if task.item_type:
                        have[task.item_type] = have.get(task.item_type, 0) + 1
                    logger.debug("Pipeline: Bot %d assigned preview pre-pick %s", bot_id, task.item_type)

        # Assign still-idle bots to future order items (N+2+)
        if future_orders:
            for future_dict in future_orders:
                future_items = future_dict.get("items_required", [])
                if not future_items:
                    continue

                future_have: Counter = Counter()
                future_need = Counter(future_items)

                for bot_id in idle_bot_ids:
                    if assignments[bot_id].has_task:
                        continue

                    bot = state.get_bot(bot_id)
                    if bot is None or len(bot.inventory) >= 3:
                        continue

                    task = self._find_future_task(
                        world, bot, future_items, claimed_items, future_have, future_need,
                    )
                    if task:
                        assignments[bot_id].task = task
                        assignments[bot_id].path = None
                        if task.item_id:
                            claimed_items.add(task.item_id)
                        if task.item_type:
                            future_have[task.item_type] += 1
                        logger.debug("Pipeline: Bot %d future pre-pick %s", bot_id, task.item_type)

        # Park remaining idle bots away from drop-off
        for bot_id in idle_bot_ids:
            if assignments[bot_id].has_task:
                continue
            bot = state.get_bot(bot_id)
            if bot is None:
                continue
            assignments[bot_id].task = Task(
                task_type=TaskType.IDLE,
                target_pos=self._idle_position(bot, world),
            )
            assignments[bot_id].path = None

    def _stage_preview_holders(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        preview: Order,
        drop_off: Pos,
        estimated_active_completion: int,
    ) -> None:
        """
        Move bots holding preview items toward drop-off when active order
        is close to completion (auto-delivery exploit).
        """
        state = world.state
        active = state.active_orders
        if not active:
            return

        remaining = active[0].items_remaining
        if len(remaining) > 4:
            return  # Too early to stage

        preview_types = set(preview.items_remaining)

        for bot_id, assignment in assignments.items():
            # Only stage PRE_PICK bots that hold preview items
            if not (assignment.task and assignment.task.task_type == TaskType.PRE_PICK):
                continue

            bot = state.get_bot(bot_id)
            if bot is None:
                continue
            if not any(inv in preview_types for inv in bot.inventory):
                continue

            nearest_do = world.nearest_drop_off(bot.position, bot_id)
            d_to_drop = world.distance(bot.position, nearest_do)
            if d_to_drop == 0:
                continue  # Already there

            # Stage when bot can arrive before/shortly after order completes
            if d_to_drop <= estimated_active_completion + 3:
                assignment.task = Task(
                    task_type=TaskType.DELIVER,
                    target_pos=nearest_do,
                )
                assignment.path = None
                logger.debug("Pipeline: Bot %d staging for auto-delivery (d=%d, eta=%d)",
                             bot_id, d_to_drop, estimated_active_completion)

    def _find_preview_task(
        self,
        world: WorldModel,
        bot: Bot,
        preview: Order,
        claimed_items: set[str],
        have: Counter,
        need: Counter,
        max_distance: int = 9999,
    ) -> Optional[Task]:
        """Find nearest preview-order item to pre-pick, respecting budget and max distance."""
        # Compute remaining budget
        budget: Counter[str] = Counter()
        for t, c in need.items():
            budget[t] = max(0, c - have.get(t, 0))

        best_task: Optional[Task] = None
        best_dist = 9999

        for item_type in preview.items_remaining:
            if budget.get(item_type, 0) <= 0:
                continue
            for item in world.items_of_type(item_type):
                if item.id in claimed_items:
                    continue
                pickup_pos = world.best_pickup_position(bot.position, item.position)
                if pickup_pos is None:
                    continue
                d = world.distance(bot.position, pickup_pos)
                if d > max_distance:
                    continue  # Too far — don't cross the map for preview
                if d < best_dist:
                    best_dist = d
                    best_task = Task(
                        task_type=TaskType.PRE_PICK,
                        target_pos=pickup_pos,
                        item_id=item.id,
                        item_type=item.type,
                        item_pos=item.position,
                        order_id=preview.id,
                    )

        return best_task

    def _find_future_task(
        self,
        world: WorldModel,
        bot: Bot,
        future_items: list[str],
        claimed_items: set[str],
        have: Counter,
        need: Counter,
    ) -> Optional[Task]:
        """Find nearest item matching a future order to pre-pick."""
        budget: Counter[str] = Counter()
        for t, c in need.items():
            budget[t] = max(0, c - have.get(t, 0))

        best_task: Optional[Task] = None
        best_dist = 9999

        for item_type in set(future_items):
            if budget.get(item_type, 0) <= 0:
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
                    )

        return best_task

    @staticmethod
    def _count_preview_items(
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        preview: Order,
    ) -> tuple[Counter, Counter]:
        """Count preview items already held by bots vs what's needed."""
        need = Counter(preview.items_remaining)
        active = world.state.active_orders
        active_types = set(active[0].items_remaining) if active else set()

        have: Counter[str] = Counter()
        for bot in world.state.bots:
            a = assignments.get(bot.id)
            is_preview_bot = (
                a is not None and a.task is not None
                and a.task.task_type == TaskType.PRE_PICK
            )
            for inv in bot.inventory:
                if inv not in need:
                    continue
                if is_preview_bot or inv not in active_types:
                    have[inv] += 1

        return have, need

    @staticmethod
    def _idle_position(bot: Bot, world: WorldModel) -> Pos:
        """Park bot away from drop-off."""
        drop_off = world.nearest_drop_off(bot.position, bot.id)
        if world.distance(bot.position, drop_off) <= 2:
            staging = world.staging_positions()
            if staging:
                d_current = world.distance(bot.position, drop_off)
                farther = [p for p in staging if world.distance(p, drop_off) > d_current]
                if farther:
                    return min(farther, key=lambda p: world.distance(bot.position, p))
                return min(staging, key=lambda p: world.distance(bot.position, p))
        return bot.position
