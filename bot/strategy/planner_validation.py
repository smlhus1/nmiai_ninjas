"""
Task validation and stuck detection for TaskPlanner.

Handles invalidation of stale tasks: items gone, orders completed,
stuck bots, full inventory, endgame parking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bot.engine.world_model import WorldModel
from bot.strategy.task import Task, TaskType, BotAssignment

if TYPE_CHECKING:
    from bot.strategy.planner import TaskPlanner

logger = logging.getLogger(__name__)


class ValidationMixin:

    def _invalidate_stale(
        self: TaskPlanner,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> None:
        """Remove tasks that are no longer valid."""
        state = world.state
        current_item_ids = {item.id for item in state.items}
        # Active/preview order IDs for staleness checks
        active_order_ids = {o.id for o in state.active_orders}
        preview_order_ids = {o.id for o in state.preview_orders}
        known_order_ids = active_order_ids | preview_order_ids
        # Item types needed by active orders
        active_remaining_types: set[str] = set()
        for o in state.active_orders:
            active_remaining_types.update(o.items_remaining)

        for bot_id, assignment in assignments.items():
            task = assignment.task
            if task is None:
                continue

            bot = state.get_bot(bot_id)
            if bot is None:
                assignment.clear()
                continue

            if assignment.route:
                # Route for a completed order — clear it
                route_oid = assignment.route.order_id
                if route_oid and route_oid not in known_order_ids:
                    logger.debug("Bot %d: route for completed order %s, clearing", bot_id, route_oid)
                    if bot.inventory and self._has_matching_items(bot, world):
                        assignment.task = Task(
                            task_type=TaskType.DELIVER,
                            target_pos=state.drop_off,
                        )
                    else:
                        assignment.task = None
                    assignment.route = None
                    assignment.route_step = 0
                    assignment.path = None
                    if assignment.task is None:
                        assignment.clear()
                    continue
                remaining_stops = assignment.route.stops[assignment.route_step:]
                remaining_stops = [
                    s for s in remaining_stops
                    if s.item_id in current_item_ids or s.item_id.startswith("camp_")
                ]
                if not remaining_stops:
                    # All remaining route items gone -> clear route, let delivery or reassignment happen
                    if bot.inventory:
                        assignment.task = Task(
                            task_type=TaskType.DELIVER,
                            target_pos=state.drop_off,
                        )
                        assignment.route = None
                        assignment.route_step = 0
                        assignment.path = None
                    else:
                        assignment.clear()
                    continue
                else:
                    # Rebuild route with surviving stops
                    original_count = len(assignment.route.stops) - assignment.route_step
                    if len(remaining_stops) < original_count:
                        assignment.route.stops = (
                            assignment.route.stops[:assignment.route_step] + remaining_stops
                        )
                        # If current stop changed, update task
                        current_stop = assignment.current_route_stop
                        if current_stop and task.item_id != current_stop.item_id:
                            assignment.task = Task(
                                task_type=TaskType.PICK_UP,
                                target_pos=current_stop.pickup_pos,
                                item_id=current_stop.item_id,
                                item_type=current_stop.item_type,
                                item_pos=current_stop.item_pos,
                                order_id=assignment.route.order_id,
                            )
                            assignment.path = None

            if task.task_type in (TaskType.PICK_UP, TaskType.PRE_PICK):
                # Detect successful pick via inventory change (handles infinite shelves)
                if not assignment.route and task.item_type:
                    if self._item_picked_up(bot_id, task.item_type, bot.inventory):
                        logger.debug("Bot %d: picked %s (inventory change), clearing task",
                                     bot_id, task.item_type)
                        assignment.clear()
                        self._stuck_pick_rounds.pop(bot_id, None)
                        continue

                # Full inventory — can't pick up
                if len(bot.inventory) >= 3:
                    self._stuck_pick_rounds.pop(bot_id, None)
                    if self._has_matching_items(bot, world) or world.is_endgame():
                        assignment.task = Task(
                            task_type=TaskType.DELIVER,
                            target_pos=state.drop_off,
                        )
                    elif self._has_matching_preview_items(bot, world):
                        assignment.task = Task(
                            task_type=TaskType.IDLE,
                            target_pos=bot.position,
                        )
                    else:
                        # Nothing matches — don't waste a round moving to drop-off
                        assignment.task = Task(
                            task_type=TaskType.IDLE,
                            target_pos=bot.position,
                        )
                    assignment.route = None
                    assignment.route_step = 0
                    assignment.path = None
                    continue

                # Stuck picking: bot at pickup position but inventory unchanged
                prev_inv = self._prev_inventory.get(bot_id, ())
                if bot.inventory == prev_inv and task.target_pos:
                    d_to_target = world.distance(bot.position, task.target_pos)
                    if d_to_target <= 1:  # At or adjacent to target
                        stuck = self._stuck_pick_rounds.get(bot_id, 0) + 1
                        self._stuck_pick_rounds[bot_id] = stuck
                        if stuck >= 5:
                            logger.info("Bot %d: stuck picking %s for %d rounds, blacklisting",
                                        bot_id, task.item_id, stuck)
                            # Blacklist this item so it won't be reassigned immediately
                            if task.item_id:
                                self._blacklisted_items[task.item_id] = state.round + 5
                            assignment.clear()
                            self._stuck_pick_rounds.pop(bot_id, None)
                            continue
                    else:
                        self._stuck_pick_rounds.pop(bot_id, None)
                else:
                    self._stuck_pick_rounds.pop(bot_id, None)

            if task.task_type == TaskType.PICK_UP:
                # Clear if picking for a completed order
                if task.order_id and task.order_id not in known_order_ids:
                    logger.debug("Bot %d: PICK_UP for completed order %s, clearing",
                                 bot_id, task.order_id)
                    assignment.clear()
                # Clear if item type doesn't match active order (stale/junk pickup)
                elif task.item_type and active_remaining_types and task.item_type not in active_remaining_types:
                    # Allow if it matches preview order (may auto-deliver on transition)
                    preview_types: set[str] = set()
                    for o in state.preview_orders:
                        preview_types.update(o.items_remaining)
                    if task.item_type not in preview_types:
                        logger.debug("Bot %d: PICK_UP for %s not needed by any order, clearing",
                                     bot_id, task.item_type)
                        assignment.clear()
                elif (task.item_id
                        and task.item_id not in current_item_ids
                        and not task.item_id.startswith("camp_")):
                    assignment.clear()
                elif task.item_pos and not world.can_complete_trip(bot, task.item_pos):
                    assignment.clear()

            elif task.task_type == TaskType.PRE_PICK:
                # Item gone
                if task.item_id and task.item_id not in current_item_ids:
                    assignment.clear()
                # Preview order became active (order transition happened)
                elif task.order_id and task.order_id in active_order_ids:
                    assignment.clear()

            elif task.task_type == TaskType.DELIVER:
                if not bot.inventory:
                    assignment.clear()
                    self._stuck_deliver_rounds.pop(bot_id, None)
                elif not self._has_matching_items(bot, world):
                    # No items match active order — clear immediately to avoid blocking drop-off
                    # (applies in endgame too: non-matching items can't be delivered)
                    assignment.clear()
                    self._stuck_deliver_rounds.pop(bot_id, None)
                elif bot.position == state.drop_off:
                    prev_inv = self._prev_inventory.get(bot_id)
                    if prev_inv == bot.inventory:
                        rounds_stuck = self._stuck_deliver_rounds.get(bot_id, 0) + 1
                        self._stuck_deliver_rounds[bot_id] = rounds_stuck
                        if rounds_stuck >= 2:
                            assignment.clear()
                            self._stuck_deliver_rounds.pop(bot_id, None)
                    else:
                        self._stuck_deliver_rounds.pop(bot_id, None)
