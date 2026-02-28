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
from typing import Optional

from bot.models import GameState, Bot, Item, Order, OrderStatus
from bot.engine.world_model import WorldModel
from bot.strategy.task import Task, TaskType, BotAssignment

logger = logging.getLogger(__name__)


class TaskPlanner:
    """
    Stateless planner — takes world model + current assignments,
    returns updated assignments.
    """

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

        # Step 1: Invalidate stale tasks
        self._invalidate_stale(world, assignments)

        # Step 2: Track what's already claimed (PICK_UP + PRE_PICK)
        claimed_items: set[str] = set()
        for a in assignments.values():
            if (
                a.task
                and a.task.task_type in (TaskType.PICK_UP, TaskType.PRE_PICK)
                and a.task.item_id
            ):
                claimed_items.add(a.task.item_id)

        # Endgame check: if active order can't be completed, switch strategy
        if world.is_endgame() and not world.can_complete_active_order():
            self._plan_endgame(world, assignments, claimed_items)
            return assignments

        # Step 3: Rush preview holders when active order is almost done
        self._rush_preview_holders(world, assignments)

        # Step 4: Assign tasks to unassigned bots
        unassigned = sorted(
            bot_id for bot_id, a in assignments.items() if not a.has_task
        )

        # Phase A: Active order picking via Hungarian (or greedy fallback)
        active_tasks = self._assign_active_tasks(
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
            task = self._find_fallback_task(world, bot, claimed_items)
            if task:
                assignments[bot_id].task = task
                assignments[bot_id].path = None
                if task.item_id:
                    claimed_items.add(task.item_id)

        return assignments

    def _assign_active_tasks(
        self,
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
                for bot_id, task in result.items():
                    assignments[bot_id].task = task
                    assignments[bot_id].path = None
                    if task.item_id:
                        claimed_items.add(task.item_id)
                    logger.debug("Bot %d assigned (hungarian): %s", bot_id, task)
        except ImportError:
            # scipy not installed — fall back to greedy
            logger.debug("scipy not available, using greedy assignment")
            for bot_id in unassigned:
                bot = state.get_bot(bot_id)
                if bot is None:
                    continue
                task = self._find_best_task(world, bot, claimed_items)
                if task:
                    assignments[bot_id].task = task
                    assignments[bot_id].path = None
                    if task.item_id:
                        claimed_items.add(task.item_id)
                    logger.debug("Bot %d assigned (greedy): %s", bot_id, task)

    def _assign_preview_tasks(
        self,
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

            # Don't pre-pick if inventory already has 2+ items (leave 1 slot open)
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
        self,
        world: WorldModel,
        bot: Bot,
        preview_order: Order,
        claimed_items: set[str],
    ) -> Optional[Task]:
        """Find nearest preview-order item to pre-pick."""
        best_task: Optional[Task] = None
        best_dist = 9999

        for item_type in preview_order.items_remaining:
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
        self,
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

    def _plan_endgame(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
        claimed_items: set[str],
    ) -> None:
        """
        Endgame strategy: maximize items delivered regardless of order completion.
        Bots with inventory -> deliver immediately, bots without -> pick nearest item.
        """
        state = world.state

        for bot_id, assignment in assignments.items():
            if assignment.has_task:
                continue  # Keep existing valid tasks

            bot = state.get_bot(bot_id)
            if bot is None:
                continue

            # Bots with inventory: deliver immediately (no batching)
            if bot.inventory:
                assignment.task = Task(
                    task_type=TaskType.DELIVER,
                    target_pos=state.drop_off,
                )
                assignment.path = None
                logger.debug("Bot %d endgame: delivering inventory", bot_id)
                continue

            # Bots without inventory: pick nearest deliverable item
            best_task: Optional[Task] = None
            best_dist = 9999

            for item in state.items:
                if item.id in claimed_items:
                    continue

                pickup_pos = world.best_pickup_position(bot.position, item.position)
                if pickup_pos is None:
                    continue

                d_pick = world.distance(bot.position, pickup_pos)
                d_drop = world.distance(pickup_pos, state.drop_off)
                # Must be able to pick + deliver in remaining rounds
                if d_pick + d_drop + 2 > world.rounds_remaining:
                    continue

                if d_pick < best_dist:
                    best_dist = d_pick
                    best_task = Task(
                        task_type=TaskType.PICK_UP,
                        target_pos=pickup_pos,
                        item_id=item.id,
                        item_type=item.type,
                        item_pos=item.position,
                    )

            if best_task:
                assignment.task = best_task
                assignment.path = None
                if best_task.item_id:
                    claimed_items.add(best_task.item_id)
                logger.debug("Bot %d endgame: picking nearest %s", bot_id, best_task.item_type)
            else:
                assignment.task = Task(task_type=TaskType.IDLE, target_pos=bot.position)

    def _invalidate_stale(
        self,
        world: WorldModel,
        assignments: dict[int, BotAssignment],
    ) -> None:
        """Remove tasks that are no longer valid."""
        state = world.state
        current_item_ids = {item.id for item in state.items}
        # Active order IDs for PRE_PICK invalidation on order transition
        active_order_ids = {o.id for o in state.active_orders}

        for bot_id, assignment in assignments.items():
            task = assignment.task
            if task is None:
                continue

            bot = state.get_bot(bot_id)
            if bot is None:
                assignment.clear()
                continue

            if task.task_type == TaskType.PICK_UP:
                # Item no longer exists (someone else grabbed it)
                if task.item_id and task.item_id not in current_item_ids:
                    logger.debug("Bot %d: item %s gone, clearing task", bot_id, task.item_id)
                    assignment.clear()

                # Can't complete trip in remaining rounds
                elif task.item_pos and not world.can_complete_trip(bot, task.item_pos):
                    logger.debug("Bot %d: not enough rounds for trip, clearing", bot_id)
                    assignment.clear()

            elif task.task_type == TaskType.PRE_PICK:
                # Item gone
                if task.item_id and task.item_id not in current_item_ids:
                    assignment.clear()
                # Preview order became active (order transition happened)
                elif task.order_id and task.order_id in active_order_ids:
                    assignment.clear()

            elif task.task_type == TaskType.DELIVER:
                # Nothing to deliver
                if not bot.inventory:
                    assignment.clear()
                # Endgame: always deliver whatever we have
                elif world.is_endgame():
                    pass  # Keep delivering
                # Normal: only deliver if matching active order
                elif not self._has_matching_items(bot, world):
                    assignment.clear()

    def _find_best_task(
        self,
        world: WorldModel,
        bot: Bot,
        claimed_items: set[str],
    ) -> Optional[Task]:
        """
        Find the best task for an unassigned bot (greedy fallback).
        Priority:
        1. If carrying items -> DELIVER
        2. Pick up item for highest-value order
        3. Pick up any unclaimed item that can be delivered in time
        4. IDLE
        """
        state = world.state

        # Priority 1: Deliver if we should
        if self._should_deliver(bot, world, claimed_items):
            return Task(
                task_type=TaskType.DELIVER,
                target_pos=state.drop_off,
            )

        # Priority 2: Find best item for best order
        best_task: Optional[Task] = None
        best_score: float = -1.0

        # Score all orders
        scored_orders = [
            (world.order_value(order), order)
            for order in state.orders
            if not order.complete
        ]
        scored_orders.sort(key=lambda x: x[0], reverse=True)

        for _, order in scored_orders:
            for item_type in order.items_remaining:
                for item in world.items_of_type(item_type):
                    if item.id in claimed_items:
                        continue

                    if not world.can_complete_trip(bot, item.position):
                        continue

                    pickup_pos = world.best_pickup_position(bot.position, item.position)
                    if pickup_pos is None:
                        continue

                    d = world.distance(bot.position, pickup_pos)
                    d_drop = world.distance(pickup_pos, state.drop_off)

                    # Score: order value / total trip cost
                    trip_cost = d + d_drop + 2  # +2 for pick_up + drop_off
                    score = world.order_value(order) / max(trip_cost, 1)

                    if score > best_score:
                        best_score = score
                        best_task = Task(
                            task_type=TaskType.PICK_UP,
                            target_pos=pickup_pos,
                            item_id=item.id,
                            item_type=item.type,
                            item_pos=item.position,
                            order_id=order.id,
                        )

        if best_task:
            return best_task

        return None  # No active-order task found

    def _find_fallback_task(
        self,
        world: WorldModel,
        bot: Bot,
        claimed_items: set[str],
    ) -> Optional[Task]:
        """Pick up ANY unclaimed item (for the +1 point) or IDLE."""
        state = world.state

        # Deliver if carrying anything
        if self._should_deliver(bot, world, claimed_items):
            return Task(
                task_type=TaskType.DELIVER,
                target_pos=state.drop_off,
            )

        for item in state.items:
            if item.id in claimed_items:
                continue
            if not world.can_complete_trip(bot, item.position):
                continue
            pickup_pos = world.best_pickup_position(bot.position, item.position)
            if pickup_pos is None:
                continue
            return Task(
                task_type=TaskType.PICK_UP,
                target_pos=pickup_pos,
                item_id=item.id,
                item_type=item.type,
                item_pos=item.position,
            )

        return Task(task_type=TaskType.IDLE, target_pos=bot.position)

    def _has_matching_items(self, bot: Bot, world: WorldModel) -> bool:
        """Check if bot has ANY inventory items matching the active order."""
        active = world.state.active_orders
        if not active:
            return False
        remaining = list(active[0].items_remaining)
        for inv_item in bot.inventory:
            if inv_item in remaining:
                return True
        return False

    def _should_deliver(
        self,
        bot: Bot,
        world: WorldModel,
        claimed_items: set[str],
    ) -> bool:
        """Decide if bot should deliver now vs pick more items."""
        if not bot.inventory:
            return False

        # Endgame: always deliver whatever we have
        if world.is_endgame():
            return True

        # Check if bot only has preview items (not matching active order)
        if not self._has_matching_items(bot, world):
            if len(bot.inventory) >= 3:
                return True  # Full inventory, park at drop-off for auto-recheck
            return False  # Still have capacity, go pick useful items instead

        if len(bot.inventory) >= 3:
            return True  # Inventory full, deliver what matches

        active = world.state.active_orders
        if not active:
            return True  # No active order info, just deliver

        order = active[0]
        remaining = list(order.items_remaining)
        # Remove items in this bot's inventory that match the order
        for inv_item in bot.inventory:
            if inv_item in remaining:
                remaining.remove(inv_item)

        if not remaining:
            return True  # Order will be complete with what we have

        # Check if there are unclaimed items on the map we can still grab
        available_slots = 3 - len(bot.inventory)
        pickable = []
        for item_type in remaining:
            for item in world.items_of_type(item_type):
                if item.id not in claimed_items:
                    pickup_pos = world.best_pickup_position(bot.position, item.position)
                    if pickup_pos is not None:
                        pickable.append(item)
                        break  # One per type is enough to decide

        if pickable and available_slots > 0:
            # Check if we have enough rounds to pick more + deliver
            nearest_pick_dist = min(
                world.distance(bot.position, world.best_pickup_position(bot.position, i.position))
                for i in pickable
                if world.best_pickup_position(bot.position, i.position) is not None
            )
            d_to_drop = world.distance(bot.position, world.state.drop_off)
            rounds_needed = nearest_pick_dist + 1 + d_to_drop + 1 + 5  # +5 margin
            if rounds_needed <= world.rounds_remaining:
                return False  # Keep picking

        return True  # Deliver what we have
