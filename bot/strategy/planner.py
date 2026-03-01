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

logger = logging.getLogger(__name__)


class TaskPlanner:
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
            task = self._find_fallback_task(world, bot, claimed_items, assignments)
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
        prev_count = Counter(prev_inv).get(item_type, 0)
        curr_count = Counter(bot_inventory).get(item_type, 0)
        return curr_count > prev_count

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

            # Check if current item has been picked up:
            # 1. Item disappeared from map (finite supply)
            # 2. Bot's inventory gained the expected type (infinite supply shelves)
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

        # Count how many of each type the preview order needs
        preview_budget = Counter(preview_order.items_remaining)
        # Subtract items already in bot's inventory
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

            # Bots with inventory: deliver if matching active order
            if bot.inventory:
                has_match = self._has_matching_items(bot, world)
                if has_match:
                    assignment.task = Task(
                        task_type=TaskType.DELIVER,
                        target_pos=state.drop_off,
                    )
                    assignment.path = None
                    logger.debug("Bot %d endgame: delivering matching inventory", bot_id)
                    continue
                elif len(bot.inventory) >= 3:
                    # Full inventory, nothing matches — can't do anything useful
                    assignment.task = Task(task_type=TaskType.IDLE, target_pos=bot.position)
                    logger.debug("Bot %d endgame: full non-matching inventory, idling", bot_id)
                    continue
                # Has non-matching items but capacity — fall through to pick matching items

            # Try multi-item route if enough time
            if world.rounds_remaining > 8:
                active_orders = state.active_orders
                if active_orders:
                    from bot.strategy.route_builder import build_routes
                    routes = build_routes(bot, world, active_orders[0], claimed_items)
                    # Find best route that fits in remaining time
                    for route in routes:
                        if route.total_cost <= world.rounds_remaining and len(route.stops) > 1:
                            first_stop = route.stops[0]
                            assignment.task = Task(
                                task_type=TaskType.PICK_UP,
                                target_pos=first_stop.pickup_pos,
                                item_id=first_stop.item_id,
                                item_type=first_stop.item_type,
                                item_pos=first_stop.item_pos,
                                order_id=route.order_id,
                            )
                            assignment.route = route
                            assignment.route_step = 0
                            assignment.path = None
                            for stop in route.stops:
                                claimed_items.add(stop.item_id)
                            logger.debug("Bot %d endgame: multi-item route (%d stops)",
                                         bot_id, len(route.stops))
                            break
                    if assignment.has_task:
                        continue

            # Bots without inventory: pick nearest deliverable item
            # Prioritize items matching active order (only those can be delivered)
            best_task: Optional[Task] = None
            best_dist = 9999
            active_types: set[str] = set()
            if state.active_orders:
                active_types = set(state.active_orders[0].items_remaining)

            for item in state.items:
                if item.id in claimed_items:
                    continue

                # In endgame, skip non-matching items (can't deliver them)
                if active_types and item.type not in active_types:
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

            # Invalidate future route stops that are gone
            if assignment.route:
                remaining_stops = assignment.route.stops[assignment.route_step:]
                remaining_stops = [
                    s for s in remaining_stops if s.item_id in current_item_ids
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
                        logger.debug("Bot %d: full inventory, delivering matching items", bot_id)
                        assignment.task = Task(
                            task_type=TaskType.DELIVER,
                            target_pos=state.drop_off,
                        )
                    elif self._has_matching_preview_items(bot, world):
                        logger.debug("Bot %d: full inventory, only preview match — idling for auto-delivery", bot_id)
                        assignment.task = Task(
                            task_type=TaskType.IDLE,
                            target_pos=bot.position,
                        )
                    else:
                        logger.debug("Bot %d: full inventory, no order match — deliver for +1 per item", bot_id)
                        assignment.task = Task(
                            task_type=TaskType.DELIVER,
                            target_pos=state.drop_off,
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
                                self._blacklisted_items[task.item_id] = state.round + 8
                            assignment.clear()
                            self._stuck_pick_rounds.pop(bot_id, None)
                            continue
                    else:
                        self._stuck_pick_rounds.pop(bot_id, None)
                else:
                    self._stuck_pick_rounds.pop(bot_id, None)

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
                    self._stuck_deliver_rounds.pop(bot_id, None)
                # Stuck detection: at drop-off but inventory unchanged
                elif bot.position == state.drop_off:
                    prev_inv = self._prev_inventory.get(bot_id)
                    if prev_inv == bot.inventory:
                        rounds_stuck = self._stuck_deliver_rounds.get(bot_id, 0) + 1
                        self._stuck_deliver_rounds[bot_id] = rounds_stuck
                        if rounds_stuck >= 2:
                            logger.debug("Bot %d: stuck at drop-off for %d rounds, clearing DELIVER",
                                         bot_id, rounds_stuck)
                            assignment.clear()
                            self._stuck_deliver_rounds.pop(bot_id, None)
                    else:
                        self._stuck_deliver_rounds.pop(bot_id, None)
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
        assignments: dict[int, BotAssignment] | None = None,
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
        if self._should_deliver(bot, world, claimed_items, assignments):
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
        assignments: dict[int, BotAssignment] | None = None,
    ) -> Optional[Task]:
        """Pick up ANY unclaimed item (for the +1 point) or IDLE."""
        state = world.state

        # Deliver if carrying anything
        if self._should_deliver(bot, world, claimed_items, assignments):
            return Task(
                task_type=TaskType.DELIVER,
                target_pos=state.drop_off,
            )

        # Can't pick if inventory full
        if len(bot.inventory) >= 3:
            return Task(task_type=TaskType.IDLE, target_pos=bot.position)

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

    def _has_matching_preview_items(self, bot: Bot, world: WorldModel) -> bool:
        """Check if bot has ANY inventory items matching the preview order (for auto-delivery on transition)."""
        preview = world.state.preview_orders
        if not preview:
            return False
        preview_types = set(preview[0].items_remaining)
        return any(inv in preview_types for inv in bot.inventory)

    def _should_deliver(
        self,
        bot: Bot,
        world: WorldModel,
        claimed_items: set[str],
        assignments: dict[int, BotAssignment] | None = None,
    ) -> bool:
        """Decide if bot should deliver now vs pick more items."""
        if not bot.inventory:
            return False

        # Bots with active routes should NOT deliver (still picking items)
        if assignments:
            assignment = assignments.get(bot.id)
            if assignment and assignment.route and not world.is_endgame():
                return False

        # Endgame: always deliver whatever we have
        if world.is_endgame():
            return True

        # Check if bot only has preview items (not matching active order)
        if not self._has_matching_items(bot, world):
            if self._has_matching_preview_items(bot, world):
                return False  # Wait for auto-delivery on order transition
            if len(bot.inventory) >= 3:
                return True  # Full inventory, nothing matches any order — deliver for +1 per item
            return False

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

        # Check if picking more would COMPLETE the order (worth +5 bonus)
        available_slots = 3 - len(bot.inventory)
        pickable_for_completion: list[Item] = []
        check_remaining = list(remaining)
        for item_type in check_remaining:
            for item in world.items_of_type(item_type):
                if item.id not in claimed_items:
                    pickup_pos = world.best_pickup_position(bot.position, item.position)
                    if pickup_pos is not None:
                        pickable_for_completion.append(item)
                        break

        if (len(pickable_for_completion) == len(remaining)
                and len(remaining) <= available_slots):
            # Can complete entire order! Estimate time
            pos = bot.position
            total_pick_dist = 0
            for item in pickable_for_completion:
                pp = world.best_pickup_position(pos, item.position)
                if pp:
                    total_pick_dist += world.distance(pos, pp)
                    pos = pp
            d_final_drop = world.distance(pos, world.state.drop_off)
            time_needed = total_pick_dist + len(pickable_for_completion) + d_final_drop + 2
            if time_needed <= world.rounds_remaining:
                return False  # Keep picking to complete order!

        # Check if there are unclaimed items on the map we can still grab
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
            rounds_needed = nearest_pick_dist + 1 + d_to_drop + 1 + 2  # +2 margin
            if rounds_needed <= world.rounds_remaining:
                return False  # Keep picking

        return True  # Deliver what we have
