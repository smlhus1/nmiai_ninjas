"""
Decision helpers for TaskPlanner.

Endgame planning, greedy task finding, fallback tasks,
and delivery decision logic.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from bot.models import Bot, Item
from bot.engine.world_model import WorldModel
from bot.strategy.task import Task, TaskType, BotAssignment

if TYPE_CHECKING:
    from bot.strategy.planner import TaskPlanner

logger = logging.getLogger(__name__)


class DecisionsMixin:

    def _plan_endgame(
        self: TaskPlanner,
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
                    # Full inventory, nothing matches — park away from drop-off
                    assignment.task = self._make_parking_task(bot, world)
                    logger.debug("Bot %d endgame: full non-matching, parking", bot_id)
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
                assignment.task = self._make_parking_task(bot, world)

    def _find_best_task(
        self: TaskPlanner,
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

        # Can't pick up if inventory full
        if len(bot.inventory) >= 3:
            return None

        # Priority 2: Find best item for ACTIVE order only
        # (preview pre-picking is handled separately by _assign_preview_tasks)
        best_task: Optional[Task] = None
        best_score: float = -1.0

        active_orders = state.active_orders
        if not active_orders:
            return None

        for order in active_orders:
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
        self: TaskPlanner,
        world: WorldModel,
        bot: Bot,
        claimed_items: set[str],
        assignments: dict[int, BotAssignment] | None = None,
        covered_types: set[str] | None = None,
    ) -> Optional[Task]:
        """Pick up any unclaimed item, or IDLE."""
        state = world.state

        if self._should_deliver(bot, world, claimed_items, assignments):
            return Task(
                task_type=TaskType.DELIVER,
                target_pos=state.drop_off,
            )

        if len(bot.inventory) >= 3:
            # Park out of the way so we don't block aisles or drop-off
            parking = world.parking_positions()
            if parking:
                other_bots = world.bot_positions_except(bot.id)
                free = [p for p in parking if p not in other_bots]
                if free:
                    target = min(free, key=lambda p: world.distance(bot.position, p))
                else:
                    target = min(parking, key=lambda p: world.distance(bot.position, p))
            else:
                target = bot.position
            return Task(task_type=TaskType.IDLE, target_pos=target)

        # Only pick up items matching active or preview orders (don't hoard junk)
        wanted_types: set[str] = set()
        for order in state.orders:
            if not order.complete:
                wanted_types.update(order.items_remaining)

        # Skip active-order types that are already fully covered by other bots
        if covered_types:
            wanted_types -= covered_types

        for item in state.items:
            if item.type not in wanted_types:
                continue
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

        # IDLE — park away from drop-off to avoid blocking deliveries
        return self._make_parking_task(bot, world)

    def _make_parking_task(
        self: TaskPlanner,
        bot: Bot,
        world: WorldModel,
    ) -> Task:
        """Create an IDLE task parked away from the drop-off zone."""
        parking = world.parking_positions()
        if parking:
            other_bots = world.bot_positions_except(bot.id)
            free = [p for p in parking if p not in other_bots]
            if free:
                target = min(free, key=lambda p: world.distance(bot.position, p))
            else:
                target = min(parking, key=lambda p: world.distance(bot.position, p))
        else:
            target = bot.position
        return Task(task_type=TaskType.IDLE, target_pos=target)

    def _has_matching_items(self: TaskPlanner, bot: Bot, world: WorldModel) -> bool:
        """Check if bot has ANY inventory items matching the active order."""
        active = world.state.active_orders
        if not active:
            return False
        remaining = list(active[0].items_remaining)
        for inv_item in bot.inventory:
            if inv_item in remaining:
                return True
        return False

    def _has_matching_preview_items(self: TaskPlanner, bot: Bot, world: WorldModel) -> bool:
        """Check if bot has ANY inventory items matching the preview order (for auto-delivery on transition)."""
        preview = world.state.preview_orders
        if not preview:
            return False
        preview_types = set(preview[0].items_remaining)
        return any(inv in preview_types for inv in bot.inventory)

    def _should_deliver(
        self: TaskPlanner,
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

        if world.is_endgame():
            return self._has_matching_items(bot, world)

        if not self._has_matching_items(bot, world):
            if self._has_matching_preview_items(bot, world):
                return False
            # Nothing matches active order — delivering won't work (non-matching
            # items stay in inventory).  Do NOT send to drop-off.
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
            # Can complete entire order! But first check if other bots already
            # have the remaining items — if so, don't try to solo-complete
            if assignments and len(world.state.bots) >= 4:
                from collections import Counter
                remaining_counter = Counter(remaining)
                team_has: Counter = Counter()
                for other_bot in world.state.bots:
                    if other_bot.id == bot.id:
                        continue
                    for inv_item in other_bot.inventory:
                        if inv_item in remaining_counter:
                            team_has[inv_item] += 1
                for bid, a in assignments.items():
                    if bid == bot.id:
                        continue
                    if (a.task and a.task.task_type == TaskType.PICK_UP
                            and a.task.item_type in remaining_counter):
                        team_has[a.task.item_type] += 1
                    if a.route:
                        for stop in a.route.stops[a.route_step:]:
                            if stop.item_type in remaining_counter:
                                team_has[stop.item_type] += 1
                if all(team_has.get(t, 0) >= remaining_counter[t]
                       for t in remaining_counter):
                    return True  # Team has it covered, deliver now

            # Estimate time to pick remaining + deliver
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
            if rounds_needed > world.rounds_remaining:
                return True  # Not enough time, deliver now

            # Don't detour if picking adds too many rounds vs direct delivery
            d_direct_drop = world.distance(bot.position, world.state.drop_off)
            detour_cost = nearest_pick_dist + 1 + d_to_drop - d_direct_drop
            if detour_cost > 10:
                return True  # Detour too expensive, deliver now

            return False  # Keep picking

        return True  # Deliver what we have
