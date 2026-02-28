"""
Hungarian (Munkres) assignment for globally optimal bot-to-route matching.

Uses scipy.optimize.linear_sum_assignment to find the assignment that
minimizes total cost across all bots. Bots are matched to multi-item
routes (1-3 items) instead of single items.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from bot.models import Bot, Item, Order
from bot.engine.world_model import WorldModel
from bot.strategy.task import Task, TaskType, BotAssignment, Route
from bot.strategy.route_builder import build_routes

logger = logging.getLogger(__name__)

BIG_COST = 99999.0


def solve_assignment(
    bots: list[Bot],
    world: WorldModel,
    assignments: dict[int, BotAssignment],
    claimed_items: set[str],
    preview_item_ids: set[str],
) -> dict[int, tuple[Task, Optional[Route]]]:
    """
    Compute globally optimal bot -> route assignment using Hungarian algorithm.

    Returns dict of bot_id -> (Task, Route) for bots that got assignments.
    Task is for the FIRST stop in the route. Route is the full plan (or None for deliveries).
    """
    state = world.state
    result: dict[int, tuple[Task, Optional[Route]]] = {}

    # Separate deliverers first (not part of Hungarian)
    picking_bots: list[Bot] = []
    for bot in bots:
        if bot.inventory and _should_deliver_quick(bot, world):
            result[bot.id] = (
                Task(task_type=TaskType.DELIVER, target_pos=state.drop_off),
                None,
            )
        else:
            picking_bots.append(bot)

    if not picking_bots:
        return result

    active_orders = state.active_orders
    if not active_orders:
        # No active orders — fall back to single-item assignment for non-order items
        return _assign_non_order_items(
            picking_bots, world, assignments, claimed_items, preview_item_ids, result
        )

    primary_order = active_orders[0]

    # Build candidate routes per bot
    bot_routes: list[list[Route]] = []
    for bot in picking_bots:
        routes = build_routes(bot, world, primary_order, claimed_items)
        bot_routes.append(routes)
        if not routes:
            logger.debug("Bot %d: no routes found for order %s (remaining=%s)",
                         bot.id, primary_order.id, list(primary_order.items_remaining))

    # Flatten all unique routes into a global list
    all_routes: list[Route] = []
    route_set: set[frozenset[str]] = set()
    for routes in bot_routes:
        for route in routes:
            key = frozenset(route.item_ids)
            if key not in route_set:
                route_set.add(key)
                all_routes.append(route)

    # Non-order items are now handled by _assign_preview_tasks (with inventory guards)
    # Hungarian only assigns active order routes to ensure active order priority

    if not all_routes:
        return result

    n_bots = len(picking_bots)
    n_routes = len(all_routes)

    # Build cost matrix: bot x route
    cost = np.full((n_bots, n_routes), BIG_COST)

    for i, bot in enumerate(picking_bots):
        current_assignment = assignments.get(bot.id)
        current_item_id = (
            current_assignment.task.item_id
            if current_assignment and current_assignment.task
            else None
        )
        # Current route item IDs for switching penalty
        current_route_ids: set[str] = set()
        if current_assignment and current_assignment.route:
            current_route_ids = current_assignment.route.item_ids

        capacity = 3 - len(bot.inventory)

        for j, route in enumerate(all_routes):
            # Skip routes that exceed remaining capacity
            if len(route.stops) > capacity:
                continue

            # Calculate cost from THIS bot's position
            first_stop = route.stops[0]
            d_first = world.distance(bot.position, first_stop.pickup_pos)

            # Sum inter-stop distances
            inter_dist = 0.0
            prev_pos = first_stop.pickup_pos
            for stop in route.stops[1:]:
                inter_dist += world.distance(prev_pos, stop.pickup_pos)
                prev_pos = stop.pickup_pos

            # Distance from last stop to drop-off
            d_delivery = world.distance(prev_pos, state.drop_off)

            # Total: travel + actions (1 per pick + 1 drop)
            total_cost = d_first + inter_dist + d_delivery + len(route.stops) + 1

            if total_cost > world.rounds_remaining:
                continue

            # Check all items are reachable from this bot
            all_reachable = all(
                world.best_pickup_position(bot.position, s.item_pos) is not None
                for s in route.stops
            )
            if not all_reachable:
                continue

            # Cost value based on order value
            is_preview = all(s.item_id in preview_item_ids for s in route.stops)
            if is_preview:
                cost_val = float(d_first + 1)
            elif route.order_id is not None:
                order = next(
                    (o for o in active_orders if o.id == route.order_id), None
                )
                order_val = max(world.order_value(order), 0.1) if order else 1.0
                # Multi-item routes get bonus: more items per trip = better value
                items_value = len(route.stops)  # Each item is worth at least +1
                cost_val = total_cost / (order_val * items_value)
            else:
                cost_val = float(total_cost)

            # Switching penalty: discourage changing route
            if current_route_ids and not current_route_ids.intersection(route.item_ids):
                cost_val += 3.0
            elif current_item_id and current_item_id not in route.item_ids:
                cost_val += 3.0

            cost[i][j] = cost_val

    # Solve assignment
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except ValueError:
        logger.warning("Hungarian assignment failed, returning partial results")
        return result

    # Post-assignment: resolve item overlaps (lower-cost assignments win)
    assigned_pairs = []
    for i, j in zip(row_ind, col_ind):
        if cost[i][j] >= BIG_COST:
            continue
        assigned_pairs.append((cost[i][j], i, j))

    # Sort by cost (lowest first = highest priority)
    assigned_pairs.sort()

    globally_claimed: set[str] = set()
    for _, i, j in assigned_pairs:
        bot = picking_bots[i]
        route = all_routes[j]

        # Check for item overlap with already-assigned routes
        overlap = route.item_ids.intersection(globally_claimed)
        if overlap:
            # Try to build a reduced route without overlapping items
            reduced_stops = [s for s in route.stops if s.item_id not in globally_claimed]
            if not reduced_stops:
                continue  # No items left, skip this bot
            route = Route(
                stops=reduced_stops,
                order_id=route.order_id,
                total_cost=route.total_cost,  # Approximate
            )

        first_stop = route.stops[0]

        # Hungarian only handles active order items — always PICK_UP
        task_type = TaskType.PICK_UP

        task = Task(
            task_type=task_type,
            target_pos=first_stop.pickup_pos,
            item_id=first_stop.item_id,
            item_type=first_stop.item_type,
            item_pos=first_stop.item_pos,
            order_id=route.order_id,
        )

        # Only include route for multi-stop routes
        final_route = route if len(route.stops) > 1 else None

        result[bot.id] = (task, final_route)
        globally_claimed.update(route.item_ids)

    return result


def _assign_non_order_items(
    bots: list[Bot],
    world: WorldModel,
    assignments: dict[int, BotAssignment],
    claimed_items: set[str],
    preview_item_ids: set[str],
    result: dict[int, tuple[Task, Optional[Route]]],
) -> dict[int, tuple[Task, Optional[Route]]]:
    """Fallback: assign bots to individual items when no active orders exist."""
    state = world.state
    candidates: list[tuple[Item, Optional[Order]]] = []
    seen_item_ids: set[str] = set()

    for item in state.items:
        if item.id in claimed_items or item.id in seen_item_ids:
            continue
        seen_item_ids.add(item.id)
        candidates.append((item, None))

    if not candidates:
        return result

    n_bots = len(bots)
    n_items = len(candidates)
    cost = np.full((n_bots, n_items), BIG_COST)

    for i, bot in enumerate(bots):
        for j, (item, _) in enumerate(candidates):
            pickup_pos = world.best_pickup_position(bot.position, item.position)
            if pickup_pos is None:
                continue
            d_pick = world.distance(bot.position, pickup_pos)
            d_drop = world.distance(pickup_pos, state.drop_off)
            if d_pick + d_drop + 2 > world.rounds_remaining:
                continue
            cost[i][j] = float(d_pick + d_drop + 2)

    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except ValueError:
        return result

    for i, j in zip(row_ind, col_ind):
        if cost[i][j] >= BIG_COST:
            continue
        bot = bots[i]
        item, _ = candidates[j]
        pickup_pos = world.best_pickup_position(bot.position, item.position)
        if pickup_pos is None:
            continue

        if item.id in preview_item_ids:
            task_type = TaskType.PRE_PICK
        else:
            task_type = TaskType.PICK_UP

        result[bot.id] = (
            Task(
                task_type=task_type,
                target_pos=pickup_pos,
                item_id=item.id,
                item_type=item.type,
                item_pos=item.position,
            ),
            None,
        )

    return result


def _should_deliver_quick(bot: Bot, world: WorldModel) -> bool:
    """Quick deliver check for Hungarian pre-filter."""
    if not bot.inventory:
        return False

    if world.is_endgame():
        return True

    active = world.state.active_orders
    if not active:
        return bool(bot.inventory)

    remaining = list(active[0].items_remaining)
    for inv_item in bot.inventory:
        if inv_item in remaining:
            return True  # Has matching item for active order

    # No active-order match: wait if inventory matches preview (auto-delivery on transition)
    preview = world.state.preview_orders
    if preview:
        preview_types = set(preview[0].items_remaining)
        if any(inv in preview_types for inv in bot.inventory):
            return False

    return len(bot.inventory) >= 3  # Full and no order match — deliver for +1 per item
