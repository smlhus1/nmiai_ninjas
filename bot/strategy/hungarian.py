"""
Hungarian (Munkres) assignment for globally optimal bot-to-route matching.

Uses scipy.optimize.linear_sum_assignment to find the assignment that
minimizes total cost across all bots. Bots are matched to multi-item
routes (1-3 items) instead of single items.
"""

from __future__ import annotations

import logging
from collections import Counter
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

    preview_orders = state.preview_orders
    preview_order = preview_orders[0] if preview_orders else None

    # Build candidate routes per bot
    bot_routes: list[list[Route]] = []
    for bot in picking_bots:
        routes = build_routes(bot, world, primary_order, claimed_items, preview_order)
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

            if route.order_id is not None:
                order = next(
                    (o for o in active_orders if o.id == route.order_id), None
                )
                if order:
                    items_in_route = Counter(s.item_type for s in route.stops)
                    items_in_inv = Counter(bot.inventory)
                    combined = items_in_route + items_in_inv
                    remaining_after = Counter(order.items_remaining)
                    for t, c in combined.items():
                        remaining_after[t] = max(0, remaining_after.get(t, 0) - c)
                    remaining_after = +remaining_after
                    completes_order = not remaining_after

                    effective_items = len(route.stops) + (5 if completes_order else 0)
                    cost_val = total_cost / max(effective_items, 1)

                    if not completes_order and remaining_after:
                        leftover_cost = _estimate_leftover_cost(
                            world, bot, remaining_after, route, claimed_items
                        )
                        cost_val += leftover_cost / max(effective_items, 1)
                else:
                    cost_val = total_cost / max(len(route.stops), 1)
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


def _estimate_leftover_cost(
    world: WorldModel,
    bot: Bot,
    remaining_types: Counter,
    current_route: "Route",
    claimed_items: set[str],
) -> float:
    """
    Estimate the round cost of picking up and delivering the items that
    don't fit in the current route. Used to penalize routes that leave
    expensive items behind.
    """
    drop_off = world.state.drop_off
    route_item_ids = current_route.item_ids

    leftover_pickups: list[tuple[float, "Pos"]] = []
    for item_type, count in remaining_types.items():
        found = 0
        for item in world.items_of_type(item_type):
            if item.id in claimed_items or item.id in route_item_ids:
                continue
            pp = world.best_pickup_position(drop_off, item.position)
            if pp is None:
                continue
            d_round_trip = world.distance(drop_off, pp) + world.distance(pp, drop_off)
            leftover_pickups.append((d_round_trip, pp))
            found += 1
            if found >= count:
                break

    if not leftover_pickups:
        return 0.0

    leftover_pickups.sort()
    total = 0.0
    batch_size = 3
    for i in range(0, len(leftover_pickups), batch_size):
        batch = leftover_pickups[i:i + batch_size]
        farthest_rt = max(rt for rt, _ in batch)
        total += farthest_rt + len(batch) + 1

    return total


def _should_deliver_quick(bot: Bot, world: WorldModel) -> bool:
    """Quick deliver check for Hungarian pre-filter.

    Only force delivery when OBVIOUSLY correct. Let Hungarian
    decide in ambiguous cases by considering multi-item routes.
    """
    if not bot.inventory:
        return False
    if world.is_endgame():
        return True
    if len(bot.inventory) >= 3:
        return True  # Full — can't pick more

    active = world.state.active_orders
    if not active:
        return bool(bot.inventory)

    order = active[0]
    remaining = list(order.items_remaining)

    # Check if bot's inventory completes the order
    remaining_copy = list(remaining)
    for inv_item in bot.inventory:
        if inv_item in remaining_copy:
            remaining_copy.remove(inv_item)
    if not remaining_copy:
        return True  # All remaining items in inventory — deliver for +5!

    # Has matching items but order not complete — let Hungarian decide
    # (it may find a multi-item route that picks more before delivering)

    # No active-order match: check preview for auto-delivery
    has_match = any(inv in remaining for inv in bot.inventory)
    if not has_match:
        preview = world.state.preview_orders
        if preview:
            preview_types = set(preview[0].items_remaining)
            if any(inv in preview_types for inv in bot.inventory):
                return False  # Wait for auto-delivery

    return False  # Don't force — let Hungarian handle it
