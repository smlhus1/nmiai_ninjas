"""
Hungarian (Munkres) assignment for globally optimal bot-to-item matching.

Uses scipy.optimize.linear_sum_assignment to find the assignment that
minimizes total cost (trip_cost / order_value) across all bots.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from bot.models import Bot, Order
from bot.engine.world_model import WorldModel
from bot.strategy.task import Task, TaskType, BotAssignment

logger = logging.getLogger(__name__)

BIG_COST = 99999.0


def solve_assignment(
    bots: list[Bot],
    world: WorldModel,
    assignments: dict[int, BotAssignment],
    claimed_items: set[str],
    preview_item_ids: set[str],
) -> dict[int, Task]:
    """
    Compute globally optimal bot -> item assignment using Hungarian algorithm.

    Returns dict of bot_id -> Task for bots that got assignments.
    Bots that should deliver or have no valid item get skipped.
    """
    state = world.state
    result: dict[int, Task] = {}

    # Separate deliverers first (not part of Hungarian)
    picking_bots: list[Bot] = []
    for bot in bots:
        if bot.inventory and _should_deliver_quick(bot, world):
            result[bot.id] = Task(
                task_type=TaskType.DELIVER,
                target_pos=state.drop_off,
            )
        else:
            picking_bots.append(bot)

    if not picking_bots:
        return result

    # Build candidate items (unclaimed, on active orders)
    candidates: list[tuple[object, Optional[Order]]] = []  # (item, order_or_None)
    active_orders = state.active_orders

    # Active order items first
    seen_item_ids: set[str] = set()
    for order in active_orders:
        for item_type in order.items_remaining:
            for item in world.items_of_type(item_type):
                if item.id in claimed_items or item.id in seen_item_ids:
                    continue
                seen_item_ids.add(item.id)
                candidates.append((item, order))

    # Then any remaining items (for +1 point)
    for item in state.items:
        if item.id in claimed_items or item.id in seen_item_ids:
            continue
        seen_item_ids.add(item.id)
        candidates.append((item, None))

    if not candidates:
        return result

    n_bots = len(picking_bots)
    n_items = len(candidates)

    # Build cost matrix
    cost = np.full((n_bots, n_items), BIG_COST)

    for i, bot in enumerate(picking_bots):
        current_target = assignments.get(bot.id)
        current_item_id = (
            current_target.task.item_id
            if current_target and current_target.task
            else None
        )

        for j, (item, order) in enumerate(candidates):
            pickup_pos = world.best_pickup_position(bot.position, item.position)
            if pickup_pos is None:
                continue

            d_pick = world.distance(bot.position, pickup_pos)
            d_drop = world.distance(pickup_pos, state.drop_off)

            # Can't complete trip in time
            if d_pick + d_drop + 2 > world.rounds_remaining:
                continue

            trip_cost = d_pick + d_drop + 2

            if item.id in preview_item_ids:
                # Preview items: just pick distance, no delivery needed
                cost_val = float(d_pick + 1)
            elif order is not None:
                order_val = max(world.order_value(order), 0.1)
                cost_val = trip_cost / order_val
            else:
                # Non-order item: raw trip cost (worth +1 point)
                cost_val = float(trip_cost)

            # Switching penalty: discourage changing target
            if current_item_id and current_item_id != item.id:
                cost_val += 3.0

            cost[i][j] = cost_val

    # Solve assignment
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except ValueError:
        logger.warning("Hungarian assignment failed, returning partial results")
        return result

    for i, j in zip(row_ind, col_ind):
        if cost[i][j] >= BIG_COST:
            continue  # No valid assignment

        bot = picking_bots[i]
        item, order = candidates[j]

        pickup_pos = world.best_pickup_position(bot.position, item.position)
        if pickup_pos is None:
            continue

        # Determine task type
        if item.id in preview_item_ids:
            task_type = TaskType.PRE_PICK
            order_id = order.id if order else None
        else:
            task_type = TaskType.PICK_UP
            order_id = order.id if order else None

        result[bot.id] = Task(
            task_type=task_type,
            target_pos=pickup_pos,
            item_id=item.id,
            item_type=item.type,
            item_pos=item.position,
            order_id=order_id,
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
            return True  # Has matching item

    return len(bot.inventory) >= 3  # Full inventory
