"""
Route builder: generates candidate multi-item routes for a bot.

Given a bot, an order, and the world model, produces a list of Route candidates
ranked by total cost. Each route is a sequence of 1-3 items to pick up before
delivering to the drop-off.

Algorithm: greedy nearest-neighbor with multiple start points.
Respects type counts from the order (e.g. if order needs 1 butter, only 1 butter in route).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from bot.models import Bot, Order, Item, Pos
from bot.engine.world_model import WorldModel
from bot.strategy.task import Route, RouteStop

logger = logging.getLogger(__name__)

MAX_ROUTE_ITEMS = 3  # Inventory capacity
MAX_CANDIDATES = 8   # Max routes to return
MAX_STARTS = 8       # Max different start items to try


def build_routes(
    bot: Bot,
    world: WorldModel,
    order: Optional[Order],
    claimed_items: set[str],
    preview_order: Optional[Order] = None,
) -> list[Route]:
    """
    Build candidate routes for a bot to pick up items for an order.

    Returns routes sorted by total_cost (cheapest first), max MAX_CANDIDATES.
    Always includes single-item routes as fallback.
    """
    if order is None:
        return []

    # Count how many of each type the order still needs
    type_budget = Counter(order.items_remaining)

    # Subtract items already in bot's inventory (they'll be delivered too)
    for inv_item in bot.inventory:
        if type_budget[inv_item] > 0:
            type_budget[inv_item] -= 1

    # Remove types with 0 budget
    type_budget = +type_budget  # Removes zero/negative entries

    if not type_budget:
        return []

    # Find candidate items per type, limited to budget count
    # For each type, pick the closest N items (where N = budget for that type)
    available: list[tuple[Item, Pos]] = []

    for item_type, count_needed in type_budget.items():
        type_items: list[tuple[Item, Pos, float]] = []
        for item in world.items_of_type(item_type):
            if item.id in claimed_items:
                continue
            pickup_pos = world.best_pickup_position(bot.position, item.position)
            if pickup_pos is None:
                continue
            d = world.distance(bot.position, pickup_pos)
            type_items.append((item, pickup_pos, d))

        # Sort by distance, take only what the order needs
        type_items.sort(key=lambda x: x[2])
        for item, pickup_pos, _ in type_items[:count_needed]:
            available.append((item, pickup_pos))

    if not available:
        logger.debug("No items available for order. Budget=%s, items_on_map=%s",
                      dict(type_budget),
                      {t: len(world.items_of_type(t)) for t in type_budget})
        return []

    # Account for items already in inventory (reduce capacity)
    capacity = MAX_ROUTE_ITEMS - len(bot.inventory)
    if capacity <= 0:
        return []

    drop_off = world.state.drop_off
    routes: list[Route] = []
    seen_item_sets: set[frozenset[str]] = set()

    # Try different starting items
    starts = available[:MAX_STARTS]

    for start_item, start_pickup in starts:
        stops: list[RouteStop] = []
        used_ids: set[str] = set()
        used_types: Counter = Counter()
        current_pos = bot.position
        total_dist = 0.0

        # First stop
        d = world.distance(current_pos, start_pickup)
        stops.append(RouteStop(
            item_id=start_item.id,
            item_type=start_item.type,
            item_pos=start_item.position,
            pickup_pos=start_pickup,
        ))
        used_ids.add(start_item.id)
        used_types[start_item.type] += 1
        total_dist += d
        current_pos = start_pickup

        # Greedily add more stops up to capacity
        for _ in range(capacity - 1):
            best_next: Optional[tuple[Item, Pos, float]] = None

            for item, pickup_pos in available:
                if item.id in used_ids:
                    continue
                # Respect type budget: don't pick more of a type than needed
                if used_types[item.type] >= type_budget.get(item.type, 0):
                    continue
                d_next = world.distance(current_pos, pickup_pos)
                if best_next is None or d_next < best_next[2]:
                    best_next = (item, pickup_pos, d_next)

            if best_next is None:
                break

            item, pickup_pos, d_next = best_next
            stops.append(RouteStop(
                item_id=item.id,
                item_type=item.type,
                item_pos=item.position,
                pickup_pos=pickup_pos,
            ))
            used_ids.add(item.id)
            used_types[item.type] += 1
            total_dist += d_next
            current_pos = pickup_pos

        # If route completes active order AND has capacity, add preview items "on the way"
        if preview_order and len(stops) < capacity:
            # Check if current stops + bot inventory covers all active remaining
            active_types_in_route = Counter(s.item_type for s in stops)
            active_types_in_inv = Counter(bot.inventory)
            combined = active_types_in_route + active_types_in_inv
            uncovered = Counter(order.items_remaining)
            for t, c in combined.items():
                uncovered[t] = max(0, uncovered.get(t, 0) - c)
            uncovered = +uncovered  # Remove zeros

            if not uncovered:  # Route completes active order!
                preview_budget = Counter(preview_order.items_remaining)
                for item_type in preview_budget:
                    if len(stops) >= capacity:
                        break
                    for item in world.items_of_type(item_type):
                        if item.id in used_ids or item.id in claimed_items:
                            continue
                        pp = world.best_pickup_position(current_pos, item.position)
                        if pp is None:
                            continue
                        # Only if "on the way" — max 4 extra steps vs direct to drop-off
                        d_direct = world.distance(current_pos, drop_off)
                        d_via = world.distance(current_pos, pp) + world.distance(pp, drop_off)
                        if d_via <= d_direct + 4:
                            stops.append(RouteStop(
                                item_id=item.id, item_type=item.type,
                                item_pos=item.position, pickup_pos=pp,
                            ))
                            used_ids.add(item.id)
                            total_dist += world.distance(current_pos, pp)
                            current_pos = pp
                            break  # One per type

        # Add delivery distance + action costs
        d_delivery = world.distance(current_pos, drop_off)
        # +1 per pick_up action, +1 for drop_off action
        total_cost = total_dist + d_delivery + len(stops) + 1

        # Check if route can be completed in time
        if total_cost > world.rounds_remaining:
            # Try shorter route (just the first stop)
            if len(stops) > 1:
                single_stop = stops[0]
                d_single = world.distance(bot.position, single_stop.pickup_pos)
                d_single_drop = world.distance(single_stop.pickup_pos, drop_off)
                single_cost = d_single + d_single_drop + 2
                if single_cost <= world.rounds_remaining:
                    item_set = frozenset({single_stop.item_id})
                    if item_set not in seen_item_sets:
                        seen_item_sets.add(item_set)
                        routes.append(Route(
                            stops=[single_stop],
                            order_id=order.id,
                            total_cost=single_cost,
                        ))
            continue

        # Deduplicate by item set (keep cheapest)
        item_set = frozenset(s.item_id for s in stops)
        if item_set in seen_item_sets:
            # Check if this ordering is cheaper
            for existing in routes:
                if existing.item_ids == set(item_set):
                    if total_cost < existing.total_cost:
                        existing.stops = list(stops)
                        existing.total_cost = total_cost
                    break
            continue

        seen_item_sets.add(item_set)
        routes.append(Route(
            stops=list(stops),
            order_id=order.id,
            total_cost=total_cost,
        ))

    # Also ensure single-item routes exist as fallback
    for item, pickup_pos in available:
        item_set = frozenset({item.id})
        if item_set in seen_item_sets:
            continue
        d_pick = world.distance(bot.position, pickup_pos)
        d_drop = world.distance(pickup_pos, drop_off)
        cost = d_pick + d_drop + 2
        if cost > world.rounds_remaining:
            continue
        seen_item_sets.add(item_set)
        routes.append(Route(
            stops=[RouteStop(
                item_id=item.id,
                item_type=item.type,
                item_pos=item.position,
                pickup_pos=pickup_pos,
            )],
            order_id=order.id,
            total_cost=cost,
        ))

    # Sort by cost and limit
    routes.sort(key=lambda r: r.total_cost)
    return routes[:MAX_CANDIDATES]
