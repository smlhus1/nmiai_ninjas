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
from itertools import permutations, combinations
from typing import Optional

from bot.models import Bot, Order, Item, Pos
from bot.engine.world_model import WorldModel
from bot.strategy.task import Route, RouteStop

logger = logging.getLogger(__name__)

MAX_ROUTE_ITEMS = 3  # Inventory capacity
MAX_CANDIDATES = 8   # Max routes to return (baseline for 3+ bots)
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

    available: list[tuple[Item, Pos]] = []
    EXTRA_ALTERNATIVES = 2

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

        type_items.sort(key=lambda x: x[2])
        for item, pickup_pos, _ in type_items[:count_needed + EXTRA_ALTERNATIVES]:
            available.append((item, pickup_pos))

        if count_needed > 1 and type_items:
            nearest = type_items[0]
            for k in range(1, count_needed):
                camp_item = Item(
                    id=f"camp_{item_type}_{k}",
                    type=item_type,
                    position=nearest[0].position,
                )
                available.append((camp_item, nearest[1]))

    if not available:
        logger.debug("No items available for order. Budget=%s, items_on_map=%s",
                      dict(type_budget),
                      {t: len(world.items_of_type(t)) for t in type_budget})
        return []

    capacity = MAX_ROUTE_ITEMS - len(bot.inventory)
    if capacity <= 0:
        return []

    drop_off = world.state.drop_off
    total_items_needed = sum(type_budget.values())

    routes: list[Route] = []
    seen_item_sets: set[frozenset[str]] = set()

    def _tsp_solve(bot_pos: Pos, stops: list[RouteStop]) -> tuple[list[RouteStop], float]:
        """Find TSP-optimal ordering and walking distance in one pass."""
        if len(stops) <= 1:
            if not stops:
                return [], 0.0
            cost = (world.distance(bot_pos, stops[0].pickup_pos)
                    + world.distance(stops[0].pickup_pos, drop_off))
            return list(stops), cost

        best_cost = float('inf')
        best_order = list(range(len(stops)))
        for perm in permutations(range(len(stops))):
            cost = world.distance(bot_pos, stops[perm[0]].pickup_pos)
            for k in range(1, len(perm)):
                cost += world.distance(stops[perm[k-1]].pickup_pos, stops[perm[k]].pickup_pos)
            cost += world.distance(stops[perm[-1]].pickup_pos, drop_off)
            if cost < best_cost:
                best_cost = cost
                best_order = list(perm)
        return [stops[i] for i in best_order], best_cost

    def _make_route(stops: list[RouteStop]) -> None:
        """Create a route from stops, apply TSP ordering, add to results."""
        ordered, walk = _tsp_solve(bot.position, stops)
        trip_cost = walk + len(ordered) + 1

        if trip_cost > world.rounds_remaining:
            if len(ordered) > 1:
                for s in ordered:
                    _make_route([s])
            return

        item_set = frozenset(s.item_id for s in ordered)
        if item_set in seen_item_sets:
            for existing in routes:
                if existing.item_ids == item_set and trip_cost < existing.total_cost:
                    existing.stops = list(ordered)
                    existing.total_cost = trip_cost
            return

        seen_item_sets.add(item_set)
        routes.append(Route(
            stops=list(ordered),
            order_id=order.id,
            total_cost=trip_cost,
        ))

    if len(available) <= 12:
        for size in range(min(capacity, len(available)), 0, -1):
            for combo in combinations(range(len(available)), size):
                combo_types = Counter(available[i][0].type for i in combo)
                valid = all(combo_types[t] <= type_budget.get(t, 0) for t in combo_types)
                if not valid:
                    continue
                stops = [
                    RouteStop(
                        item_id=available[i][0].id, item_type=available[i][0].type,
                        item_pos=available[i][0].position, pickup_pos=available[i][1],
                    ) for i in combo
                ]
                _make_route(stops)
    else:
        starts = available[:MAX_STARTS]
        for start_item, start_pickup in starts:
            stops: list[RouteStop] = []
            used_ids: set[str] = set()
            used_types: Counter = Counter()
            current_pos = bot.position

            stops.append(RouteStop(
                item_id=start_item.id, item_type=start_item.type,
                item_pos=start_item.position, pickup_pos=start_pickup,
            ))
            used_ids.add(start_item.id)
            used_types[start_item.type] += 1
            current_pos = start_pickup

            for _ in range(capacity - 1):
                best_next: Optional[tuple[Item, Pos, float]] = None
                for item, pickup_pos in available:
                    if item.id in used_ids:
                        continue
                    if used_types[item.type] >= type_budget.get(item.type, 0):
                        continue
                    d_next = world.distance(current_pos, pickup_pos)
                    if best_next is None or d_next < best_next[2]:
                        best_next = (item, pickup_pos, d_next)
                if best_next is None:
                    break
                item, pickup_pos, d_next = best_next
                stops.append(RouteStop(
                    item_id=item.id, item_type=item.type,
                    item_pos=item.position, pickup_pos=pickup_pos,
                ))
                used_ids.add(item.id)
                used_types[item.type] += 1
                current_pos = pickup_pos

            _make_route(stops)

    # Cross-order pre-picking: if route completes active order, add preview items
    # Only for single-bot scenarios — with multiple bots, dedicated preview pickers handle this
    # to avoid over-picking preview items (causes permanent stale inventory)
    if preview_order and len(world.state.bots) <= 2:
        for route in list(routes):
            if len(route.stops) >= capacity:
                continue
            active_types = Counter(s.item_type for s in route.stops) + Counter(bot.inventory)
            uncovered = Counter(order.items_remaining)
            for t, c in active_types.items():
                uncovered[t] = max(0, uncovered.get(t, 0) - c)
            if +uncovered:
                continue
            used_ids = {s.item_id for s in route.stops}
            last_pos = route.stops[-1].pickup_pos
            preview_budget = Counter(preview_order.items_remaining)
            for item_type in preview_budget:
                if len(route.stops) >= capacity:
                    break
                for item in world.items_of_type(item_type):
                    if item.id in used_ids or item.id in claimed_items:
                        continue
                    pp = world.best_pickup_position(last_pos, item.position)
                    if pp is None:
                        continue
                    d_direct = world.distance(last_pos, drop_off)
                    d_via = world.distance(last_pos, pp) + world.distance(pp, drop_off)
                    margin = 6 if len(world.state.bots) <= 2 else 4
                    if d_via <= d_direct + margin:
                        route.stops.append(RouteStop(
                            item_id=item.id, item_type=item.type,
                            item_pos=item.position, pickup_pos=pp,
                        ))
                        used_ids.add(item.id)
                        last_pos = pp
                        break

    # Single-item fallbacks
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
                item_id=item.id, item_type=item.type,
                item_pos=item.position, pickup_pos=pickup_pos,
            )],
            order_id=order.id,
            total_cost=cost,
        ))

    routes.sort(key=lambda r: r.total_cost)

    # With few bots, ensure route size diversity so Hungarian can evaluate
    # order-completion tradeoffs (e.g., 3-stop completing an order vs 2-stop).
    # With many bots, parallelism matters more — just use cheapest routes.
    if len(world.state.bots) <= 2:
        best_by_size: dict[int, Route] = {}
        for r in routes:
            n = len(r.stops)
            if n not in best_by_size or r.total_cost < best_by_size[n].total_cost:
                best_by_size[n] = r

        result = routes[:MAX_CANDIDATES]
        result_ids = {frozenset(r.item_ids) for r in result}
        for r in best_by_size.values():
            if frozenset(r.item_ids) not in result_ids:
                result.append(r)
        return result

    # More candidates for small multi-bot scenarios
    n_cands = MAX_CANDIDATES + 4 if len(world.state.bots) == 3 else MAX_CANDIDATES
    return routes[:n_cands]
