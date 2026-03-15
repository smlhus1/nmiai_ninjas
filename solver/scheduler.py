"""Round-aware scheduler: all bots work on active order, then next.

Key game mechanic: drop_off only delivers items matching the ACTIVE order.
So all bots must focus on completing the active order before starting the next.

Strategy: for each order, generate trips, assign to available bots,
simulate completion time, then move to next order.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .grid import GameMap, Pos
from .pathfinding import DistanceCache
from .orders import Order, OrderQueue, ShelfIndex
from .trips import TripPlanner, Trip, PickupStep, trip_cost, _best_permutation


@dataclass
class BotSchedule:
    """Tracks a bot's timeline."""

    bot_id: int
    pos: Pos
    available_at: int = 0
    trips: list[ScheduledTrip] = field(default_factory=list)


@dataclass
class ScheduledTrip:
    """A trip assigned to a specific bot with timing."""

    bot_id: int
    trip: Trip
    order_idx: int
    start_round: int
    delivery_round: int


@dataclass
class OrderResult:
    """Result of planning one order."""

    order: Order
    order_idx: int
    trips: list[ScheduledTrip]
    all_delivered_round: int
    items_count: int


@dataclass
class FullPlan:
    """Complete game plan."""

    order_results: list[OrderResult]
    total_score: int
    total_rounds: int
    trips_per_bot: dict[int, int]
    orders_completed: int


class Scheduler:
    """All bots work on active order, greedy assignment, order by order."""

    def __init__(
        self,
        game_map: GameMap,
        dist_cache: DistanceCache,
        shelf_index: ShelfIndex,
        trip_planner: TripPlanner,
        max_rounds: int = 500,
    ) -> None:
        self.game_map = game_map
        self.dist_cache = dist_cache
        self.shelf_index = shelf_index
        self.trip_planner = trip_planner
        self.max_rounds = max_rounds

    def schedule(
        self,
        order_queue: OrderQueue,
        bot_count: int,
        spawn: Pos,
        max_items_per_trip: int = 3,
    ) -> FullPlan:
        """Schedule all orders sequentially — all bots focus on active order."""
        bots = [BotSchedule(bot_id=i, pos=spawn) for i in range(bot_count)]
        order_results: list[OrderResult] = []
        current_round = 0

        for order_idx, order in enumerate(order_queue):
            if current_round >= self.max_rounds:
                break

            result = self._schedule_order(
                order, order_idx, bots, current_round, max_items_per_trip,
            )
            order_results.append(result)
            current_round = result.all_delivered_round

        # Score
        total_score = 0
        orders_completed = 0
        for r in order_results:
            if r.all_delivered_round <= self.max_rounds:
                total_score += r.items_count
                if r.items_count >= r.order.item_count:
                    total_score += 5
                    orders_completed += 1

        return FullPlan(
            order_results=order_results,
            total_score=total_score,
            total_rounds=min(current_round, self.max_rounds),
            trips_per_bot={b.bot_id: len(b.trips) for b in bots},
            orders_completed=orders_completed,
        )

    def _schedule_order(
        self,
        order: Order,
        order_idx: int,
        bots: list[BotSchedule],
        current_round: int,
        max_items_per_trip: int,
    ) -> OrderResult:
        """Plan an order: generate trips, assign to best available bots."""
        items_needed = order.items_as_counter()
        trips = self._generate_trips(items_needed, max_items_per_trip)

        if not trips:
            return OrderResult(
                order=order, order_idx=order_idx, trips=[],
                all_delivered_round=current_round, items_count=0,
            )

        # Assign trips to bots greedily: earliest-finishing bot
        scheduled = self._assign_greedy(trips, bots, order_idx, current_round)

        end_round = max(
            (st.delivery_round for st in scheduled),
            default=current_round,
        )

        items_count = sum(st.trip.item_count for st in scheduled)

        return OrderResult(
            order=order,
            order_idx=order_idx,
            trips=scheduled,
            all_delivered_round=end_round,
            items_count=items_count,
        )

    def _generate_trips(
        self,
        items_needed: Counter[str],
        max_items_per_trip: int,
    ) -> list[Trip]:
        """Generate zone-optimized trips for an order."""
        zone_items: dict[Pos, list[PickupStep]] = {
            dz: [] for dz in self.game_map.drop_off_zones
        }

        for item_type, count in items_needed.items():
            for _ in range(count):
                best_step = None
                best_zone = None
                best_rt = float("inf")

                for dz in self.game_map.drop_off_zones:
                    entries = self.shelf_index.get(item_type, dz)
                    if not entries:
                        continue
                    entry = entries[0]
                    d_to = self.dist_cache.distance(dz, entry.pickup_pos)
                    if d_to is not None:
                        rt = d_to + entry.distance_to_dropoff
                        if rt < best_rt:
                            best_rt = rt
                            best_zone = dz
                            best_step = PickupStep(
                                item_type=item_type,
                                shelf_pos=entry.shelf_pos,
                                pickup_pos=entry.pickup_pos,
                            )

                if best_step and best_zone:
                    zone_items[best_zone].append(best_step)

        trips: list[Trip] = []
        for dz, steps in zone_items.items():
            if not steps:
                continue
            steps.sort(key=lambda s: self.dist_cache.distance(dz, s.pickup_pos) or 999)
            for i in range(0, len(steps), max_items_per_trip):
                batch = steps[i : i + max_items_per_trip]
                result = _best_permutation(dz, batch, dz, self.dist_cache)
                if result:
                    ordered, cost = result
                    trips.append(Trip(steps=ordered, drop_off=dz, cost=cost))

        return trips

    def _assign_greedy(
        self,
        trips: list[Trip],
        bots: list[BotSchedule],
        order_idx: int,
        current_round: int,
    ) -> list[ScheduledTrip]:
        """Assign trips to bots — bot with earliest finish time gets each trip."""
        scheduled: list[ScheduledTrip] = []

        # Sort trips by cost (do expensive ones first — they need the closest bots)
        for trip in sorted(trips, key=lambda t: -t.cost):
            best_bot = None
            best_end = float("inf")

            for bot in bots:
                start = max(current_round, bot.available_at)
                c = trip_cost(bot.pos, trip.steps, trip.drop_off, self.dist_cache)
                if c is None:
                    continue
                end = start + c
                if end < best_end:
                    best_end = end
                    best_bot = bot

            if best_bot is None:
                continue

            start = max(current_round, best_bot.available_at)
            c = trip_cost(best_bot.pos, trip.steps, trip.drop_off, self.dist_cache)
            end = start + c

            st = ScheduledTrip(
                bot_id=best_bot.bot_id,
                trip=trip,
                order_idx=order_idx,
                start_round=start,
                delivery_round=int(end),
            )
            scheduled.append(st)
            best_bot.trips.append(st)
            best_bot.pos = trip.drop_off
            best_bot.available_at = int(end)

        return scheduled
