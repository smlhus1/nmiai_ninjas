"""Bot-to-trip assignment using Hungarian algorithm and greedy scheduling."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .grid import GameMap, Pos
from .pathfinding import DistanceCache
from .orders import Order, OrderQueue, ShelfIndex
from .trips import Trip, TripPlanner, PickupStep, trip_cost


@dataclass
class BotState:
    """Tracks a single bot's state during assignment planning."""

    bot_id: int
    pos: Pos
    inventory: list[str] = field(default_factory=list)
    assigned_trip: Trip | None = None
    available_round: int = 0  # round when bot is free


@dataclass
class ScheduledTrip:
    """A trip assigned to a specific bot at a specific time."""

    bot_id: int
    trip: Trip
    start_round: int
    end_round: int  # estimated completion round


@dataclass
class OrderPlan:
    """Complete plan for fulfilling one order."""

    order: Order
    scheduled_trips: list[ScheduledTrip]
    estimated_start: int
    estimated_end: int

    @property
    def total_items(self) -> int:
        return sum(t.trip.item_count for t in self.scheduled_trips)


class AssignmentEngine:
    """Assigns trips to bots to minimize total makespan for each order."""

    def __init__(
        self,
        game_map: GameMap,
        dist_cache: DistanceCache,
        shelf_index: ShelfIndex,
        trip_planner: TripPlanner,
    ) -> None:
        self.game_map = game_map
        self.dist_cache = dist_cache
        self.shelf_index = shelf_index
        self.trip_planner = trip_planner

    def plan_order(
        self,
        order: Order,
        bots: list[BotState],
        current_round: int,
        max_items_per_trip: int = 3,
    ) -> OrderPlan:
        """Create a plan to fulfill an entire order.

        Uses Hungarian assignment to optimally match bots to trips.
        """
        items_needed = order.items_as_counter()

        # Generate candidate trips for each drop-off zone
        all_candidate_trips: list[Trip] = []

        for dz in self.game_map.drop_off_zones:
            trips = self.trip_planner.generate_trips(
                items_needed,
                bot_pos=dz,  # reference position for TSP ordering
                max_items=max_items_per_trip,
                drop_off=dz,
            )
            all_candidate_trips.extend(trips)

        # Deduplicate: keep unique trips by items covered
        trips = self._deduplicate_trips(all_candidate_trips, items_needed)

        if not trips:
            return OrderPlan(
                order=order,
                scheduled_trips=[],
                estimated_start=current_round,
                estimated_end=current_round,
            )

        # Build cost matrix: bots x trips
        scheduled = self._hungarian_assign(bots, trips, current_round)

        est_end = max(
            (st.end_round for st in scheduled),
            default=current_round,
        )

        return OrderPlan(
            order=order,
            scheduled_trips=scheduled,
            estimated_start=current_round,
            estimated_end=est_end,
        )

    def _deduplicate_trips(
        self,
        candidates: list[Trip],
        items_needed: Counter[str],
    ) -> list[Trip]:
        """Select the best set of trips that covers all needed items.

        Greedy: pick cheapest trip, subtract its items, repeat.
        """
        remaining = Counter(items_needed)
        selected: list[Trip] = []

        # Score trips by cost-effectiveness (cost per item)
        available = list(candidates)

        while remaining.total() > 0 and available:
            best_trip = None
            best_score = float("inf")

            for trip in available:
                # How many needed items does this trip cover?
                covers = sum(
                    min(remaining.get(s.item_type, 0), 1)
                    for s in trip.steps
                )
                if covers == 0:
                    continue
                score = trip.cost / covers
                if score < best_score:
                    best_score = score
                    best_trip = trip

            if best_trip is None:
                break

            selected.append(best_trip)
            available.remove(best_trip)

            # Subtract items covered
            for step in best_trip.steps:
                if remaining[step.item_type] > 0:
                    remaining[step.item_type] -= 1
                    if remaining[step.item_type] == 0:
                        del remaining[step.item_type]

        return selected

    def _hungarian_assign(
        self,
        bots: list[BotState],
        trips: list[Trip],
        current_round: int,
    ) -> list[ScheduledTrip]:
        """Use Hungarian algorithm to assign trips to bots optimally."""
        n_bots = len(bots)
        n_trips = len(trips)

        if n_trips == 0:
            return []

        # If more trips than bots, some bots will do multiple trips
        # Build cost matrix for single-round assignment
        cost_matrix = np.full((n_bots, n_trips), 1e9)

        for i, bot in enumerate(bots):
            for j, trip in enumerate(trips):
                c = trip_cost(bot.pos, trip.steps, trip.drop_off, self.dist_cache)
                if c is not None:
                    # Add wait time if bot isn't available yet
                    wait = max(0, bot.available_round - current_round)
                    cost_matrix[i, j] = c + wait

        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        scheduled: list[ScheduledTrip] = []
        assigned_bots: set[int] = set()

        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] >= 1e9:
                continue

            bot = bots[r]
            trip = trips[c]
            start = max(current_round, bot.available_round)
            end = start + int(cost_matrix[r, c])

            scheduled.append(ScheduledTrip(
                bot_id=bot.bot_id,
                trip=trip,
                start_round=start,
                end_round=end,
            ))

            # Update bot state for potential next assignment
            bot.pos = trip.drop_off
            bot.available_round = end
            assigned_bots.add(r)

        # Handle remaining trips (more trips than bots)
        unassigned_trips = [
            trips[j] for j in range(n_trips)
            if j not in set(col_idx)
        ]

        if unassigned_trips:
            # Recursively assign remaining trips
            more = self._hungarian_assign(bots, unassigned_trips, current_round)
            scheduled.extend(more)

        return scheduled

    def plan_full_game(
        self,
        order_queue: OrderQueue,
        bot_count: int,
        spawn: Pos,
        max_items_per_trip: int = 3,
    ) -> list[OrderPlan]:
        """Plan trips for all orders in sequence.

        Returns a list of OrderPlans, one per order.
        """
        bots = [
            BotState(bot_id=i, pos=spawn)
            for i in range(bot_count)
        ]

        plans: list[OrderPlan] = []
        current_round = 0

        for order in order_queue:
            plan = self.plan_order(
                order=order,
                bots=bots,
                current_round=current_round,
                max_items_per_trip=max_items_per_trip,
            )
            plans.append(plan)

            # Advance to when this order is estimated complete
            current_round = plan.estimated_end

        return plans
