"""Smoke test for Phase 2: Trip planner and assignment engine."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from solver.grid import GameMap
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue, ShelfIndex
from solver.trips import TripPlanner, trip_cost
from solver.assignment import AssignmentEngine, BotState

RECON_PATH = os.path.join(REPO_ROOT, "logs", "74001e7f_2026-03-14_recon.json")


def main() -> None:
    print("=" * 60)
    print("Phase 2 Smoke Test — Trips & Assignment")
    print("=" * 60)

    # Load Phase 1 components
    gm = GameMap.from_recon(RECON_PATH)
    dist_cache = DistanceCache(gm.grid)
    oq = OrderQueue.from_recon(RECON_PATH)
    si = ShelfIndex(gm, dist_cache)
    tp = TripPlanner(gm, dist_cache, si)

    # Test 1: Generate trips for first order
    order = oq.active
    assert order is not None
    items = order.items_as_counter()
    print(f"\nOrder: {order.id} — {order.item_count} items: {order.items_required}")

    trips = tp.generate_trips(items, bot_pos=gm.spawn, max_items=3)
    print(f"Generated {len(trips)} trips (max 3 items each):")
    total_items = 0
    total_cost = 0
    for i, trip in enumerate(trips):
        print(f"  Trip {i}: {trip.item_types} -> {trip.drop_off} (cost={trip.cost})")
        total_items += trip.item_count
        total_cost += trip.cost
    print(f"  Total items covered: {total_items}/{order.item_count}")
    print(f"  Total movement cost: {total_cost}")

    # Test 2: Zone-aware trips
    zone_trips = tp.generate_zone_trips(items, bot_pos=gm.spawn, max_items=3)
    zone_cost = sum(t.cost for t in zone_trips)
    zone_items = sum(t.item_count for t in zone_trips)
    print(f"\nZone-aware trips: {len(zone_trips)} trips, {zone_items} items, cost={zone_cost}")
    for i, trip in enumerate(zone_trips):
        print(f"  Trip {i}: {trip.item_types} -> {trip.drop_off} (cost={trip.cost})")

    # Test 3: Assignment engine — single order
    print(f"\n--- Assignment Engine: Single Order ---")
    ae = AssignmentEngine(gm, dist_cache, si, tp)
    bots = [BotState(bot_id=i, pos=gm.spawn) for i in range(gm.bot_count)]

    plan = ae.plan_order(order, bots, current_round=0)
    print(f"Order {plan.order.id}: {plan.total_items} items in {len(plan.scheduled_trips)} trips")
    print(f"Estimated: rounds {plan.estimated_start} -> {plan.estimated_end}")
    for st in plan.scheduled_trips:
        print(f"  Bot {st.bot_id}: {st.trip.item_types} -> {st.trip.drop_off} "
              f"(rounds {st.start_round}-{st.end_round})")

    # Test 4: Full game plan
    print(f"\n--- Full Game Plan (all {len(oq)} orders) ---")
    # Reset bot states
    all_plans = ae.plan_full_game(oq, gm.bot_count, gm.spawn, max_items_per_trip=3)

    total_score = 0
    total_round = 0
    for op in all_plans:
        items_delivered = op.total_items
        order_bonus = 5 if items_delivered >= op.order.item_count else 0
        score = items_delivered + order_bonus
        total_score += score
        total_round = max(total_round, op.estimated_end)
        if op.order.id in ("order_0", "order_1", "order_2", "order_33"):
            print(f"  {op.order.id}: {items_delivered} items, "
                  f"rounds {op.estimated_start}-{op.estimated_end}, "
                  f"score +{score}")

    print(f"\nTotal estimated score: {total_score}")
    print(f"Total estimated rounds: {total_round}")
    print(f"Orders planned: {len(all_plans)}")

    # Sanity checks
    assert len(trips) > 0, "Should generate at least one trip"
    assert total_items >= order.item_count, "Should cover all items in first order"
    assert plan.estimated_end > 0, "Plan should take at least 1 round"

    print("\n" + "=" * 60)
    print("Phase 2 smoke test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
