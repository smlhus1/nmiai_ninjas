"""Test parallel scheduler with pipelining."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from solver.grid import GameMap
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue, ShelfIndex
from solver.trips import TripPlanner
from solver.scheduler import Scheduler

RECON_PATH = os.path.join(REPO_ROOT, "logs", "74001e7f_2026-03-14_recon.json")


def main() -> None:
    print("=" * 60)
    print("Phase 2b — Pipelined Scheduler Test")
    print("=" * 60)

    gm = GameMap.from_recon(RECON_PATH)
    dist_cache = DistanceCache(gm.grid)
    oq = OrderQueue.from_recon(RECON_PATH)
    si = ShelfIndex(gm, dist_cache)
    tp = TripPlanner(gm, dist_cache, si)

    # Test different lookahead values
    for lookahead in [0, 1, 2, 4]:
        scheduler = Scheduler(gm, dist_cache, si, tp, max_rounds=500)
        plan = scheduler.schedule(oq, gm.bot_count, gm.spawn, max_items_per_trip=3, lookahead=lookahead)

        active_bots = sum(1 for c in plan.trips_per_bot.values() if c > 0)
        total_trips = sum(plan.trips_per_bot.values())

        print(f"\nLookahead={lookahead}: score={plan.total_score}, "
              f"orders={plan.orders_completed}/{len(plan.order_results)}, "
              f"rounds={plan.total_rounds}, "
              f"bots={active_bots}/{gm.bot_count}, trips={total_trips}")

    # Detailed view of best config
    print(f"\n{'='*60}")
    print("Detailed: lookahead=2")
    print(f"{'='*60}")

    scheduler = Scheduler(gm, dist_cache, si, tp, max_rounds=500)
    plan = scheduler.schedule(oq, gm.bot_count, gm.spawn, max_items_per_trip=3, lookahead=2)

    for r in plan.order_results:
        done = "OK" if r.items_count >= r.order.item_count else "PARTIAL"
        over = " OVER" if r.all_delivered_round > 500 else ""
        print(f"  {r.order.id}: {r.items_count}/{r.order.item_count} items, "
              f"done@{r.all_delivered_round}{over} [{done}]")

    # Bot utilization
    print(f"\nBot trips: {dict(sorted(plan.trips_per_bot.items()))}")

    # Zone distribution
    zone_counts: dict = {}
    for r in plan.order_results:
        for st in r.trips:
            dz = st.trip.drop_off
            zone_counts[dz] = zone_counts.get(dz, 0) + 1
    print(f"Zone distribution: {dict(sorted(zone_counts.items()))}")


if __name__ == "__main__":
    main()
