"""Test reactive simulator."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from solver.grid import GameMap
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue, ShelfIndex, Order
from solver.trips import TripPlanner
from solver.sim import ReactiveSim

RECON_PATH = os.path.join(REPO_ROOT, "logs", "74001e7f_2026-03-14_recon.json")


def main() -> None:
    print("=" * 60)
    print("Reactive Sim Test")
    print("=" * 60)

    gm = GameMap.from_recon(RECON_PATH)
    dist_cache = DistanceCache(gm.grid)
    oq = OrderQueue.from_recon(RECON_PATH)
    si = ShelfIndex(gm, dist_cache)
    orders = list(oq)

    # Test with different batch sizes
    for max_items in [1, 2, 3]:
        sim = ReactiveSim(gm, dist_cache, si, orders, max_rounds=500)
        result = sim.run(max_items_per_trip=max_items)
        print(f"\nmax_items={max_items}: score={result.score}, items={result.items_delivered}, "
              f"orders={result.orders_completed}, collisions={result.collisions}")

    # Verbose run with best config
    print(f"\n{'='*60}")
    print("Verbose run (max_items=1)")
    print(f"{'='*60}")
    sim = ReactiveSim(gm, dist_cache, si, orders, max_rounds=500)
    result = sim.run(max_items_per_trip=1, verbose=True)

    print(f"\nFinal: score={result.score}, items={result.items_delivered}, "
          f"orders={result.orders_completed}")


if __name__ == "__main__":
    main()
