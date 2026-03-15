"""CLI runner for the solver.

Usage:
    py -m solver.run --recon logs/74001e7f_2026-03-14_recon.json
    py -m solver.run --recon logs/74001e7f_2026-03-14_recon.json --max-items 2
"""

from __future__ import annotations

import argparse
import sys
import time

from .grid import GameMap
from .pathfinding import DistanceCache
from .orders import OrderQueue, ShelfIndex
from .sim import ReactiveSim


def main() -> None:
    parser = argparse.ArgumentParser(description="Solver for NM i AI Grocery Bot")
    parser.add_argument("--recon", required=True, help="Path to recon JSON file")
    parser.add_argument("--max-items", type=int, default=2, help="Max items per trip (default: 2)")
    parser.add_argument("--max-rounds", type=int, default=500, help="Max simulation rounds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed output")
    args = parser.parse_args()

    print(f"Loading recon: {args.recon}")
    t0 = time.time()

    gm = GameMap.from_recon(args.recon)
    dist_cache = DistanceCache(gm.grid)
    oq = OrderQueue.from_recon(args.recon)
    si = ShelfIndex(gm, dist_cache)

    t_load = time.time() - t0
    print(f"  Grid: {gm.grid.width}x{gm.grid.height}, {gm.bot_count} bots")
    print(f"  Orders: {len(oq)}, Items: {sum(o.item_count for o in oq)}")
    print(f"  Drop-off zones: {gm.drop_off_zones}")
    print(f"  Load time: {t_load:.2f}s")

    print(f"\nRunning solver (max_items={args.max_items}, max_rounds={args.max_rounds})...")
    t1 = time.time()

    sim = ReactiveSim(gm, dist_cache, si, list(oq), max_rounds=args.max_rounds)
    result = sim.run(max_items_per_trip=args.max_items, verbose=args.verbose)

    t_run = time.time() - t1

    print(f"\n{'='*50}")
    print(f"Score:      {result.score}")
    print(f"Items:      {result.items_delivered}")
    print(f"Orders:     {result.orders_completed}/{len(oq)}")
    print(f"Rounds:     {result.rounds_used}/{args.max_rounds}")
    print(f"Collisions: {result.collisions}")
    print(f"Time:       {t_run:.2f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
