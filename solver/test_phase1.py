"""Smoke test for Phase 1 solver components."""

from __future__ import annotations

import os
import sys

# Run from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from solver.grid import GameMap, pickup_positions
from solver.pathfinding import DistanceCache, bfs_distances
from solver.orders import OrderQueue, ShelfIndex

RECON_PATH = os.path.join(REPO_ROOT, "logs", "74001e7f_2026-03-14_recon.json")


def main() -> None:
    print("=" * 60)
    print("Phase 1 Smoke Test — Nightmare Recon")
    print("=" * 60)

    # Load map
    gm = GameMap.from_recon(RECON_PATH)
    grid = gm.grid

    walkable = grid.walkable_cells()
    print(f"\nGrid: {grid.width}x{grid.height}")
    print(f"Walls: {len(grid.walls)}")
    print(f"Shelves: {len(grid.shelves)}")
    print(f"Obstacles (walls+shelves): {len(grid.obstacles)}")
    print(f"Walkable cells: {len(walkable)}")
    print(f"Total cells: {grid.width * grid.height}")
    print(f"Bot count: {gm.bot_count}")
    print(f"Spawn: {gm.spawn}")
    print(f"Drop-off zones: {gm.drop_off_zones}")
    print(f"Item types: {len(gm.shelf_map)}")

    # Distance from spawn to each drop-off
    print("\n--- Distances from spawn to drop-off zones ---")
    dist_cache = DistanceCache(grid)
    spawn_distances = dist_cache.distance_from(gm.spawn)

    for dz in gm.drop_off_zones:
        d = spawn_distances.get(dz)
        print(f"  Spawn {gm.spawn} -> Drop-off {dz}: {d} steps")

    # Average distance from shelves to nearest drop-off
    print("\n--- Average shelf-to-nearest-dropoff distance ---")
    total_dist = 0
    count = 0
    for item_type, positions in gm.shelf_map.items():
        for shelf_pos in positions:
            pickups = pickup_positions(grid, shelf_pos)
            if not pickups:
                continue
            best = float("inf")
            for dz in gm.drop_off_zones:
                dz_map = dist_cache.distance_from(dz)
                for pp in pickups:
                    d = dz_map.get(pp)
                    if d is not None and d < best:
                        best = d
            if best < float("inf"):
                total_dist += best
                count += 1

    if count > 0:
        print(f"  Average distance: {total_dist / count:.1f} steps ({count} reachable shelves)")
    else:
        print("  No reachable shelves!")

    # Drop-off zone distances to each other
    print("\n--- Inter-dropoff distances ---")
    for i, dz1 in enumerate(gm.drop_off_zones):
        for j, dz2 in enumerate(gm.drop_off_zones):
            if i < j:
                d = dist_cache.distance(dz1, dz2)
                print(f"  {dz1} <-> {dz2}: {d} steps")

    # Order sequence
    print("\n--- First 3 orders ---")
    oq = OrderQueue.from_recon(RECON_PATH)
    print(f"Total orders: {len(oq)}")

    for i, order in enumerate(oq):
        if i >= 3:
            break
        print(f"  {order.id}: {order.items_required}")
        print(f"    ({order.item_count} items, activated round {order.activated_round})")

    # ShelfIndex test
    print("\n--- ShelfIndex sample ---")
    si = ShelfIndex(gm, dist_cache)
    sample_type = oq.active.items_required[0] if oq.active else si.item_types()[0]
    print(f"Item type '{sample_type}' nearest to each drop-off:")
    for dz in gm.drop_off_zones:
        entry = si.nearest(sample_type, dz)
        if entry:
            print(f"  Drop-off {dz}: shelf={entry.shelf_pos} pickup={entry.pickup_pos} dist={entry.distance_to_dropoff}")

    print("\n" + "=" * 60)
    print("Phase 1 smoke test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
