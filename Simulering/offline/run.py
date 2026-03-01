"""
Offline optimization pipeline for Grocery Bot.

Usage:
    # Step 1: Analyze a game log
    python -m offline.run analyze logs/game_log_2026-03-01_score95.json

    # Step 2: Check determinism (compare two logs from same day)
    python -m offline.run check-determinism logs/run1.json logs/run2.json

    # Step 3: Generate optimal plan from a game log
    python -m offline.run plan logs/game_log_2026-03-01_score95.json

    # Step 4: Generate plan for specific difficulty
    python -m offline.run plan logs/game_log.json --bots 3
"""
from __future__ import annotations
import json
import sys
import os
from .grid import Grid
from .analyzer import analyze, load_log, check_determinism
from .planner import OfflinePlanner
from .replay import save_plan


def cmd_analyze(log_path: str):
    """Analyze a game log."""
    log = load_log(log_path)
    results = analyze(log, verbose=True)
    return results


def cmd_check_determinism(path1: str, path2: str):
    """Compare two logs for determinism."""
    return check_determinism(path1, path2)


def cmd_plan(log_path: str, num_bots: int = None):
    """Generate optimal plan from game log."""
    log = load_log(log_path)
    results = analyze(log, verbose=True)

    # Build grid from first round
    r0 = log["rounds"][0]
    grid = Grid.from_game_state(r0)
    print(f"Grid: {grid.width}x{grid.height}, "
          f"{len(grid.walls)} walls, {len(grid.shelves)} shelves")

    # Extract key info
    drop_off = tuple(r0["drop_off"])
    bots = r0["bots"]
    n_bots = num_bots or len(bots)
    spawn_positions = [tuple(b["position"]) for b in bots]

    type_positions = {
        t: [tuple(p) for p in ps]
        for t, ps in results["type_shelf_positions"].items()
    }

    order_seq = [
        {"id": o["id"], "items_required": o["items"]}
        for o in results["order_sequence"]
    ]

    print(f"\nPlanning for {n_bots} bot(s), {len(order_seq)} orders, "
          f"drop-off at {drop_off}")
    print(f"Spawn: {spawn_positions}")

    # Precompute BFS
    print("Precomputing BFS distances...", end=" ", flush=True)
    grid.precompute_all_bfs()
    print("done.")

    # Plan
    planner = OfflinePlanner(grid, drop_off, spawn_positions)

    if n_bots == 1:
        plans = planner.plan_full_game(order_seq, type_positions, bot_index=0)
        actions = planner.to_action_sequence(plans, bot_id=0)

        # Compute fingerprint for file naming
        import hashlib
        fp_data = f"{grid.width}x{grid.height}_{sorted(grid.walls)}"
        fingerprint = hashlib.sha256(fp_data.encode()).hexdigest()[:12]

        save_plan(actions, fingerprint, metadata={
            "score_estimate": sum(
                len(b.items) for p in plans for b in p.batches
            ) + sum(1 for p in plans if p.batches) * 5,
            "num_orders": sum(1 for p in plans if p.batches),
            "num_items": sum(
                len(b.items) for p in plans for b in p.batches
            ),
        })

    else:
        all_actions = planner.plan_multi_bot(
            order_seq, type_positions, n_bots
        )
        # Save each bot's plan
        import hashlib
        fp_data = f"{grid.width}x{grid.height}_{sorted(grid.walls)}"
        fingerprint = hashlib.sha256(fp_data.encode()).hexdigest()[:12]

        for bot_id, actions in all_actions.items():
            save_plan(actions, f"{fingerprint}_bot{bot_id}", metadata={
                "bot_id": bot_id,
                "num_actions": len(actions),
            })

    # --- Detailed order-by-order breakdown ---
    if n_bots == 1:
        print("\n=== ORDER-BY-ORDER BREAKDOWN ===")
        cumulative = 0
        for i, plan in enumerate(plans):
            if not plan.batches:
                xo = "from cross-order auto-delivery!"
                print(f"  Order {i:2d} ({plan.order_id}): "
                      f"0 rounds — {xo}")
                continue

            items = sum(len(b.items) for b in plan.batches)
            score = items + 5
            cumulative += plan.total_cost
            xo_str = ""
            if plan.cross_order_items:
                xo_types = [x.item_type for x in plan.cross_order_items]
                xo_str = f" + cross-order: {xo_types}"

            print(f"  Order {i:2d} ({plan.order_id}): "
                  f"{plan.total_cost:3d} rounds, {items} items, "
                  f"+{score} score{xo_str}")
            for j, batch in enumerate(plan.batches):
                types = [item.item_type for item in batch.items]
                print(f"         Batch {j}: {types} ({batch.cost}r)")

        print(f"\n  Total: {cumulative} rounds used of 300")


def cmd_compare_with_reactive(log_path: str):
    """
    Compare reactive bot performance with offline optimal estimate.
    """
    log = load_log(log_path)
    results = analyze(log, verbose=False)

    reactive_score = results["final_score"]
    reactive_orders = sum(1 for d in results["deliveries"]
                          if d["score_delta"] >= 6)  # 5+1 minimum for order completion

    # Estimate offline optimal
    r0 = log["rounds"][0]
    grid = Grid.from_game_state(r0)
    grid.precompute_all_bfs()

    drop_off = tuple(r0["drop_off"])
    spawn = [tuple(b["position"]) for b in r0["bots"]]

    type_positions = {
        t: [tuple(p) for p in ps]
        for t, ps in results["type_shelf_positions"].items()
    }
    order_seq = [
        {"id": o["id"], "items_required": o["items"]}
        for o in results["order_sequence"]
    ]

    planner = OfflinePlanner(grid, drop_off, spawn)
    plans = planner.plan_full_game(order_seq, type_positions)

    offline_items = sum(len(b.items) for p in plans for b in p.batches)
    offline_orders = sum(1 for p in plans if p.batches)
    offline_score = offline_items + offline_orders * 5

    print(f"\n{'='*50}")
    print(f"REACTIVE vs OFFLINE COMPARISON")
    print(f"{'='*50}")
    print(f"  Reactive:  {reactive_score:3d} points")
    print(f"  Offline:   {offline_score:3d} points (estimate)")
    print(f"  Delta:     +{offline_score - reactive_score} points")
    print(f"  Improvement: {(offline_score/reactive_score - 1)*100:.0f}%")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "analyze" and len(sys.argv) >= 3:
        cmd_analyze(sys.argv[2])

    elif cmd == "check-determinism" and len(sys.argv) >= 4:
        cmd_check_determinism(sys.argv[2], sys.argv[3])

    elif cmd == "plan" and len(sys.argv) >= 3:
        bots = None
        if "--bots" in sys.argv:
            idx = sys.argv.index("--bots")
            bots = int(sys.argv[idx + 1])
        cmd_plan(sys.argv[2], num_bots=bots)

    elif cmd == "compare" and len(sys.argv) >= 3:
        cmd_compare_with_reactive(sys.argv[2])

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
