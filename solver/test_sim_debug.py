"""Debug: trace first 30 rounds of simulation."""

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
from solver.sim import PlanSimulator
from solver.executor import Action, BotExecState, Executor

RECON_PATH = os.path.join(REPO_ROOT, "logs", "74001e7f_2026-03-14_recon.json")


def main() -> None:
    gm = GameMap.from_recon(RECON_PATH)
    dist_cache = DistanceCache(gm.grid)
    oq = OrderQueue.from_recon(RECON_PATH)
    si = ShelfIndex(gm, dist_cache)
    tp = TripPlanner(gm, dist_cache, si)

    scheduler = Scheduler(gm, dist_cache, si, tp, max_rounds=500)
    plan = scheduler.schedule(oq, gm.bot_count, gm.spawn, max_items_per_trip=3, lookahead=0)

    # Print scheduled trips for first 3 orders
    print("=== Scheduled trips ===")
    for r in plan.order_results[:3]:
        print(f"\n{r.order.id}: {r.order.items_required}")
        for st in r.trips:
            items = [s.item_type for s in st.trip.steps]
            pickups = [s.pickup_pos for s in st.trip.steps]
            print(f"  Bot {st.bot_id}: items={items} pickups={pickups} -> {st.trip.drop_off}")

    # Manual trace of bot 0
    print("\n=== Bot 0 trace (first trip) ===")
    if plan.order_results and plan.order_results[0].trips:
        st0 = [t for t in plan.order_results[0].trips if t.bot_id == 0]
        if st0:
            trip = st0[0].trip
            print(f"Trip: {[s.item_type for s in trip.steps]}")
            for step in trip.steps:
                print(f"  Pickup {step.item_type} at shelf={step.shelf_pos} pos={step.pickup_pos}")
            print(f"  Deliver to {trip.drop_off}")

            # Compute paths
            executor = Executor(gm, dist_cache)
            current = gm.spawn
            for step in trip.steps:
                path = executor.compute_path(current, step.pickup_pos)
                print(f"  Path {current} -> {step.pickup_pos}: {len(path)} steps, path={path[:5]}...")
                current = step.pickup_pos
            path = executor.compute_path(current, trip.drop_off)
            print(f"  Path {current} -> {trip.drop_off}: {len(path)} steps")

    # Run sim with verbose first 50 rounds for bot 0
    print("\n=== Sim trace (bot 0, first 50 rounds) ===")
    oq2 = OrderQueue.from_recon(RECON_PATH)
    bots: list[BotExecState] = [
        BotExecState(bot_id=i, pos=gm.spawn)
        for i in range(gm.bot_count)
    ]

    sim = PlanSimulator(gm, dist_cache, oq2, max_rounds=500)

    # Manually assign trips
    bot_queues: dict[int, list] = {i: [] for i in range(gm.bot_count)}
    for or_result in plan.order_results:
        for st in or_result.trips:
            bot_queues[st.bot_id].append(st)
    for q in bot_queues.values():
        q.sort(key=lambda st: st.delivery_round)

    # Assign first trip
    for bot in bots:
        sim._assign_next_trip(bot, bot_queues)

    b0 = bots[0]
    print(f"Bot 0 initial: pos={b0.pos}, goal={b0.current_goal}, "
          f"target={b0.target_pos}, item={b0.target_item}, "
          f"path_len={len(b0.path)}")
    if b0.path:
        print(f"  path start: {b0.path[:5]}")

    for r in range(50):
        old_pos = b0.pos
        old_goal = b0.current_goal
        old_inv = list(b0.inventory)

        active_items = set(oq2.active.items_required) if oq2.active else set()
        actions = sim.executor.execute_round(bots, active_items)

        action = actions[0]
        if action != Action.WAIT or r < 5 or b0.pos != old_pos or b0.inventory != old_inv:
            print(f"  R{r:3d}: {action.value:8s} pos={old_pos}->{b0.pos} "
                  f"goal={old_goal} inv={len(b0.inventory)} path_left={len(b0.path)}")


if __name__ == "__main__":
    main()
