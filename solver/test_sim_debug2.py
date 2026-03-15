"""Trace sim execution to find delivery failures."""

from __future__ import annotations
import os, sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from collections import Counter
from solver.grid import GameMap
from solver.pathfinding import DistanceCache
from solver.orders import OrderQueue, ShelfIndex
from solver.trips import TripPlanner
from solver.scheduler import Scheduler
from solver.sim import Simulator, SimBot, Action

RECON_PATH = os.path.join(REPO_ROOT, "logs", "74001e7f_2026-03-14_recon.json")


def main():
    gm = GameMap.from_recon(RECON_PATH)
    dist_cache = DistanceCache(gm.grid)
    oq = OrderQueue.from_recon(RECON_PATH)
    si = ShelfIndex(gm, dist_cache)
    tp = TripPlanner(gm, dist_cache, si)

    scheduler = Scheduler(gm, dist_cache, si, tp, max_rounds=500)
    plan = scheduler.schedule(oq, gm.bot_count, gm.spawn, max_items_per_trip=3)

    # Run sim manually to trace
    orders = list(OrderQueue.from_recon(RECON_PATH))
    sim = Simulator(gm, dist_cache, OrderQueue.from_recon(RECON_PATH), max_rounds=100)

    bots = [SimBot(bot_id=i, pos=gm.spawn) for i in range(gm.bot_count)]

    bot_queues = {i: [] for i in range(gm.bot_count)}
    for or_result in plan.order_results:
        for st in or_result.trips:
            bot_queues[st.bot_id].append(st)
    for q in bot_queues.values():
        q.sort(key=lambda st: st.delivery_round)

    for bot in bots:
        sim._assign_next_trip(bot, bot_queues)

    active_idx = 0
    items_remaining = Counter(orders[0].items_as_counter())

    print(f"Order 0: {orders[0].items_required}")
    print(f"Items needed: {dict(items_remaining)}")
    print(f"\nBots with trips:")
    for bot in bots:
        if bot.goal != "idle":
            print(f"  Bot {bot.bot_id}: goal={bot.goal}, item={bot.pending_pickup_item}, "
                  f"path_len={len(bot.path)}, target={bot.path[-1] if bot.path else 'at_target'}")

    score = 0
    for r in range(100):
        actions = sim._decide_actions(bots)
        new_pos = sim._resolve_moves(bots, actions)

        for bot in bots:
            action = actions[bot.bot_id]
            np = new_pos[bot.bot_id]

            if action == Action.PICK_UP and bot.goal == "do_pickup":
                if bot.pending_pickup_item:
                    bot.inventory.append(bot.pending_pickup_item)
                    bot.pending_pickup_item = None
                    print(f"R{r:3d}: Bot {bot.bot_id} PICKUP {bot.inventory[-1]} at {bot.pos} inv={bot.inventory}")

                    if bot.remaining_steps:
                        step = bot.remaining_steps.pop(0)
                        bot.pending_pickup_item = step.item_type
                        bot.path = sim._compute_path(bot.pos, step.pickup_pos)
                        bot.goal = "move_to_pickup"
                    else:
                        bot.path = sim._compute_path(bot.pos, bot.trip_dropoff)
                        bot.dropoff_target = bot.trip_dropoff
                        bot.goal = "move_to_dropoff"

            elif action == Action.DROP_OFF and bot.goal == "do_dropoff":
                if bot.inventory and active_idx < len(orders):
                    to_keep = []
                    delivered = 0
                    for item in bot.inventory:
                        if items_remaining[item] > 0:
                            items_remaining[item] -= 1
                            delivered += 1
                        else:
                            to_keep.append(item)

                    print(f"R{r:3d}: Bot {bot.bot_id} DROPOFF at {bot.pos} "
                          f"delivered={delivered} kept={to_keep} "
                          f"remaining={dict({k:v for k,v in items_remaining.items() if v>0})}")

                    score += delivered
                    bot.inventory = to_keep

                    if items_remaining.total() == 0:
                        score += 5
                        active_idx += 1
                        print(f"      ORDER {active_idx-1} COMPLETE! score={score}")
                        if active_idx < len(orders):
                            items_remaining = Counter(orders[active_idx].items_as_counter())
                            print(f"      Next order: {orders[active_idx].items_required}")

                bot.goal = "idle"
                sim._assign_next_trip(bot, bot_queues)
                if bot.goal != "idle":
                    print(f"      Bot {bot.bot_id} next trip: item={bot.pending_pickup_item}")

            else:
                if np != bot.pos:
                    bot.pos = np
                    if bot.path and bot.path[0] == np:
                        bot.path.pop(0)
                    if bot.goal == "move_to_pickup" and not bot.path:
                        bot.goal = "do_pickup"
                    elif bot.goal == "move_to_dropoff" and not bot.path:
                        bot.goal = "do_dropoff"

        for bot in bots:
            if bot.goal == "idle":
                sim._assign_next_trip(bot, bot_queues)

    print(f"\nFinal score at round 100: {score}")


if __name__ == "__main__":
    main()
