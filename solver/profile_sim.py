"""Profile a sim run: where does each bot spend its time?"""

from __future__ import annotations
import json, logging, sys, os
from pathlib import Path
from collections import defaultdict

os.environ["PYTHONUNBUFFERED"] = "1"
logging.basicConfig(level=logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.config import CoordinatorConfig
from Simulering.offline.simulator import Simulator
from Simulering.offline.bot_adapter import BotAdapter


def profile(recon_path: str, config_path: str | None = None):
    if config_path:
        cfg = CoordinatorConfig.from_dict(json.loads(Path(config_path).read_text()))
    else:
        cfg = CoordinatorConfig.nightmare()

    # Patch BotAdapter to intercept actions
    bot_actions = defaultdict(list)
    bot_positions = defaultdict(list)
    order_events = []

    class ProfilingAdapter(BotAdapter):
        def __call__(self, game_state: dict) -> dict:
            rnd = game_state.get("round", 0)
            # Track positions
            for bot in game_state.get("bots", []):
                bot_positions[bot["id"]].append(tuple(bot["position"]))
            # Track orders
            for order in game_state.get("orders", []):
                if order.get("status") == "active" and order.get("complete"):
                    pass  # completed this round

            response = super().__call__(game_state)

            for a in response.get("actions", []):
                bid = a.get("bot", a.get("bot_id", -1))
                act = a.get("action", "wait")
                bot_actions[bid].append(act)

            return response

    adapter = ProfilingAdapter(suppress_logs=True, config=cfg)
    sim = Simulator.from_recon_file(recon_path)
    result = sim.run(adapter)

    score = result["score"]
    items = result["items_delivered"]
    orders = result["orders_completed"]
    rounds = result["rounds_used"]

    print(f"Final: score={score}, items={items}, orders={orders}, rounds={rounds}")

    n_bots = max(bot_actions.keys()) + 1 if bot_actions else 0

    move_actions = {"move_up", "move_down", "move_left", "move_right"}

    print(f"\n{'='*60}")
    print("PER-BOT TIME BREAKDOWN")
    print(f"{'='*60}")

    total_move = 0
    total_wait = 0
    total_stuck = 0
    total_pick = 0
    total_drop = 0

    for bid in range(n_bots):
        actions = bot_actions[bid]
        positions = bot_positions[bid]

        move = sum(1 for a in actions if a in move_actions)
        pick = sum(1 for a in actions if a == "pick_up")
        drop = sum(1 for a in actions if a == "drop_off")
        wait = sum(1 for a in actions if a == "wait")

        # Stuck = sent move but didn't move
        stuck = 0
        for i in range(1, min(len(positions), len(actions)+1)):
            if i < len(positions) and positions[i] == positions[i-1] and i-1 < len(actions) and actions[i-1] in move_actions:
                stuck += 1

        total_move += move
        total_wait += wait
        total_stuck += stuck
        total_pick += pick
        total_drop += drop

        pct = move * 100 // max(1, move + wait)
        print(f"  B{bid:2d}: move={move:3d} pick={pick:2d} drop={drop:2d} wait={wait:3d} stuck={stuck:3d} ({pct}% active)")

    print(f"\n  TOTAL across {n_bots} bots, {rounds} rounds:")
    print(f"    Moving: {total_move} ({total_move*100//(n_bots*rounds)}%)")
    print(f"    Waiting: {total_wait} ({total_wait*100//(n_bots*rounds)}%)")
    print(f"    Stuck (blocked): {total_stuck} ({total_stuck*100//max(1,total_move)}% of moves)")
    print(f"    Pickups: {total_pick}")
    print(f"    Dropoffs: {total_drop}")

    # Score velocity
    print(f"\n{'='*60}")
    print("SCORE VELOCITY")
    print(f"{'='*60}")
    print(f"  {score} score / {rounds} rounds = {score/rounds:.2f} score/round")
    print(f"  {orders} orders / {rounds} rounds = {orders/rounds:.3f} orders/round")
    print(f"  Need: 0.14 orders/round for 1300+ in 500 rounds")
    print(f"  Current efficiency: {orders/rounds*100/0.14:.0f}% of target")

    # Wasted capacity
    bot_rounds = n_bots * rounds
    useful = total_move + total_pick + total_drop
    wasted = total_wait + total_stuck
    print(f"\n  Bot-rounds: {bot_rounds}")
    print(f"  Useful: {useful} ({useful*100//bot_rounds}%)")
    print(f"  Wasted: {wasted} ({wasted*100//bot_rounds}%)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--recon", required=True)
    p.add_argument("--config", default=None)
    args = p.parse_args()
    profile(args.recon, args.config)
