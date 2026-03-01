"""Round-by-round trace of bot behavior. Shows exactly where time goes."""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from Simulering.offline.simulator import Simulator
from Simulering.offline.bot_adapter import BotAdapter


def trace(recon_path: str, detailed: bool = False) -> None:
    recon = json.loads(Path(recon_path).read_text(encoding="utf-8"))
    sim = Simulator.from_recon_data(recon)
    adapter = BotAdapter(suppress_logs=True)

    sim.reset()

    trip_start = 1
    deliveries = []
    order_idx = 0
    round_log = []

    for rnd in range(1, sim.max_rounds + 1):
        state = sim._get_state()
        sd = state.to_dict()

        bot = state.bots[0]
        pos = tuple(bot.position)
        inv = list(bot.inventory)
        score = state.score

        active_orders = [o for o in state.orders if o.status == "active"]
        active_remaining = active_orders[0].items_remaining if active_orders else []

        response = adapter(sd)
        actions_list = response.get("actions", [])

        act_type = "wait"
        for a in actions_list:
            if isinstance(a, dict) and a.get("bot") == bot.id:
                act_type = a.get("action", "wait")
                break

        sim.step(actions_list)
        new_state = sim._get_state()
        new_bot = new_state.bots[0]
        new_pos = tuple(new_bot.position)
        new_inv = list(new_bot.inventory)
        new_score = new_state.score

        if detailed:
            if act_type == "pick_up":
                status = f"PICK inv={new_inv}"
            elif act_type == "drop_off":
                status = f"DROP {inv} -> {new_inv}"
            elif new_pos == pos:
                status = f"STUCK ({act_type})"
            else:
                status = act_type
            round_log.append(
                f"  R{rnd:3d} {pos}->{new_pos} {status:40s} remain={list(active_remaining)}"
            )

        if new_score > score:
            delta = new_score - score
            items_delivered = max(0, len(inv) - len(new_inv))
            if items_delivered == 0 and delta > 0:
                items_delivered = delta if delta <= 5 else delta - 5
            order_bonus = delta - items_delivered
            deliveries.append({
                "round": rnd,
                "items_delivered": items_delivered,
                "trip_rounds": rnd - trip_start + 1,
                "order_bonus": order_bonus,
                "order_idx": order_idx,
            })
            if order_bonus >= 5:
                order_idx += 1
            trip_start = rnd + 1

    final_score = new_state.score
    print(f"=== TRACE: {Path(recon_path).name} ===")
    print(f"Score: {final_score} | Grid: {recon['grid_size']} | Bots: {recon['bot_count']}")
    print(f"Orders in recon: {len(recon['order_sequence'])}")
    print()

    total_items = sum(d["items_delivered"] for d in deliveries)
    orders_done = sum(1 for d in deliveries if d["order_bonus"] >= 5)

    print(f"Items delivered: {total_items}")
    print(f"Orders completed: {orders_done}")
    print(f"Trips to drop-off: {len(deliveries)}")
    print()

    print(f"{'#':>3} {'Rnd':>4} {'Items':>5} {'Order':>6} {'Len':>4} {'R/I':>5}")
    print("-" * 35)
    for i, d in enumerate(deliveries):
        rpi = d["trip_rounds"] / max(d["items_delivered"], 1)
        bonus = f"done#{d['order_idx']}" if d["order_bonus"] >= 5 else ""
        print(f"{i+1:3d} {d['round']:4d} {d['items_delivered']:5d} {bonus:>6} {d['trip_rounds']:4d} {rpi:5.1f}")

    print()
    one_item = [d for d in deliveries if d["items_delivered"] == 1]
    if one_item:
        wasted = sum(d["trip_rounds"] for d in one_item)
        print(f"1-ITEM TRIPS: {len(one_item)} trips using {wasted} rounds")

    slow = [d for d in deliveries if d["trip_rounds"] / max(d["items_delivered"], 1) > 10]
    if slow:
        print(f"SLOW TRIPS (>10 r/i): {len(slow)}")

    if deliveries:
        used = deliveries[-1]["round"]
        avg_rpi = used / max(total_items, 1)
        print(f"\nRounds used: {used}/300  |  Avg r/item: {avg_rpi:.1f}")
        print(f"At 5.0 r/item: ~{int(300/5)} items ~{int(300/5)//3} orders = ~{int(300/5) + int(300/5)//3*5} pts")

    if detailed:
        print(f"\n=== ROUND-BY-ROUND ===")
        for line in round_log:
            print(line)


if __name__ == "__main__":
    detailed = "--detail" in sys.argv
    path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    trace(path, detailed=detailed)
