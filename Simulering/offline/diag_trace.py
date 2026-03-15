"""
Diagnostic trace: per-round per-bot actions to find where slow orders lose time.

Usage:
    py -m Simulering.offline.diag_trace logs/31642503_2026-03-03_recon.json
    py -m Simulering.offline.diag_trace logs/31642503_2026-03-03_recon.json --detail

Output:
- Order completion timeline (round, order#, gap)
- Per-order-span breakdown: picks, delivers, moves, idles per bot
- Idle rounds and drop-off wait time
- Highlights spans >25 rounds (slow orders)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Simulering.offline.simulator import Simulator
from Simulering.offline.bot_adapter import BotAdapter


def _action_for_bot(actions: list[dict], bot_id: int) -> str:
    for a in actions:
        if isinstance(a, dict) and a.get("bot") == bot_id:
            return a.get("action", "wait")
    return "wait"


def _categorize(action: str) -> str:
    if action == "pick_up":
        return "pick"
    if action == "drop_off":
        return "deliver"
    if action and action.startswith("move_"):
        return "move"
    return "idle"


def run_trace(recon_path: str, detailed: bool = False) -> None:
    recon = json.loads(Path(recon_path).read_text(encoding="utf-8"))
    sim = Simulator.from_recon_data(recon)
    adapter = BotAdapter(suppress_logs=True)

    sim.reset()
    n_bots = len(sim._get_state().bots)

    # Per-round: (round, order_completed_this_round, {bot_id: action_category})
    round_actions: list[tuple[int, int, dict[int, str]]] = []
    orders_before = 0

    for rnd in range(1, sim.max_rounds + 1):
        state = sim._get_state()
        sd = state.to_dict()
        orders_before = sim._orders_completed

        response = adapter(sd)
        actions_list = response.get("actions", [])

        bot_cats: dict[int, str] = {}
        for bid in range(n_bots):
            act = _action_for_bot(actions_list, bid)
            bot_cats[bid] = _categorize(act)

        sim.step(actions_list)
        orders_after = sim._orders_completed
        completed_this_round = orders_after - orders_before

        round_actions.append((rnd, completed_this_round, bot_cats))

    # Build order completion timeline (rounds at which each order completed)
    completion_rounds: list[int] = []
    for rnd, completed, _ in round_actions:
        for _ in range(completed):
            completion_rounds.append(rnd)

    # Spans: from round 1 to first completion, then between completions
    spans: list[tuple[int, int, int]] = []  # (start_round, end_round, order_index)
    if completion_rounds:
        spans.append((1, completion_rounds[0], 1))
        for i in range(1, len(completion_rounds)):
            spans.append((completion_rounds[i - 1] + 1, completion_rounds[i], i + 1))
    else:
        spans.append((1, sim.max_rounds, 0))

    final_score = sim._score

    # Per-span stats: picks, delivers, moves, idles (per bot and total)
    def span_stats(start: int, end: int) -> dict:
        picks = delivers = moves = idles = 0
        by_bot: dict[int, dict[str, int]] = {b: {"pick": 0, "deliver": 0, "move": 0, "idle": 0} for b in range(n_bots)}
        for idx, (rnd, _, bot_cats) in enumerate(round_actions):
            if start <= rnd <= end:
                for bid, cat in bot_cats.items():
                    by_bot[bid][cat] += 1
                    if cat == "pick":
                        picks += 1
                    elif cat == "deliver":
                        delivers += 1
                    elif cat == "move":
                        moves += 1
                    else:
                        idles += 1
        return {
            "picks": picks,
            "delivers": delivers,
            "moves": moves,
            "idles": idles,
            "rounds": end - start + 1,
            "by_bot": by_bot,
        }

    # Output
    print("=== DIAG TRACE:", Path(recon_path).name, "===")
    print(f"Score: {final_score}  |  Bots: {n_bots}  |  Orders: {len(completion_rounds)}")
    print()

    print("Order completion timeline:")
    prev_r = 0
    for i, rnd in enumerate(completion_rounds, 1):
        gap = rnd - prev_r
        label = "TREG" if gap > 25 else "ok" if gap > 18 else "RASK"
        print(f"  R{rnd:>3d}: Order #{i:>2d} complete (gap={gap:>2d} rounds) [{label}]")
        prev_r = rnd

    total_orders = len(completion_rounds)
    avg_gap = (completion_rounds[-1] if completion_rounds else 0) / max(total_orders, 1)
    print(f"\nAverage: {avg_gap:.1f} rounds per order")
    print()

    # Slow spans detail
    print("Slow order spans (gap > 25 rounds):")
    for start, end, order_idx in spans:
        gap = end - start + 1
        if gap <= 25 or order_idx == 0:
            continue
        st = span_stats(start, end)
        print(f"  Order #{order_idx}: R{start}-R{end} ({gap} rounds)")
        print(f"    Total: pick={st['picks']} deliver={st['delivers']} move={st['moves']} idle={st['idles']}")
        for bid in range(n_bots):
            b = st["by_bot"][bid]
            print(f"    Bot {bid}: pick={b['pick']} deliver={b['deliver']} move={b['move']} idle={b['idle']}")
        idle_pct = 100 * st["idles"] / (n_bots * gap) if gap else 0
        print(f"    Idle share: {idle_pct:.0f}% of bot-rounds")
        print()

    # Summary: idle rounds overall
    total_idles = sum(
        sum(1 for c in bot_cats.values() if c == "idle")
        for _, _, bot_cats in round_actions
    )
    total_rounds_used = round_actions[-1][0] if round_actions else 0
    total_bot_rounds = n_bots * total_rounds_used
    print(f"Overall: {total_idles} idle bot-rounds of {total_bot_rounds} ({100 * total_idles / max(total_bot_rounds, 1):.0f}%)")

    if detailed:
        print("\n=== ROUND-BY-ROUND (first 100 + slow spans) ===")
        slow_starts = {s[0] for s in spans if s[1] - s[0] + 1 > 25}
        for rnd, completed, bot_cats in round_actions:
            if rnd <= 100 or (rnd - 1) in slow_starts or rnd in slow_starts:
                line = f"  R{rnd:>3d}"
                if completed:
                    line += f" [ORDER #{len([x for x in completion_rounds if x <= rnd])} DONE]"
                for bid in range(n_bots):
                    line += f" B{bid}:{bot_cats[bid][:1]}"
                print(line)


if __name__ == "__main__":
    detailed = "--detail" in sys.argv or "--detailed" in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = paths[0] if paths else "logs/31642503_2026-03-03_recon.json"
    run_trace(path, detailed=detailed)
