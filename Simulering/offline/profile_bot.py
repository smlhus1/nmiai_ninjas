"""
Profile the live bot: run simulation and identify optimization opportunities.

Tracks per-round per-bot behavior to find wasted time, bottlenecks,
and specific improvement targets.

Usage (from project root):
    py -m Simulering.offline.profile_bot --recon logs/6fb8097b_2026-03-01_recon.json
    py -m Simulering.offline.profile_bot --scenario easy
    py -m Simulering.offline.profile_bot --scenario medium
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Simulering.offline.simulator import Simulator
from Simulering.offline.bot_adapter import BotAdapter

Pos = tuple[int, int]


class ProfilingAdapter(BotAdapter):
    """Wraps BotAdapter to collect per-round per-bot telemetry."""

    def __init__(self):
        super().__init__(suppress_logs=True)
        self.round_data: list[dict] = []
        self.action_counts: Counter = Counter()
        self.task_type_rounds: Counter = Counter()
        self.bot_idle_rounds: Counter = Counter()
        self.order_completion_rounds: list[tuple[str, int]] = []
        self.pickup_rounds: list[tuple[int, str, int]] = []  # (bot_id, type, round)
        self.delivery_rounds: list[tuple[int, int]] = []  # (bot_id, round)
        self._prev_score = 0
        self._prev_orders_status: dict[str, str] = {}
        self._prev_inventories: dict[int, list[str]] = {}

    def __call__(self, state_dict):
        response = super().__call__(state_dict)
        rnd = state_dict["round"]
        score = state_dict.get("score", 0)
        bots = state_dict["bots"]
        actions = response.get("actions", [])
        orders = state_dict["orders"]

        action_map = {a.get("bot", -1): a.get("action", "wait") for a in actions}
        coord = self._coordinator

        rd = {
            "round": rnd,
            "score": score,
            "score_delta": score - self._prev_score,
            "bots": {},
        }

        for bot_data in bots:
            bid = bot_data["id"]
            pos = tuple(bot_data["position"])
            inv = bot_data["inventory"]
            action = action_map.get(bid, "wait")

            task_type = "NONE"
            task_target = None
            if coord:
                asgn = coord._assignments.get(bid)
                if asgn and asgn.task:
                    task_type = asgn.task.task_type.name
                    task_target = asgn.task.target_pos

            self.action_counts[action] += 1
            self.task_type_rounds[(bid, task_type)] += 1

            if task_type == "IDLE":
                self.bot_idle_rounds[bid] += 1

            # Detect pickups via inventory change
            prev_inv = self._prev_inventories.get(bid, [])
            if len(inv) > len(prev_inv):
                new_items = list(inv)
                for item in prev_inv:
                    if item in new_items:
                        new_items.remove(item)
                for item_type in new_items:
                    self.pickup_rounds.append((bid, item_type, rnd))

            # Detect deliveries via inventory shrink + score increase
            if len(inv) < len(prev_inv) and score > self._prev_score:
                self.delivery_rounds.append((bid, rnd))

            rd["bots"][bid] = {
                "pos": pos, "inv": inv, "action": action,
                "task": task_type, "target": task_target,
            }
            self._prev_inventories[bid] = list(inv)

        # Track order completions
        for o in orders:
            oid = o["id"]
            prev_status = self._prev_orders_status.get(oid)
            if prev_status and prev_status != "complete" and o.get("complete"):
                self.order_completion_rounds.append((oid, rnd))
            self._prev_orders_status[oid] = o.get("status", "active")

        self.round_data.append(rd)
        self._prev_score = score
        return response


def analyze(adapter: ProfilingAdapter, result: dict) -> dict:
    """Analyze profiling data and return optimization report."""
    total_rounds = result["rounds_used"]
    num_bots = len(adapter.round_data[0]["bots"]) if adapter.round_data else 1
    total_bot_rounds = total_rounds * num_bots

    # Task type distribution per bot
    task_dist = defaultdict(lambda: Counter())
    for (bid, ttype), count in adapter.task_type_rounds.items():
        task_dist[bid][ttype] = count

    # Scoring rate over time (divide game into 6 phases)
    phase_size = max(total_rounds // 6, 1)
    phases = []
    for i in range(0, total_rounds, phase_size):
        end = min(i + phase_size, total_rounds)
        phase_rounds = adapter.round_data[i:end]
        if not phase_rounds:
            continue
        score_start = phase_rounds[0]["score"]
        score_end = phase_rounds[-1]["score"]
        phases.append({
            "rounds": f"R{i}-R{end}",
            "score_gained": score_end - score_start,
            "rate": (score_end - score_start) / (end - i) if end > i else 0,
        })

    # Wait/idle analysis
    wait_count = adapter.action_counts.get("wait", 0)
    total_actions = sum(adapter.action_counts.values())
    wait_pct = (wait_count / total_actions * 100) if total_actions else 0

    idle_total = sum(adapter.bot_idle_rounds.values())
    idle_pct = (idle_total / total_bot_rounds * 100) if total_bot_rounds else 0

    # Average pickup-to-delivery cycle time
    cycle_times = []
    for bid in range(num_bots):
        bot_pickups = [(t, r) for b, t, r in adapter.pickup_rounds if b == bid]
        bot_deliveries = [r for b, r in adapter.delivery_rounds if b == bid]
        for dr in bot_deliveries:
            # Find most recent pickup before this delivery
            prev_picks = [r for _, r in bot_pickups if r < dr]
            if prev_picks:
                cycle_times.append(dr - max(prev_picks))

    avg_cycle = sum(cycle_times) / len(cycle_times) if cycle_times else 0

    # Order completion gaps
    order_gaps = []
    prev_round = 0
    for oid, rnd in sorted(adapter.order_completion_rounds, key=lambda x: x[1]):
        order_gaps.append({"order": oid, "round": rnd, "gap": rnd - prev_round})
        prev_round = rnd

    # Congestion: rounds where multiple bots are within 2 cells of drop-off
    congestion_rounds = 0
    if adapter.round_data:
        drop_off = None
        for rd in adapter.round_data:
            bots_near_dropoff = 0
            for bid, bdata in rd["bots"].items():
                if bdata["task"] == "DELIVER":
                    bots_near_dropoff += 1
            if bots_near_dropoff >= 2:
                congestion_rounds += 1

    return {
        "total_rounds": total_rounds,
        "num_bots": num_bots,
        "final_score": result["score"],
        "items_delivered": result["items_delivered"],
        "orders_completed": result["orders_completed"],
        "action_counts": dict(adapter.action_counts),
        "wait_pct": wait_pct,
        "idle_pct": idle_pct,
        "idle_per_bot": dict(adapter.bot_idle_rounds),
        "task_dist": {bid: dict(counts) for bid, counts in task_dist.items()},
        "phases": phases,
        "avg_cycle_time": avg_cycle,
        "order_gaps": order_gaps,
        "congestion_rounds": congestion_rounds,
        "total_pickups": len(adapter.pickup_rounds),
        "total_deliveries": len(adapter.delivery_rounds),
    }


def print_report(report: dict) -> None:
    """Print human-readable optimization report."""
    print("=" * 65)
    print(f"  PROFILING REPORT — {report['num_bots']} bots, "
          f"{report['total_rounds']} rounds")
    print("=" * 65)

    print(f"\n  Score: {report['final_score']}  "
          f"({report['items_delivered']} items, "
          f"{report['orders_completed']} orders)")
    print(f"  Pickups: {report['total_pickups']}  "
          f"Deliveries: {report['total_deliveries']}  "
          f"Avg cycle: {report['avg_cycle_time']:.1f} rounds")

    print(f"\n--- Efficiency ---")
    print(f"  Wait actions: {report['wait_pct']:.1f}% of all actions")
    print(f"  IDLE task rounds: {report['idle_pct']:.1f}% of bot-rounds")
    for bid, idle in sorted(report["idle_per_bot"].items()):
        pct = idle / report["total_rounds"] * 100
        print(f"    Bot {bid}: {idle} idle rounds ({pct:.0f}%)")

    print(f"\n--- Task distribution (rounds per task type) ---")
    for bid in sorted(report["task_dist"]):
        counts = report["task_dist"][bid]
        parts = [f"{t}={c}" for t, c in sorted(counts.items(), key=lambda x: -x[1])]
        print(f"  Bot {bid}: {', '.join(parts)}")

    print(f"\n--- Scoring rate by phase ---")
    for phase in report["phases"]:
        bar = "#" * int(phase["rate"] * 20)
        print(f"  {phase['rounds']:>10s}: +{phase['score_gained']:3d}  "
              f"({phase['rate']:.2f}/round) {bar}")

    print(f"\n--- Order completion ---")
    for og in report["order_gaps"]:
        print(f"  {og['order']:>10s} completed at R{og['round']:3d}  "
              f"(gap: {og['gap']} rounds)")

    print(f"\n--- Delivery congestion ---")
    print(f"  Rounds with 2+ bots delivering: {report['congestion_rounds']}")

    # Optimization suggestions
    print(f"\n{'=' * 65}")
    print(f"  OPTIMIZATION OPPORTUNITIES")
    print(f"{'=' * 65}")

    suggestions = []

    if report["wait_pct"] > 15:
        suggestions.append(
            f"HIGH WAIT RATE ({report['wait_pct']:.0f}%): Bots spend too much time "
            f"waiting. PIBT collision resolution or corridor navigation may need "
            f"improvement.")

    if report["idle_pct"] > 10:
        suggestions.append(
            f"HIGH IDLE RATE ({report['idle_pct']:.0f}%): Bots lack useful tasks. "
            f"Consider more aggressive preview pre-picking or wider task search.")

    if report["avg_cycle_time"] > 20:
        suggestions.append(
            f"SLOW CYCLES ({report['avg_cycle_time']:.0f} rounds avg): Pickup-to-"
            f"delivery takes too long. Route optimization or closer item selection "
            f"could help.")

    if report["congestion_rounds"] > 30:
        suggestions.append(
            f"DELIVERY CONGESTION ({report['congestion_rounds']} rounds): Multiple "
            f"bots crowd drop-off. Better staging or staggered delivery timing needed.")

    # Check for scoring plateaus
    for i, phase in enumerate(report["phases"]):
        if phase["score_gained"] == 0 and i < len(report["phases"]) - 1:
            suggestions.append(
                f"SCORING PLATEAU at {phase['rounds']}: Zero points gained. "
                f"Bots may be deadlocked or working on wrong tasks.")

    # Check for uneven bot utilization
    idle_counts = report["idle_per_bot"]
    if idle_counts:
        max_idle = max(idle_counts.values())
        min_idle = min(idle_counts.values()) if idle_counts else 0
        if max_idle - min_idle > 50:
            worst = max(idle_counts, key=idle_counts.get)
            suggestions.append(
                f"UNEVEN UTILIZATION: Bot {worst} idles {max_idle} rounds vs "
                f"{min_idle} for the least idle bot. Task assignment may be unbalanced.")

    # Check order gaps
    for og in report["order_gaps"]:
        if og["gap"] > 60:
            suggestions.append(
                f"SLOW ORDER: {og['order']} took {og['gap']} rounds. "
                f"May need parallelized picking or route optimization.")

    if not suggestions:
        suggestions.append("No major bottlenecks detected. Fine-tuning parameters "
                           "may yield marginal gains.")

    for i, s in enumerate(suggestions, 1):
        print(f"\n  {i}. {s}")
    print()


def _make_easy_scenario() -> Simulator:
    width, height = 12, 10
    walls: set[Pos] = set()
    for x in range(width):
        walls.add((x, 0)); walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y)); walls.add((width - 1, y))
    shelf_types: dict[Pos, str] = {
        (2,2):"milk",(2,3):"milk",(2,6):"milk",(2,7):"milk",
        (3,2):"bread",(3,3):"bread",(3,6):"bread",(3,7):"bread",
        (5,2):"butter",(5,3):"butter",(5,6):"butter",(5,7):"butter",
        (6,2):"yogurt",(6,3):"yogurt",(6,6):"yogurt",(6,7):"yogurt",
        (8,2):"milk",(8,3):"bread",(8,6):"butter",(8,7):"yogurt",
        (9,2):"milk",(9,3):"bread",(9,6):"butter",(9,7):"yogurt",
    }
    orders = [{"id":f"order_{i}","items_required":it} for i,it in enumerate([
        ["milk","bread","butter"],["yogurt","milk","bread"],
        ["butter","yogurt","milk"],["bread","butter","yogurt"],
        ["milk","yogurt","bread","butter"],["milk","bread","butter"],
        ["yogurt","milk","bread"],["butter","yogurt","milk"],
        ["bread","butter","yogurt"],["milk","bread","butter"],
        ["yogurt","bread","milk"],["butter","milk","yogurt"],
        ["milk","bread","butter"],["yogurt","milk","bread"],
        ["butter","yogurt","milk"],["bread","butter","yogurt"],
        ["milk","yogurt","bread","butter"],["milk","bread","butter"],
        ["yogurt","milk","bread"],["butter","yogurt","milk"],
    ])]
    return Simulator(
        width=width, height=height, walls=walls,
        shelves=set(shelf_types.keys()), drop_off=(1, 8),
        spawn_positions=[(10, 8)],
        order_sequence=orders, item_types_at_shelves=shelf_types,
    )


def _make_medium_scenario() -> Simulator:
    width, height = 16, 12
    walls: set[Pos] = set()
    for x in range(width):
        walls.add((x, 0)); walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y)); walls.add((width - 1, y))
    types_list = ["milk","bread","butter","yogurt","cheese","juice","eggs","ham"]
    shelf_types: dict[Pos, str] = {}
    ti = 0
    for ax in [2,3,5,6,8,9,11,12]:
        for sy in [2,3,6,7,9]:
            shelf_types[(ax,sy)] = types_list[ti % len(types_list)]
            ti += 1
    orders = [{"id":f"order_{i}","items_required":it} for i,it in enumerate([
        ["milk","bread","butter"],["yogurt","cheese","juice"],
        ["eggs","ham","milk"],["bread","butter","yogurt","cheese"],
        ["juice","eggs","ham"],["milk","cheese","bread","yogurt"],
        ["butter","juice","eggs","ham"],["milk","bread","cheese"],
        ["yogurt","ham","eggs"],["butter","juice","milk"],
        ["bread","cheese","yogurt","ham"],["eggs","milk","butter"],
        ["juice","bread","cheese"],["yogurt","ham","milk","eggs"],
        ["butter","bread","juice"],["cheese","yogurt","milk"],
        ["ham","eggs","butter","juice"],["milk","bread","yogurt"],
        ["cheese","ham","eggs"],["butter","juice","milk","bread"],
    ])]
    return Simulator(
        width=width, height=height, walls=walls,
        shelves=set(shelf_types.keys()), drop_off=(1, 10),
        spawn_positions=[(14, 10), (14, 8), (14, 6)],
        order_sequence=orders, item_types_at_shelves=shelf_types,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile live bot in simulator")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recon", type=str, help="Recon JSON file")
    source.add_argument("--scenario", choices=["easy", "medium"])
    source.add_argument("--all", action="store_true",
                        help="Run all available scenarios + recon files")
    args = parser.parse_args()

    scenarios: list[tuple[str, Simulator]] = []

    if args.all:
        scenarios.append(("Built-in Easy", _make_easy_scenario()))
        scenarios.append(("Built-in Medium", _make_medium_scenario()))
        for p in sorted((_ROOT / "logs").glob("*_recon.json")):
            scenarios.append((f"Recon: {p.name}", Simulator.from_recon_file(str(p))))
    elif args.recon:
        p = Path(args.recon)
        scenarios.append((f"Recon: {p.name}", Simulator.from_recon_file(str(p))))
    else:
        if args.scenario == "easy":
            scenarios.append(("Built-in Easy", _make_easy_scenario()))
        else:
            scenarios.append(("Built-in Medium", _make_medium_scenario()))

    for label, sim in scenarios:
        print(f"\n{'#' * 65}")
        print(f"  {label}")
        print(f"  Map: {sim.width}x{sim.height}, {len(sim.shelves)} shelves, "
              f"{len(sim.spawn_positions)} bots")
        print(f"{'#' * 65}")

        adapter = ProfilingAdapter()
        t0 = time.perf_counter()
        result = sim.run(adapter, verbose=False)
        elapsed = time.perf_counter() - t0
        adapter.finalize(result)

        report = analyze(adapter, result)
        print_report(report)
        print(f"  (Simulation took {elapsed:.2f}s)")
        adapter.reset()


if __name__ == "__main__":
    main()
