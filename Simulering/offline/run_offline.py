"""
Offline test runner: run the live bot through the Simulator.

Workflows:
  1. Latest recon:     Auto-find newest recon file for a difficulty (RECOMMENDED)
  2. From recon file:  Replay a specific captured game
  3. From test scenario: Built-in scenarios (smoke test only, NOT representative)
  4. Side-by-side:     Compare live bot vs ParameterizedStrategy

Usage (from project root):
    py -m Simulering.offline.run_offline --latest easy
    py -m Simulering.offline.run_offline --latest medium --compare
    py -m Simulering.offline.run_offline --recon logs/abc12345_2026-03-01_recon.json
    py -m Simulering.offline.run_offline --scenario easy  # smoke test only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root on path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Simulering.offline.simulator import Simulator
from Simulering.offline.bot_adapter import BotAdapter
from Simulering.offline.strategy import StrategyParams, ParameterizedStrategy
from Simulering.offline.recon_utils import find_latest_recon

Pos = tuple[int, int]


def _make_easy_scenario() -> Simulator:
    """Built-in easy: 12x10, 1 bot, 4 item types, 20 orders."""
    width, height = 12, 10
    walls: set[Pos] = set()
    for x in range(width):
        walls.add((x, 0)); walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y)); walls.add((width - 1, y))

    shelf_types: dict[Pos, str] = {
        (2,2): "milk", (2,3): "milk", (2,6): "milk", (2,7): "milk",
        (3,2): "bread", (3,3): "bread", (3,6): "bread", (3,7): "bread",
        (5,2): "butter", (5,3): "butter", (5,6): "butter", (5,7): "butter",
        (6,2): "yogurt", (6,3): "yogurt", (6,6): "yogurt", (6,7): "yogurt",
        (8,2): "milk", (8,3): "bread", (8,6): "butter", (8,7): "yogurt",
        (9,2): "milk", (9,3): "bread", (9,6): "butter", (9,7): "yogurt",
    }
    shelves = set(shelf_types.keys())

    orders = [
        {"id": f"order_{i}", "items_required": items}
        for i, items in enumerate([
            ["milk","bread","butter"], ["yogurt","milk","bread"],
            ["butter","yogurt","milk"], ["bread","butter","yogurt"],
            ["milk","yogurt","bread","butter"], ["milk","bread","butter"],
            ["yogurt","milk","bread"], ["butter","yogurt","milk"],
            ["bread","butter","yogurt"], ["milk","bread","butter"],
            ["yogurt","bread","milk"], ["butter","milk","yogurt"],
            ["milk","bread","butter"], ["yogurt","milk","bread"],
            ["butter","yogurt","milk"], ["bread","butter","yogurt"],
            ["milk","yogurt","bread","butter"], ["milk","bread","butter"],
            ["yogurt","milk","bread"], ["butter","yogurt","milk"],
        ])
    ]

    return Simulator(
        width=width, height=height, walls=walls, shelves=shelves,
        drop_off=(1, 8), spawn_positions=[(10, 8)],
        order_sequence=orders, item_types_at_shelves=shelf_types,
    )


def _make_medium_scenario() -> Simulator:
    """Built-in medium: 16x12, 3 bots, 8 item types, 20 orders."""
    width, height = 16, 12
    walls: set[Pos] = set()
    for x in range(width):
        walls.add((x, 0)); walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y)); walls.add((width - 1, y))

    types_list = ["milk","bread","butter","yogurt","cheese","juice","eggs","ham"]
    shelf_types: dict[Pos, str] = {}
    ti = 0
    for ax in [2, 3, 5, 6, 8, 9, 11, 12]:
        for sy in [2, 3, 6, 7, 9]:
            shelf_types[(ax, sy)] = types_list[ti % len(types_list)]
            ti += 1
    shelves = set(shelf_types.keys())

    orders = [
        {"id": f"order_{i}", "items_required": items}
        for i, items in enumerate([
            ["milk","bread","butter"], ["yogurt","cheese","juice"],
            ["eggs","ham","milk"], ["bread","butter","yogurt","cheese"],
            ["juice","eggs","ham"], ["milk","cheese","bread","yogurt"],
            ["butter","juice","eggs","ham"], ["milk","bread","cheese"],
            ["yogurt","ham","eggs"], ["butter","juice","milk"],
            ["bread","cheese","yogurt","ham"], ["eggs","milk","butter"],
            ["juice","bread","cheese"], ["yogurt","ham","milk","eggs"],
            ["butter","bread","juice"], ["cheese","yogurt","milk"],
            ["ham","eggs","butter","juice"], ["milk","bread","yogurt"],
            ["cheese","ham","eggs"], ["butter","juice","milk","bread"],
        ])
    ]

    return Simulator(
        width=width, height=height, walls=walls, shelves=shelves,
        drop_off=(1, 10), spawn_positions=[(14, 10), (14, 8), (14, 6)],
        order_sequence=orders, item_types_at_shelves=shelf_types,
    )


def run_live_bot(sim: Simulator, *, verbose: bool = True,
                 save_recon: bool = False, viz: bool = False) -> dict:
    """Run the live bot's full pipeline through the simulator."""
    adapter = BotAdapter(save_recon=save_recon, suppress_logs=not verbose)
    if viz:
        adapter._start_viz = True

    t0 = time.perf_counter()
    result = sim.run(adapter, verbose=verbose)
    elapsed = time.perf_counter() - t0

    recon = adapter.finalize(result)

    result["elapsed_s"] = elapsed
    result["recon_data"] = recon

    if viz and adapter._coordinator and adapter._coordinator._viz:
        viz_obj = adapter._coordinator._viz
        replay_path = Path(__file__).resolve().parent.parent.parent / "viz" / "public" / "replay.json"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(replay_path, "w") as f:
            json.dump({"type": "replay", "states": viz_obj._all_states}, f)
        print(f"\n  Replay saved: {replay_path}")
        print(f"  States: {len(viz_obj._all_states)}")
        print(f"  Open http://localhost:3000 to view replay\n")

    adapter.reset()
    return result


def run_simple_strategy(sim: Simulator, params: StrategyParams | None = None,
                        *, verbose: bool = False) -> dict:
    """Run ParameterizedStrategy through the simulator."""
    params = params or StrategyParams()
    strategy = ParameterizedStrategy(
        params, sim.width, sim.height, sim.walls, sim.shelves,
    )
    strategy.precompute_bfs()

    t0 = time.perf_counter()
    result = sim.run(strategy, verbose=verbose)
    elapsed = time.perf_counter() - t0

    result["elapsed_s"] = elapsed
    return result


def print_result(label: str, result: dict) -> None:
    print(f"  {label}:")
    print(f"    Score:  {result['score']}")
    print(f"    Items:  {result['items_delivered']}")
    print(f"    Orders: {result['orders_completed']}")
    print(f"    Rounds: {result['rounds_used']}")
    print(f"    Time:   {result.get('elapsed_s', 0):.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live bot offline through the Simulator",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recon", type=str,
                        help="Path to recon JSON file from a live game")
    source.add_argument("--scenario", choices=["easy", "medium"],
                        help="Built-in test scenario (smoke test only, not representative)")
    source.add_argument("--latest", choices=["easy", "medium", "hard", "expert", "nightmare"],
                        help="Auto-find latest recon file for difficulty (recommended)")

    parser.add_argument("--compare", action="store_true",
                        help="Also run ParameterizedStrategy for comparison")
    parser.add_argument("--save-recon", action="store_true",
                        help="Save recon data from the live bot run")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-round logging")
    parser.add_argument("--viz", action="store_true",
                        help="Start visualization server on ws://localhost:8765")
    args = parser.parse_args()

    if args.recon:
        path = Path(args.recon)
        if not path.exists():
            print(f"Error: recon file not found: {path}")
            sys.exit(1)
        print(f"Loading simulator from recon: {path}")
        sim = Simulator.from_recon_file(str(path))
    elif args.latest:
        path = find_latest_recon(args.latest)
        if path is None:
            print(f"Error: no recon file found for '{args.latest}'. "
                  f"Run a live game first to generate recon data.")
            sys.exit(1)
        print(f"Loading latest {args.latest} recon: {path.name}")
        sim = Simulator.from_recon_file(str(path))
    else:
        print(f"WARNING: Built-in scenarios are NOT representative of live games!")
        print(f"         Use --latest {args.scenario} for realistic testing.")
        print(f"Using built-in scenario: {args.scenario}")
        sim = _make_easy_scenario() if args.scenario == "easy" else _make_medium_scenario()

    print(f"Map: {sim.width}x{sim.height}, "
          f"{len(sim.shelves)} shelves, "
          f"{len(sim.spawn_positions)} bots, "
          f"{len(sim.order_sequence)} orders")
    print()

    print("Running live bot...")
    live_result = run_live_bot(
        sim,
        verbose=not args.quiet,
        save_recon=args.save_recon,
        viz=args.viz,
    )
    print()
    print_result("Live Bot", live_result)

    if args.compare:
        print()
        print("Running ParameterizedStrategy (default params)...")
        simple_result = run_simple_strategy(sim)
        print_result("Simple Strategy", simple_result)

        delta = live_result["score"] - simple_result["score"]
        sign = "+" if delta >= 0 else ""
        print(f"\n  Delta: {sign}{delta} (live bot vs simple)")

    if args.save_recon and live_result.get("recon_data"):
        recon = live_result["recon_data"]
        fp = recon.get("fingerprint", "unknown")
        out_path = _ROOT / "logs" / f"{fp}_offline_recon.json"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(recon, indent=2), encoding="utf-8")
        print(f"\n  Recon data saved: {out_path}")


if __name__ == "__main__":
    main()
