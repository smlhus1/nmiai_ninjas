"""
Offline test runner: run the live bot through the Simulator.

Three workflows:
  1. From recon file:  Replay a captured game with the full live bot pipeline
  2. From test scenario: Use built-in Easy/Medium scenarios for quick iteration
  3. Side-by-side:     Compare live bot vs ParameterizedStrategy on same scenario

Usage (from project root):
    py -m Simulering.offline.run_offline --recon logs/abc12345_2026-03-01_recon.json
    py -m Simulering.offline.run_offline --scenario easy
    py -m Simulering.offline.run_offline --scenario medium --compare
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


def _make_hard_scenario() -> Simulator:
    """Built-in hard: 22x14, 5 bots, 4 aisles, 12 item types, 30 orders."""
    width, height = 22, 14
    walls: set[Pos] = set()
    # Border walls
    for x in range(width):
        walls.add((x, 0)); walls.add((x, height - 1))
    for y in range(height):
        walls.add((0, y)); walls.add((width - 1, y))

    # 12 item types for hard mode
    types_list = [
        "milk", "bread", "butter", "yogurt", "cheese", "juice",
        "eggs", "ham", "coffee", "tea", "cereal", "pasta",
    ]

    # 4 aisles: each aisle is shelf-walkway-shelf (3 cols)
    # Aisle 1: cols 3,4,5   (shelves at 3,5, walkway at 4)
    # Aisle 2: cols 7,8,9   (shelves at 7,9, walkway at 8)
    # Aisle 3: cols 11,12,13 (shelves at 11,13, walkway at 12)
    # Aisle 4: cols 15,16,17 (shelves at 15,17, walkway at 16)
    # Corridors: y=1 (top), y=6 (mid), y=12 (bottom)
    # Shelf rows: y=2..5 (upper section), y=7..11 (lower section)
    shelf_cols = [3, 5, 7, 9, 11, 13, 15, 17]
    shelf_rows_upper = [2, 3, 4, 5]
    shelf_rows_lower = [7, 8, 9, 10, 11]

    shelf_types: dict[Pos, str] = {}
    ti = 0
    for sx in shelf_cols:
        for sy in shelf_rows_upper + shelf_rows_lower:
            shelf_types[(sx, sy)] = types_list[ti % len(types_list)]
            ti += 1

    shelves = set(shelf_types.keys())

    # 5 bots spawning at bottom-right (inside border)
    spawns = [(20, 12), (20, 11), (20, 10), (19, 12), (19, 11)]

    # 30 orders, 3-5 items each (using all 12 types)
    orders = [
        {"id": f"order_{i}", "items_required": items}
        for i, items in enumerate([
            ["milk", "bread", "butter"],
            ["yogurt", "cheese", "juice"],
            ["eggs", "ham", "coffee"],
            ["tea", "cereal", "pasta"],
            ["milk", "cheese", "eggs", "tea"],
            ["bread", "yogurt", "ham", "cereal"],
            ["butter", "juice", "coffee", "pasta"],
            ["milk", "bread", "cheese", "yogurt", "eggs"],
            ["ham", "coffee", "tea"],
            ["cereal", "pasta", "milk"],
            ["bread", "butter", "yogurt", "cheese"],
            ["juice", "eggs", "ham", "coffee", "tea"],
            ["cereal", "pasta", "milk"],
            ["bread", "butter", "yogurt"],
            ["cheese", "juice", "eggs", "ham"],
            ["coffee", "tea", "cereal", "pasta", "milk"],
            ["bread", "butter", "yogurt"],
            ["cheese", "juice", "eggs"],
            ["ham", "coffee", "tea", "cereal"],
            ["pasta", "milk", "bread", "butter", "yogurt"],
            ["cheese", "juice", "eggs"],
            ["ham", "coffee", "tea"],
            ["cereal", "pasta", "milk", "bread"],
            ["butter", "yogurt", "cheese"],
            ["juice", "eggs", "ham", "coffee", "tea"],
            ["cereal", "pasta", "milk"],
            ["bread", "butter", "yogurt", "cheese"],
            ["juice", "eggs", "ham"],
            ["coffee", "tea", "cereal", "pasta"],
            ["milk", "bread", "butter", "yogurt", "cheese"],
        ])
    ]

    return Simulator(
        width=width, height=height, walls=walls, shelves=shelves,
        drop_off=(1, 12), spawn_positions=spawns,
        order_sequence=orders, item_types_at_shelves=shelf_types,
    )


def run_live_bot(sim: Simulator, *, verbose: bool = True,
                 save_recon: bool = False) -> dict:
    """Run the live bot's full pipeline through the simulator."""
    adapter = BotAdapter(save_recon=save_recon, suppress_logs=not verbose)

    t0 = time.perf_counter()
    result = sim.run(adapter, verbose=verbose)
    elapsed = time.perf_counter() - t0

    recon = adapter.finalize(result)
    adapter.reset()

    result["elapsed_s"] = elapsed
    result["recon_data"] = recon
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
    source.add_argument("--scenario", choices=["easy", "medium", "hard"],
                        help="Built-in test scenario")

    parser.add_argument("--compare", action="store_true",
                        help="Also run ParameterizedStrategy for comparison")
    parser.add_argument("--save-recon", action="store_true",
                        help="Save recon data from the live bot run")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-round logging")
    args = parser.parse_args()

    if args.recon:
        path = Path(args.recon)
        if not path.exists():
            print(f"Error: recon file not found: {path}")
            sys.exit(1)
        print(f"Loading simulator from recon: {path}")
        sim = Simulator.from_recon_file(str(path))
    else:
        print(f"Using built-in scenario: {args.scenario}")
        if args.scenario == "easy":
            sim = _make_easy_scenario()
        elif args.scenario == "medium":
            sim = _make_medium_scenario()
        else:
            sim = _make_hard_scenario()

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
