"""
Parameter optimizer for Grocery Bot strategy.
Runs 1k-100k iterations of simulator with mutated parameters.

Usage:
    python -m offline.optimize logs/game_log.json --iterations 10000

    # Or from code:
    from offline.optimize import optimize
    best_params, best_score = optimize(simulator, iterations=10000)
"""
from __future__ import annotations
import json
import sys
import time
from dataclasses import asdict
from .simulator import Simulator
from .strategy import StrategyParams, ParameterizedStrategy


def optimize(sim: Simulator, iterations: int = 10_000,
             initial_params: StrategyParams = None,
             verbose: bool = True) -> tuple[StrategyParams, int]:
    """
    Hill-climbing optimizer with restarts.
    
    Mutates strategy parameters, runs simulator, keeps improvements.
    Uses simulated annealing schedule for exploration vs exploitation.
    
    Returns: (best_params, best_score)
    """
    # Precompute BFS once — shared across ALL iterations
    shared_cache: dict = {}
    probe = ParameterizedStrategy(
        StrategyParams(), sim.width, sim.height, sim.walls, sim.shelves,
        shared_bfs_cache=shared_cache,
    )
    if verbose:
        print("Precomputing BFS cache...", end=" ", flush=True)
    probe.precompute_bfs()
    if verbose:
        print(f"done ({len(shared_cache)} cells cached)")

    best_params = initial_params or StrategyParams()
    best_score = _evaluate(sim, best_params, shared_cache)

    if verbose:
        print(f"Initial score: {best_score}")
        print(f"Running {iterations} iterations...")

    start = time.time()
    improvements = 0
    stale_count = 0
    restart_count = 0

    # Track top N for diversity
    top_n: list[tuple[int, StrategyParams]] = [(best_score, best_params)]

    for i in range(iterations):
        # Annealing: temperature decreases over time
        progress = i / iterations
        temperature = max(0.1, 1.0 - progress * 0.8)

        # Restart from random if stuck
        if stale_count > 200:
            # Pick a random parent from top performers
            import random
            if top_n and random.random() < 0.5:
                _, parent = random.choice(top_n[:5])
            else:
                parent = StrategyParams()  # fully random
                parent = parent.mutate(temperature=3.0)  # big mutations
            current_params = parent
            stale_count = 0
            restart_count += 1
        else:
            current_params = best_params

        # Mutate
        candidate = current_params.mutate(temperature)

        # Evaluate
        score = _evaluate(sim, candidate, shared_cache)

        if score > best_score:
            best_score = score
            best_params = candidate
            improvements += 1
            stale_count = 0

            # Update top N
            top_n.append((score, candidate))
            top_n.sort(key=lambda x: -x[0])
            top_n = top_n[:10]

            if verbose and (improvements <= 10 or improvements % 5 == 0):
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                print(f"  [{i+1:6d}] New best: {best_score:3d} "
                      f"({rate:.0f} iter/s, {improvements} improvements)")
        else:
            stale_count += 1

        # Progress report
        if verbose and (i + 1) % 1000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"  [{i+1:6d}] Best: {best_score:3d}, "
                  f"{rate:.0f} iter/s, "
                  f"{improvements} improvements, "
                  f"{restart_count} restarts")

    elapsed = time.time() - start

    if verbose:
        print(f"\n{'='*50}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*50}")
        print(f"  Iterations: {iterations}")
        print(f"  Time: {elapsed:.1f}s ({iterations/elapsed:.0f} iter/s)")
        print(f"  Best score: {best_score}")
        print(f"  Improvements: {improvements}")
        print(f"  Restarts: {restart_count}")
        print(f"\n  Best parameters:")
        for k, v in asdict(best_params).items():
            default = getattr(StrategyParams(), k)
            marker = " ←" if v != default else ""
            print(f"    {k}: {v}{marker}")

    return best_params, best_score


def _evaluate(sim: Simulator, params: StrategyParams,
              shared_cache: dict = None) -> int:
    """Run simulator with given parameters, return score."""
    strategy = ParameterizedStrategy(
        params,
        width=sim.width,
        height=sim.height,
        walls=sim.walls,
        shelves=sim.shelves,
        shared_bfs_cache=shared_cache,
    )
    result = sim.run(strategy, verbose=False)
    return result["score"]


def grid_search(sim: Simulator, verbose: bool = True) -> tuple[StrategyParams, int]:
    """
    Systematic grid search over key parameters.
    Tests ~500 combinations. Slower but more thorough than hill climbing.
    """
    # Precompute BFS
    shared_cache: dict = {}
    probe = ParameterizedStrategy(
        StrategyParams(), sim.width, sim.height, sim.walls, sim.shelves,
        shared_bfs_cache=shared_cache,
    )
    probe.precompute_bfs()

    best_params = StrategyParams()
    best_score = 0
    tested = 0

    ranges = {
        "min_items_to_deliver": [1, 2, 3],
        "cross_order_threshold": [0, 1, 2],
        "cross_order_max": [0, 1, 2],
        "w_completion": [0.0, 3.0, 5.0, 10.0],
        "endgame_threshold": [20, 30, 40, 50],
    }

    # Generate all combinations
    from itertools import product
    keys = list(ranges.keys())
    for combo in product(*[ranges[k] for k in keys]):
        params = StrategyParams()
        for k, v in zip(keys, combo):
            setattr(params, k, v)

        score = _evaluate(sim, params, shared_cache)
        tested += 1

        if score > best_score:
            best_score = score
            best_params = params
            if verbose:
                combo_str = ", ".join(f"{k}={v}" for k, v in zip(keys, combo))
                print(f"  [{tested:4d}] New best: {score:3d} ({combo_str})")

    if verbose:
        print(f"\nGrid search: tested {tested} combinations, best={best_score}")

    return best_params, best_score


def compare_strategies(sim: Simulator, n: int = 10,
                       verbose: bool = True) -> None:
    """Run multiple strategies and compare."""
    # Precompute BFS once
    shared_cache: dict = {}
    probe = ParameterizedStrategy(
        StrategyParams(), sim.width, sim.height, sim.walls, sim.shelves,
        shared_bfs_cache=shared_cache,
    )
    probe.precompute_bfs()

    strategies = {
        "default": StrategyParams(),
        "greedy_deliver": StrategyParams(min_items_to_deliver=1,
                                          cross_order_max=0),
        "patient_deliver": StrategyParams(min_items_to_deliver=3,
                                           deliver_if_close=2),
        "aggressive_cross": StrategyParams(cross_order_threshold=2,
                                            cross_order_max=2),
        "completion_focused": StrategyParams(w_completion=10.0,
                                              w_distance=0.5),
        "early_endgame": StrategyParams(endgame_threshold=50),
        "late_endgame": StrategyParams(endgame_threshold=15),
    }

    print(f"{'Strategy':<22} {'Score':>6} {'Items':>6} {'Orders':>7}")
    print("-" * 45)

    results = {}
    for name, params in strategies.items():
        total_score = 0
        total_items = 0
        total_orders = 0

        for _ in range(n):
            strategy = ParameterizedStrategy(
                params, sim.width, sim.height, sim.walls, sim.shelves,
                shared_bfs_cache=shared_cache,
            )
            result = sim.run(strategy, verbose=False)
            total_score += result["score"]
            total_items += result["items_delivered"]
            total_orders += result["orders_completed"]

        avg_score = total_score / n
        avg_items = total_items / n
        avg_orders = total_orders / n
        results[name] = avg_score

        print(f"  {name:<20} {avg_score:6.1f} {avg_items:6.1f} {avg_orders:7.1f}")

    print()
    best_name = max(results, key=results.get)
    print(f"  Winner: {best_name} ({results[best_name]:.1f})")


def save_params(params: StrategyParams, path: str):
    """Save optimized parameters to JSON."""
    with open(path, "w") as f:
        json.dump(asdict(params), f, indent=2)
    print(f"Parameters saved: {path}")


def load_params(path: str) -> StrategyParams:
    """Load parameters from JSON."""
    with open(path) as f:
        data = json.load(f)
    return StrategyParams(**data)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    """
    Usage:
        python -m offline.optimize <log.json> [--iterations N] [--grid-search] [--compare]
    """
    if len(sys.argv) < 2:
        print(main.__doc__)
        return

    log_path = sys.argv[1]
    sim = Simulator.from_game_log(log_path)

    if "--compare" in sys.argv:
        print("Comparing built-in strategies...\n")
        compare_strategies(sim, n=1)  # n=1 since deterministic
        return

    if "--grid-search" in sys.argv:
        print("Running grid search...\n")
        params, score = grid_search(sim)
        save_params(params, "logs/best_params_grid.json")
        return

    # Hill climbing (default)
    iterations = 10_000
    if "--iterations" in sys.argv:
        idx = sys.argv.index("--iterations")
        iterations = int(sys.argv[idx + 1])

    params, score = optimize(sim, iterations=iterations)
    save_params(params, "logs/best_params.json")

    # Show final run details
    print(f"\nFinal run with best params:")
    strategy = ParameterizedStrategy(
        params, sim.width, sim.height, sim.walls, sim.shelves
    )
    strategy.precompute_bfs()
    sim.run(strategy, verbose=True)


if __name__ == "__main__":
    main()
