"""
Nightmare optimizer: grid search + hill climbing over NightmareParams.

Phase 1: Grid search over discrete parameters
Phase 2: Hill climbing from top configs
Phase 3: Seed search on best config

Usage:
    py -m nightmare_lab --recon logs/74001e7f_2026-03-09_recon.json
    py -m nightmare_lab --recon logs/74001e7f_2026-03-09_recon.json --quick
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Simulering.offline.simulator import Simulator
from nightmare_lab.strategy import NightmareParams, NightmareStrategy


def evaluate(sim: Simulator, params: NightmareParams) -> int:
    strategy = NightmareStrategy(params)
    result = sim.run(strategy)
    return result["score"]


def evaluate_multi_seed(sim: Simulator, params: NightmareParams,
                        n_seeds: int = 5) -> tuple[int, int, float]:
    """Evaluate across multiple seeds. Returns (max_score, best_seed, avg_score)."""
    best_score = 0
    best_seed = 0
    total = 0
    for seed in range(n_seeds):
        p = NightmareParams(**{**params.to_dict(), 'seed': seed})
        p.zone_bots = params.zone_bots
        score = evaluate(sim, p)
        total += score
        if score > best_score:
            best_score = score
            best_seed = seed
    return best_score, best_seed, total / n_seeds


def grid_search(sim: Simulator, verbose: bool = True) -> list[tuple[int, NightmareParams]]:
    """Grid search over key discrete parameters."""
    configs: list[tuple[int, NightmareParams]] = []
    best_score = 0
    t0 = time.time()

    max_del_values = [1, 2, 3, 5]
    max_carry_values = [1, 2, 3]
    zone_pen_values = [0, 10, 15, 20, 30]
    w_comp_values = [5, 10, 15]
    carry_close_values = [5, 8, 12]

    total_combos = (len(max_del_values) * len(max_carry_values) *
                    len(zone_pen_values) * len(w_comp_values) *
                    len(carry_close_values))

    if verbose:
        print(f"Grid search: {total_combos} configurations x 5 seeds...")

    count = 0
    for max_del in max_del_values:
        for max_carry in max_carry_values:
            for zone_pen in zone_pen_values:
                for w_comp in w_comp_values:
                    for carry_close in carry_close_values:
                        params = NightmareParams(
                            max_deliverers=max_del,
                            max_carry=max_carry,
                            zone_penalty=zone_pen,
                            w_completion=w_comp,
                            carry_if_close=carry_close,
                        )
                        max_s, best_seed, avg_s = evaluate_multi_seed(sim, params, n_seeds=5)
                        configs.append((max_s, NightmareParams(
                            **{**params.to_dict(), 'seed': best_seed}
                        )))
                        count += 1

                        if max_s > best_score:
                            best_score = max_s
                            elapsed = time.time() - t0
                            if verbose:
                                print(f"  [{count}/{total_combos}] NEW BEST: {max_s} "
                                      f"(avg={avg_s:.1f}) del={max_del} carry={max_carry} "
                                      f"zp={zone_pen} wc={w_comp} cc={carry_close} "
                                      f"seed={best_seed} [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    if verbose:
        print(f"  Grid search done: {count} configs in {elapsed:.1f}s")

    configs.sort(key=lambda x: -x[0])
    return configs


def hill_climb(sim: Simulator, start_params: NightmareParams,
               iterations: int = 2000, verbose: bool = True) -> tuple[int, NightmareParams]:
    """Hill climbing with multi-seed evaluation."""
    current = start_params
    current_score = evaluate(sim, current)
    best = current
    best_score = current_score
    stale = 0
    t0 = time.time()

    for i in range(iterations):
        temp = max(0.3, 1.0 - (i / iterations) * 0.7)

        if stale > iterations // 8:
            current = NightmareParams.random()
            current_score = evaluate(sim, current)
            stale = 0

        candidate = current.mutate(temperature=temp)
        # Test with a few seeds
        max_s, best_seed, avg_s = evaluate_multi_seed(sim, candidate, n_seeds=3)
        candidate_with_seed = NightmareParams(**{**candidate.to_dict(), 'seed': best_seed})
        candidate_with_seed.zone_bots = candidate.zone_bots

        if max_s >= current_score:
            current = candidate_with_seed
            current_score = max_s
            stale = 0

            if max_s > best_score:
                best_score = max_s
                best = candidate_with_seed
                if verbose:
                    elapsed = time.time() - t0
                    print(f"    [{i+1}/{iterations}] NEW BEST: {best_score} "
                          f"(avg={avg_s:.1f}) seed={best_seed} [{elapsed:.1f}s]")
        else:
            stale += 1

    return best_score, best


def seed_search(sim: Simulator, params: NightmareParams,
                n_seeds: int = 200, verbose: bool = True) -> tuple[int, int]:
    """Search for the best seed for a given config."""
    best_score = 0
    best_seed = 0

    for seed in range(n_seeds):
        p = NightmareParams(**{**params.to_dict(), 'seed': seed})
        p.zone_bots = params.zone_bots
        score = evaluate(sim, p)
        if score > best_score:
            best_score = score
            best_seed = seed
            if verbose:
                print(f"  seed={seed}: {score}")

    return best_score, best_seed


def optimize(sim: Simulator, iterations: int = 5000,
             hill_starts: int = 3, verbose: bool = True):
    """Full pipeline: grid search -> hill climbing -> seed search."""
    t_start = time.time()

    # Phase 1
    if verbose:
        print("=" * 60)
        print("PHASE 1: Grid Search")
        print("=" * 60)
    grid_results = grid_search(sim, verbose)

    if verbose:
        print(f"\nTop 5:")
        for i, (score, params) in enumerate(grid_results[:5]):
            print(f"  {i+1}. Score {score}: {params.to_dict()}")

    # Phase 2
    if verbose:
        print("\n" + "=" * 60)
        print(f"PHASE 2: Hill Climbing ({iterations} iterations, {hill_starts} starts)")
        print("=" * 60)

    per_start = max(200, iterations // hill_starts)
    overall_best_score = grid_results[0][0]
    overall_best_params = grid_results[0][1]

    for idx, (score, params) in enumerate(grid_results[:hill_starts]):
        if verbose:
            print(f"\n  Start {idx+1}/{hill_starts} (from score {score}):")
        s, p = hill_climb(sim, params, per_start, verbose)
        if s > overall_best_score:
            overall_best_score = s
            overall_best_params = p

    # Phase 3: Seed search
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 3: Seed Search (200 seeds)")
        print("=" * 60)

    best_seed_score, best_seed = seed_search(sim, overall_best_params, 200, verbose)
    if best_seed_score > overall_best_score:
        overall_best_score = best_seed_score
        overall_best_params = NightmareParams(
            **{**overall_best_params.to_dict(), 'seed': best_seed}
        )
        overall_best_params.zone_bots = grid_results[0][1].zone_bots

    total_time = time.time() - t_start
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Best score: {overall_best_score}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Parameters: {overall_best_params.to_dict()}")

    return overall_best_score, overall_best_params


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nightmare strategy optimizer")
    parser.add_argument("--recon", type=str, help="Path to nightmare recon JSON")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--hill-starts", type=int, default=3)
    parser.add_argument("--quick", action="store_true",
                        help="Quick: grid search + 100 seeds only")
    args = parser.parse_args()

    recon_path = args.recon
    if not recon_path:
        logs_dir = Path(__file__).parent.parent / "logs"
        candidates = sorted(logs_dir.glob("74001e7f_*_recon.json"), reverse=True)
        if candidates:
            recon_path = str(candidates[0])
            print(f"Auto-found recon: {recon_path}")
        else:
            print("ERROR: No nightmare recon found.")
            sys.exit(1)

    sim = Simulator.from_recon_file(recon_path)
    print(f"Grid: {sim.width}x{sim.height}, Bots: {len(sim.spawn_positions)}, "
          f"Orders: {len(sim.order_sequence)}")

    # Baseline
    baseline = evaluate(sim, NightmareParams())
    print(f"Baseline: {baseline}\n")

    if args.quick:
        # Quick mode: just grid search + seed search
        grid_results = grid_search(sim)
        best_params = grid_results[0][1]
        best_score, best_seed = seed_search(sim, best_params, 100)
        print(f"\nBest: {best_score} (seed={best_seed})")
        print(f"Params: {best_params.to_dict()}")
    else:
        best_score, best_params = optimize(
            sim, args.iterations, args.hill_starts
        )

    # Save
    result_path = Path(recon_path).parent / "nightmare_best.json"
    with open(result_path, "w") as f:
        json.dump({
            "best_score": best_score,
            "baseline": baseline,
            "recon": str(recon_path),
            "params": best_params.to_dict(),
        }, f, indent=2)
    print(f"\nSaved to {result_path}")


if __name__ == "__main__":
    main()
