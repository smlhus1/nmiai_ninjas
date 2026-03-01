"""
Coordinator config optimizer: vary bot strategy parameters, simulate each, find the best.

Runs the FULL live bot pipeline (Coordinator + TaskPlanner + Hungarian + PIBT)
with different parameter configurations through the Simulator. Each iteration
varies decision thresholds like route sizes, endgame timing, stuck detection, etc.

This is the core optimization loop: the simulator is deterministic per config,
so different configs produce different scores. Hill climbing finds the best.

Usage (from project root):
    py -m Simulering.offline.optimize_plan --recon logs/31642503_2026-03-01_recon.json
    py -m Simulering.offline.optimize_plan --recon logs/6fb8097b_2026-03-01_recon.json --iterations 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bot.config import CoordinatorConfig
from Simulering.offline.simulator import Simulator
from Simulering.offline.bot_adapter import BotAdapter


def evaluate(sim: Simulator, config: CoordinatorConfig) -> dict:
    """Run full bot pipeline with given config. Returns result dict with score."""
    adapter = BotAdapter(suppress_logs=True, config=config)
    result = sim.run(adapter, verbose=False)
    adapter.reset()
    return result


def optimize(
    recon_path: str,
    iterations: int = 200,
    verbose: bool = True,
) -> tuple[CoordinatorConfig, int]:
    """
    Hill-climbing optimizer over CoordinatorConfig parameters.

    Returns (best_config, best_score).
    """
    recon = json.loads(Path(recon_path).read_text(encoding="utf-8"))
    sim = Simulator.from_recon_data(recon)

    # Baseline with defaults
    default_cfg = CoordinatorConfig()
    if verbose:
        print("Baseline (default config)...")
    baseline = evaluate(sim, default_cfg)
    if verbose:
        print(f"  Score: {baseline['score']}  "
              f"(items={baseline['items_delivered']}, orders={baseline['orders_completed']})")

    best_cfg = default_cfg
    best_score = baseline["score"]
    improvements = 0
    stale = 0

    top_n: list[tuple[int, CoordinatorConfig]] = [(best_score, best_cfg)]

    if verbose:
        print(f"\nOptimizing over {iterations} iterations...")

    t0 = time.perf_counter()

    for i in range(iterations):
        progress = i / iterations
        temperature = max(0.2, 1.0 - progress * 0.7)

        # Restart from top performer if stuck
        if stale > 40 and top_n:
            import random
            parent = random.choice(top_n[:5])[1]
            stale = 0
        else:
            parent = best_cfg

        candidate = parent.mutate(temperature)

        try:
            result = evaluate(sim, candidate)
            score = result["score"]
        except Exception as e:
            if verbose:
                print(f"  [{i+1:4d}] ERROR: {e}")
            stale += 1
            continue

        if score > best_score:
            best_score = score
            best_cfg = candidate
            improvements += 1
            stale = 0

            top_n.append((score, candidate))
            top_n.sort(key=lambda x: -x[0])
            top_n = top_n[:10]

            if verbose:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                diff = asdict(candidate)
                defaults = asdict(CoordinatorConfig())
                changed = {k: v for k, v in diff.items() if v != defaults[k]}
                print(f"  [{i+1:4d}] NEW BEST: {score:3d}  "
                      f"items={result['items_delivered']} orders={result['orders_completed']}  "
                      f"changes={changed}  [{rate:.1f} iter/s]")
        else:
            stale += 1

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1:4d}] best={best_score:3d}  "
                  f"({rate:.1f} iter/s, {improvements} improvements, stale={stale})")

    elapsed = time.perf_counter() - t0

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  OPTIMIZATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Iterations:   {iterations}")
        print(f"  Time:         {elapsed:.1f}s ({iterations/elapsed:.1f} iter/s)")
        print(f"  Baseline:     {baseline['score']}")
        print(f"  Best:         {best_score}  (+{best_score - baseline['score']})")
        print(f"  Improvements: {improvements}")
        print()
        defaults = asdict(CoordinatorConfig())
        best_dict = asdict(best_cfg)
        print(f"  Best config (changes from default):")
        for k, v in best_dict.items():
            marker = " <--" if v != defaults[k] else ""
            print(f"    {k}: {v}{marker}")

    return best_cfg, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Coordinator config via simulation")
    parser.add_argument("--recon", required=True, help="Recon JSON file from a live game")
    parser.add_argument("--iterations", type=int, default=200,
                        help="Number of hill-climbing iterations (default: 200)")
    parser.add_argument("--save", type=str, help="Save best config to this JSON path")
    args = parser.parse_args()

    best_cfg, best_score = optimize(args.recon, iterations=args.iterations)

    # Save config
    out = Path(args.save) if args.save else _ROOT / "logs" / "best_config.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(asdict(best_cfg), indent=2), encoding="utf-8")
    print(f"\n  Config saved to: {out}")


if __name__ == "__main__":
    main()
