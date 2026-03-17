"""
Validate MLPlanner vs V2TaskPlanner baseline in sim.

Runs both planners N times against the same recon file and compares scores.

Usage:
    py -m ml.validate \
      --recon logs/74001e7f_2026-03-16_score274_recon.json \
      --checkpoint models/scorer_74001e7f_2026-03-16.pt \
      --n-runs 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Simulering.offline.bot_adapter import BotAdapter
from Simulering.offline.simulator import Simulator
from bot.config import CoordinatorConfig


def run_n_games(
    recon: dict,
    n_runs: int,
    config: CoordinatorConfig | None = None,
    label: str = "",
) -> list[dict]:
    """Run n_runs games and return results."""
    results = []
    for i in range(n_runs):
        sim = Simulator.from_recon_data(recon)
        adapter = BotAdapter(suppress_logs=True, config=config)
        t0 = time.perf_counter()
        result = sim.run(adapter)
        elapsed = time.perf_counter() - t0
        result["elapsed_s"] = elapsed
        result["plan_ms_avg"] = elapsed * 1000 / max(result["rounds_used"], 1)
        results.append(result)
        adapter.reset()
        print(f"  {label} run {i+1}/{n_runs}: score={result['score']} "
              f"items={result['items_delivered']} orders={result['orders_completed']} "
              f"rounds={result['rounds_used']} ({elapsed:.1f}s)")
    return results


def validate(recon_path: Path, checkpoint_path: Path | None, n_runs: int) -> None:
    recon = json.loads(recon_path.read_text(encoding="utf-8"))
    live_score = recon.get("final_score")

    print(f"Recon: {recon_path.name}")
    if live_score:
        print(f"Live score: {live_score}")
    print(f"Runs per planner: {n_runs}\n")

    # --- V2TaskPlanner baseline ---
    print("=== V2TaskPlanner (baseline) ===")
    baseline_cfg = CoordinatorConfig.for_difficulty(recon.get("bot_count", 20))
    baseline_results = run_n_games(recon, n_runs, config=baseline_cfg, label="V2")

    baseline_scores = [r["score"] for r in baseline_results]
    baseline_mean = statistics.mean(baseline_scores)
    baseline_std = statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0.0

    # --- MLPlanner ---
    print("\n=== MLPlanner ===")
    ml_cfg = CoordinatorConfig.for_difficulty(recon.get("bot_count", 20))
    ml_cfg.use_ml_planner = True
    ml_results = run_n_games(recon, n_runs, config=ml_cfg, label="ML")

    ml_scores = [r["score"] for r in ml_results]
    ml_mean = statistics.mean(ml_scores)
    ml_std = statistics.stdev(ml_scores) if len(ml_scores) > 1 else 0.0

    # --- Timing ---
    ml_plan_ms = statistics.mean([r["plan_ms_avg"] for r in ml_results])

    # --- Report ---
    delta = ml_mean - baseline_mean
    delta_pct = (delta / max(baseline_mean, 1)) * 100

    print(f"\n{'='*55}")
    print(f"{'Planner':<20} {'Mean':>8} {'Std':>8} {'Min':>6} {'Max':>6}")
    print(f"{'-'*55}")
    print(f"{'V2TaskPlanner':<20} {baseline_mean:>8.1f} {baseline_std:>8.1f} "
          f"{min(baseline_scores):>6} {max(baseline_scores):>6}")
    print(f"{'MLPlanner':<20} {ml_mean:>8.1f} {ml_std:>8.1f} "
          f"{min(ml_scores):>6} {max(ml_scores):>6}")
    print(f"{'='*55}")
    print(f"Delta: {delta:+.1f} points ({delta_pct:+.1f}%)")
    print(f"ML plan() avg: {ml_plan_ms:.1f}ms/round")

    if live_score:
        print(f"Live score: {live_score}")

    # Verdict
    if ml_mean > 393:
        print("\nVerdict: GO — MLPlanner exceeds target (393)")
    elif ml_mean >= 200:
        print("\nVerdict: PROMISING — MLPlanner functional but needs more training")
    elif ml_mean >= baseline_mean * 0.8:
        print("\nVerdict: CLOSE — within 20% of baseline, needs tuning")
    else:
        print("\nVerdict: NEEDS MORE TRAINING — significant gap to baseline")

        # Diagnostic: worst rounds
        if ml_results:
            worst = min(ml_results, key=lambda r: r["score"])
            print(f"\nWorst run: score={worst['score']} items={worst['items_delivered']} "
                  f"orders={worst['orders_completed']}")


def main():
    parser = argparse.ArgumentParser(description="Validate MLPlanner vs baseline")
    parser.add_argument("--recon", required=True, help="Recon JSON file")
    parser.add_argument("--checkpoint", help="ScorerMLP checkpoint (default: auto-detect)")
    parser.add_argument("--n-runs", type=int, default=5, help="Runs per planner")
    args = parser.parse_args()

    recon_path = Path(args.recon)
    if not recon_path.exists():
        print(f"Recon not found: {recon_path}")
        sys.exit(1)

    ckpt = Path(args.checkpoint) if args.checkpoint else None
    validate(recon_path, ckpt, args.n_runs)


if __name__ == "__main__":
    main()
