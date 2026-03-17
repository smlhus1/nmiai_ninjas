"""
TASK-2-2: Sim-to-Live Gap Calibration

Verifies that simulator scores are a reliable training signal by comparing
BotAdapter sim scores against known live scores from recon files.

Usage:
    py -m oracle.sim_live_calibration
    py -m oracle.sim_live_calibration --pattern "logs/74001e7f_2026-03-1*_recon.json"
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from Simulering.offline.bot_adapter import BotAdapter
from Simulering.offline.simulator import Simulator


def calibrate(recon_paths: list[Path]) -> None:
    gaps: list[float] = []

    for path in sorted(recon_paths):
        recon = json.loads(path.read_text(encoding="utf-8"))
        live_score = recon.get("final_score")
        if live_score is None:
            print(f"[{path.name}] SKIP — no final_score")
            continue

        sim = Simulator.from_recon_data(recon)
        adapter = BotAdapter(suppress_logs=True)
        result = sim.run(adapter)
        sim_score = result["score"]
        adapter.reset()

        gap_pct = (sim_score - live_score) / max(live_score, 1) * 100
        gaps.append(gap_pct)
        print(f"[{path.name}] sim={sim_score}  live={live_score}  gap={gap_pct:+.1f}%")

    if not gaps:
        print("\nNo recon files with final_score found.")
        return

    mean = statistics.mean(gaps)
    std = statistics.stdev(gaps) if len(gaps) > 1 else 0.0

    print(f"\n{'='*50}")
    print(f"Files compared: {len(gaps)}")
    print(f"Mean gap: {mean:+.1f}%  Stddev: {std:.1f}%")

    if abs(mean) < 10:
        print("CONCLUSION: Sim-to-live gap OK (<10%) — sim is reliable training signal")
    elif abs(mean) < 20:
        print("WARNING: Sim-to-live gap 10-20% — usable but noisy")
    else:
        print("WARNING: Sim-to-live gap >20% — consider live-logging for training data")


def main():
    parser = argparse.ArgumentParser(description="Sim-to-Live Gap Calibration")
    parser.add_argument(
        "--pattern",
        default="logs/74001e7f_2026-03-1[1-6]_recon.json",
        help="Glob pattern for recon files (default: daily nightmare recons)",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in glob.glob(args.pattern)]
    if not paths:
        print(f"No files matching pattern: {args.pattern}")
        sys.exit(1)

    print(f"Calibrating sim vs live across {len(paths)} recon files...\n")
    calibrate(paths)


if __name__ == "__main__":
    main()
