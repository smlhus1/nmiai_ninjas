"""Auto-grind: run all difficulties in a loop, track best scores, stop at target.

Usage:
    py auto_grind.py [--target 3031] [--mapf-nightmare mapf_plan.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("grind")

ACCESS_TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiI2MDFkN2QwMi0yZTViLTQxNjgtODZiZC02OGFlMjk0M2QzNDEi"
    "LCJlbWFpbCI6InN0aWFuNDJAZ21haWwuY29tIiwiZXhwIjoxNzc0MjAzOTQ0fQ."
    "fK5N9Q-thmwwCTj1uYsGLJhtFGq-S0nA0XU6QhqjiU8"
)

MAP_IDS = {
    "easy": "c89da2ec-3ca7-40c9-a3b1-8036fca3d0b7",
    "medium": "3c523f5e-160b-452c-9ffc-171ef1e845f5",
    "hard": "05ddc283-9097-4314-824c-90b3269a3d95",
    "expert": "c7c7f564-2496-4ab1-9179-7532979adcb4",
    "nightmare": "120c51da-c765-4bab-8b79-bba945a59e7c",
}

FINGERPRINTS = {
    "easy": "31642503",
    "medium": "6fb8097b",
    "hard": "8d88a034",
    "expert": "515edd5d",
    "nightmare": "74001e7f",
}


def get_best_scores() -> dict[str, int]:
    """Read best scores from recon files (excluding extended)."""
    bests: dict[str, int] = {}
    for diff, fp in FINGERPRINTS.items():
        best = 0
        for f in Path("logs").glob(f"{fp}_*recon*.json"):
            if "extended" in f.name:
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                s = d.get("final_score", d.get("score", 0))
                if isinstance(s, (int, float)) and s > best:
                    best = int(s)
            except Exception:
                pass
        bests[diff] = best
    return bests


def request_game(difficulty: str) -> str | None:
    """Request a game token from the API, return ws_url."""
    import subprocess
    map_id = MAP_IDS[difficulty]
    try:
        for attempt in range(4):
            result = subprocess.run(
                [
                    "curl.exe", "-s",
                    "https://api.ainm.no/games/request",
                    "-H", "content-type: application/json",
                    "-b", f"access_token={ACCESS_TOKEN}",
                    "-H", "origin: https://app.ainm.no",
                    "--data-raw", json.dumps({"map_id": map_id}),
                ],
                capture_output=True, text=True, timeout=15,
            )
            data = json.loads(result.stdout)
            if "ws_url" in data:
                return data["ws_url"]
            detail = str(data.get("detail", ""))
            # Per-game cooldown — wait and retry
            if "Cooldown" in detail and "Hourly" not in detail:
                import re
                m = re.search(r"(\d+)s", detail)
                wait = int(m.group(1)) + 2 if m else 65
                logger.info("Cooldown %ds for %s, waiting...", wait, difficulty)
                time.sleep(wait)
                continue
            # Hourly rate limit — back off for a long time
            if "Hourly" in detail or "limit" in detail.lower():
                import re
                m = re.search(r"(\d+)\s*games", detail)
                logger.warning("HOURLY LIMIT HIT — sleeping 30 minutes")
                time.sleep(1800)
                continue
            logger.error("No ws_url in response for %s: %s", difficulty, data)
            return None
        logger.error("Max retries for %s", difficulty)
        return None
    except Exception as e:
        logger.error("Failed to request %s game: %s", difficulty, e)
        return None


def run_game(ws_url: str, mapf_plan: str | None = None) -> int | None:
    """Run a game and return the score."""
    cmd = [sys.executable, "main.py", "--url", ws_url]
    if mapf_plan:
        cmd += ["--mapf", mapf_plan]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        # Parse score from output
        for line in result.stderr.split("\n") + result.stdout.split("\n"):
            if "Game over" in line and "Score:" in line:
                # Extract score from "Score: 326 | Rounds: ..."
                parts = line.split("Score:")[1].split("|")[0].strip()
                return int(parts)
        logger.error("No score found in output")
        if result.stderr:
            logger.error("stderr: %s", result.stderr[-500:])
        return None
    except subprocess.TimeoutExpired:
        logger.error("Game timed out")
        return None
    except Exception as e:
        logger.error("Game failed: %s", e)
        return None


def print_scoreboard(bests: dict[str, int], target: int, iteration: int) -> None:
    total = sum(bests.values())
    logger.info("=" * 50)
    logger.info("SCOREBOARD — Iteration %d", iteration)
    logger.info("-" * 50)
    for diff in ["easy", "medium", "hard", "expert", "nightmare"]:
        logger.info("  %-10s %4d", diff, bests[diff])
    logger.info("-" * 50)
    logger.info("  TOTAL:     %4d / %d  (gap: %d)", total, target, target - total)
    logger.info("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=3031)
    parser.add_argument("--mapf-nightmare", type=str, default=None,
                       help="MAPF plan file for nightmare")
    parser.add_argument("--difficulties", type=str, default="easy,medium,hard,expert,nightmare",
                       help="Comma-separated difficulties to grind")
    args = parser.parse_args()

    difficulties = [d.strip() for d in args.difficulties.split(",")]
    bests = get_best_scores()
    total = sum(bests.values())

    logger.info("Starting grind — target: %d, current: %d, gap: %d",
                args.target, total, args.target - total)
    print_scoreboard(bests, args.target, 0)

    if total >= args.target:
        logger.info("TARGET REACHED! %d >= %d", total, args.target)
        return

    iteration = 0
    while total < args.target:
        iteration += 1

        for diff in difficulties:
            logger.info("--- %s (iteration %d) ---", diff.upper(), iteration)

            ws_url = request_game(diff)
            if not ws_url:
                logger.warning("Skipping %s — no token", diff)
                time.sleep(5)
                continue

            mapf = args.mapf_nightmare if diff == "nightmare" else None
            score = run_game(ws_url, mapf_plan=mapf)

            if score is not None:
                if score > bests[diff]:
                    logger.info("NEW BEST %s: %d -> %d (+%d)",
                               diff, bests[diff], score, score - bests[diff])
                    bests[diff] = score
                else:
                    logger.info("%s: %d (best: %d)", diff, score, bests[diff])

            # Cooldown between games
            time.sleep(2)

        total = sum(bests.values())
        print_scoreboard(bests, args.target, iteration)

        if total >= args.target:
            logger.info("TARGET REACHED! %d >= %d", total, args.target)
            break

        # Pause between full rounds
        logger.info("Waiting 5s before next round...")
        time.sleep(5)


if __name__ == "__main__":
    main()
