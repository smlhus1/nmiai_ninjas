"""Probe loop: run live games to discover more orders, merge recons, replay best.

Each iteration:
1. Run live (reactive V2) → discovers new orders
2. Merge into master recon
3. Capture best MAPF plan from merged recon
4. Replay MAPF plan live → higher score (more orders known)

Usage:
    py probe_loop.py --difficulty nightmare --iterations 5
    py probe_loop.py --difficulty nightmare --iterations 10 --recon logs/74001e7f_2026-03-17_recon.json
"""
import json
import re
import subprocess
import sys
import time
import argparse
from pathlib import Path

API_URL = "https://api.ainm.no/games/request"
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
    "easy": "31642503", "medium": "6fb8097b", "hard": "8d88a034",
    "expert": "515edd5d", "nightmare": "74001e7f",
}


def get_token(difficulty: str) -> str | None:
    map_id = MAP_IDS[difficulty]
    for attempt in range(5):
        result = subprocess.run([
            "curl.exe", "-s", API_URL,
            "-H", "content-type: application/json",
            "-b", f"access_token={ACCESS_TOKEN}",
            "-H", "origin: https://app.ainm.no",
            "--data-raw", json.dumps({"map_id": map_id}),
        ], capture_output=True, text=True, timeout=30)

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None

        if "Cooldown" in data.get("detail", ""):
            wait = int(re.search(r"(\d+)", data["detail"]).group(1)) + 2
            print(f"  Cooldown {wait}s...", flush=True)
            time.sleep(wait)
            continue

        return data.get("ws_url")
    return None


def run_game(ws_url: str, mapf_plan: str | None = None) -> tuple[int, str | None]:
    """Run game, return (score, recon_path)."""
    cmd = ["py", "main.py", "--url", ws_url]
    if mapf_plan:
        cmd.extend(["--mapf", mapf_plan])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    output = result.stdout + result.stderr

    score = 0
    for line in output.split("\n"):
        if "Game over" in line and "Score:" in line:
            m = re.search(r"Score:\s*(\d+)", line)
            if m:
                score = int(m.group(1))

    # Find newest recon
    fp = FINGERPRINTS.get("nightmare", "74001e7f")
    recons = sorted(Path("logs").glob(f"{fp}_*_recon.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    recon_path = str(recons[0]) if recons else None
    return score, recon_path


def merge_recons(base_path: str, new_path: str, output_path: str) -> int:
    """Merge order sequences. Orders matched by position in sequence (not content)."""
    base = json.loads(Path(base_path).read_text())
    new = json.loads(Path(new_path).read_text())

    base_orders = base.get("order_sequence", [])
    new_orders = new.get("order_sequence", [])

    # Orders come in the same sequence every game (deterministic seed).
    # New game may see MORE orders (if it completes faster or runs longer).
    if len(new_orders) > len(base_orders):
        # Verify prefix matches
        match = True
        for i in range(min(len(base_orders), len(new_orders))):
            if base_orders[i].get("items_required") != new_orders[i].get("items_required"):
                match = False
                break

        if match:
            added = len(new_orders) - len(base_orders)
            base["order_sequence"] = new_orders
            print(f"  Extended: +{added} orders (total {len(new_orders)})", flush=True)
        else:
            print(f"  Order sequence mismatch — keeping base ({len(base_orders)} orders)", flush=True)
            return len(base_orders)
    else:
        print(f"  No new orders (base={len(base_orders)}, new={len(new_orders)})", flush=True)
        return len(base_orders)

    # Update final_score if new is higher
    new_score = new.get("final_score", 0)
    if new_score > base.get("final_score", 0):
        base["final_score"] = new_score

    Path(output_path).write_text(json.dumps(base, indent=2), encoding="utf-8")
    return len(base["order_sequence"])


def capture_best_plan(recon_path: str, output: str) -> int:
    """Run auto_best_plan.py and return sim score."""
    result = subprocess.run(
        ["py", "auto_best_plan.py", recon_path, output],
        capture_output=True, text=True, timeout=120,
    )
    output_text = result.stdout + result.stderr
    score = 0
    for line in output_text.split("\n"):
        if "score=" in line and "Plan saved" in line:
            m = re.search(r"score=(\d+)", line)
            if m:
                score = int(m.group(1))
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="nightmare")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--recon", help="Initial recon (auto-detect if omitted)")
    args = parser.parse_args()

    fp = FINGERPRINTS[args.difficulty]

    # Find initial recon
    if args.recon:
        master_recon = args.recon
    else:
        recons = sorted(Path("logs").glob(f"{fp}_*_recon.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not recons:
            print("No recon found. Run a game first.")
            sys.exit(1)
        master_recon = str(recons[0])

    merged_path = f"logs/{fp}_merged_recon.json"
    # Copy initial to merged
    Path(merged_path).write_text(Path(master_recon).read_text())

    n_orders = len(json.loads(Path(merged_path).read_text()).get("order_sequence", []))
    best_live = 0

    print(f"\n{'='*60}")
    print(f"PROBE LOOP: {args.iterations} iterations, {args.difficulty}")
    print(f"Initial: {master_recon} ({n_orders} orders)")
    print(f"{'='*60}\n")

    for i in range(args.iterations):
        print(f"--- Iteration {i+1}/{args.iterations} ({n_orders} orders) ---", flush=True)

        # Step 1: Probe game (reactive V2 — discovers new orders)
        print("  [1] Probe game...", flush=True)
        ws = get_token(args.difficulty)
        if not ws:
            print("  SKIP: no token", flush=True)
            continue

        score, recon_path = run_game(ws)
        print(f"  Probe score: {score}", flush=True)

        if recon_path:
            n_orders = merge_recons(merged_path, recon_path, merged_path)

        # Step 2: Capture best MAPF plan from merged recon
        print("  [2] Capture best plan...", flush=True)
        plan_path = f"mapf_plan_probe_{i+1}.json"
        sim_score = capture_best_plan(merged_path, plan_path)
        print(f"  Sim score: {sim_score}", flush=True)

        # Step 3: Replay live
        print("  [3] Replay live...", flush=True)
        ws2 = get_token(args.difficulty)
        if not ws2:
            print("  SKIP: no token for replay", flush=True)
            continue

        live_score, recon2 = run_game(ws2, mapf_plan=plan_path)
        print(f"  Live score: {live_score}", flush=True)

        if live_score > best_live:
            best_live = live_score
            print(f"  *** NEW BEST LIVE: {best_live} ***", flush=True)

        # Merge replay recon too (may have seen more orders)
        if recon2:
            n_orders = merge_recons(merged_path, recon2, merged_path)

        print(f"  Orders: {n_orders}, Best live: {best_live}\n", flush=True)

    print(f"\n{'='*60}")
    print(f"DONE. Best live: {best_live}, Orders discovered: {n_orders}")
    print(f"Merged recon: {merged_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
