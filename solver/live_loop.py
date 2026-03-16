"""Live optimization loop: search → capture → live → extend recon → repeat.

The core loop for pushing past 381 toward 1000+:
1. Run evolutionary search on current recon
2. Capture best genome as MAPF plan
3. Request game token and run live with MAPF replay
4. Capture new orders from live game (extends recon)
5. Merge new orders into recon
6. Repeat with more orders

Usage:
    py -m solver.live_loop --recon <initial_recon> [--iterations 5]
    py -m solver.live_loop --recon logs/74001e7f_2026-03-15_recon.json --iterations 3
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import shutil
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

# API config
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


def request_game_token(difficulty: str = "nightmare", max_retries: int = 5) -> str | None:
    """Request a new game token from the API. Handles cooldown. Returns WebSocket URL or None."""
    map_id = MAP_IDS.get(difficulty, MAP_IDS["nightmare"])
    cmd = [
        "curl.exe", "-s", API_URL,
        "-H", "content-type: application/json",
        "-b", f"access_token={ACCESS_TOKEN}",
        "-H", "origin: https://app.ainm.no",
        "--data-raw", json.dumps({"map_id": map_id}),
    ]

    for attempt in range(max_retries):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            data = json.loads(result.stdout)

            # Check for cooldown
            detail = data.get("detail", "")
            if "Cooldown" in detail or "cooldown" in detail:
                # Extract wait time
                import re
                match = re.search(r"(\d+)s", detail)
                wait = int(match.group(1)) + 2 if match else 60
                print(f"  Cooldown: waiting {wait}s (attempt {attempt+1})...", flush=True)
                time.sleep(wait)
                continue

            ws_url = data.get("url") or data.get("ws_url") or data.get("game_url")
            if ws_url:
                print(f"  Got game token: {ws_url[:80]}...", flush=True)
                return ws_url

            print(f"  API response: {result.stdout[:200]}", flush=True)
            return None
        except Exception as e:
            print(f"  Token request failed: {e}", flush=True)
            return None

    print(f"  Failed after {max_retries} attempts", flush=True)
    return None


def run_live_game(ws_url: str, mapf_plan: str | None = None, timeout: int = 180,
                  use_botadapter: bool = True) -> str | None:
    """Run the bot live and return path to new recon file (or None).

    use_botadapter: If True, use BotAdapter (max orders for recon capture).
                    If False, use MAPF replay (for score testing).
    """
    cmd = ["py", "main.py", "--url", ws_url]
    if not use_botadapter and mapf_plan:
        cmd.extend(["--mapf", mapf_plan])

    print(f"  Running live game...", flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr

        # Find the recon file from output
        for line in output.split("\n"):
            if "recon" in line.lower() and ".json" in line:
                # Try to extract path
                for word in line.split():
                    if word.endswith(".json") and "recon" in word:
                        if Path(word).exists():
                            return word
                        # Try logs/ prefix
                        p = Path("logs") / Path(word).name
                        if p.exists():
                            return str(p)

        # Look for newest recon file in logs/
        recon_files = sorted(Path("logs").glob("74001e7f_*_recon.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        if recon_files:
            newest = str(recon_files[0])
            print(f"  Found recon: {newest}", flush=True)
            return newest

        print(f"  No recon found in output", flush=True)
        return None
    except subprocess.TimeoutExpired:
        print(f"  Game timed out after {timeout}s", flush=True)
        return None
    except Exception as e:
        print(f"  Live game failed: {e}", flush=True)
        return None


def merge_recons(base_path: str, new_path: str, output_path: str) -> int:
    """Merge orders from new recon into base recon. Returns total order count."""
    with open(base_path) as f:
        base = json.load(f)
    with open(new_path) as f:
        new = json.load(f)

    base_orders = base.get("order_sequence", [])
    new_orders = new.get("order_sequence", [])

    # Find orders in new that aren't in base (by items_required match)
    base_signatures = set()
    for o in base_orders:
        sig = tuple(sorted(o.get("items_required", [])))
        base_signatures.add(sig)

    added = 0
    for o in new_orders:
        sig = tuple(sorted(o.get("items_required", [])))
        if sig not in base_signatures:
            base_orders.append(o)
            base_signatures.add(sig)
            added += 1

    base["order_sequence"] = base_orders
    Path(output_path).write_text(json.dumps(base, indent=2), encoding="utf-8")

    print(f"  Merged: {added} new orders, total={len(base_orders)}", flush=True)
    return len(base_orders)


def run_search(recon_path: str, pop: int = 30, gens: int = 30,
               workers: int = 12) -> tuple[str, int]:
    """Run evolutionary search. Returns (plan_path, score)."""
    print(f"  Searching: pop={pop}, gens={gens}, workers={workers}...", flush=True)
    cmd = [
        "py", "-m", "solver.capture_genome",
        "--recon", recon_path,
        "--search",
        "--pop", str(pop),
        "--gens", str(gens),
        "--workers", str(workers),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        output = result.stdout + result.stderr

        # Find plan file and score from output
        plan_path = None
        score = 0
        for line in output.split("\n"):
            if "Plan saved:" in line:
                parts = line.split("Plan saved:")[1].strip()
                plan_path = parts.split()[0]
            if "score=" in line and "Final" in line:
                for part in line.split():
                    if part.startswith("score="):
                        try:
                            score = int(part.split("=")[1].rstrip(","))
                        except ValueError:
                            pass

        if plan_path and Path(plan_path).exists():
            print(f"  Search result: score={score}, plan={plan_path}", flush=True)
            return plan_path, score

        # Check for plan files
        plan_files = sorted(Path(".").glob("mapf_plan_genome_*.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if plan_files:
            return str(plan_files[0]), score

        print(f"  Search failed, output:\n{output[-500:]}", flush=True)
        return None, 0
    except Exception as e:
        print(f"  Search failed: {e}", flush=True)
        return None, 0


def live_loop(
    initial_recon: str,
    iterations: int = 5,
    pop: int = 30,
    gens: int = 30,
    workers: int = 12,
    difficulty: str = "nightmare",
):
    """Main loop: search → capture → live → extend → repeat."""
    current_recon = initial_recon
    best_score = 0
    best_plan = None

    with open(current_recon) as f:
        recon = json.load(f)
    total_orders = len(recon.get("order_sequence", []))
    print(f"\n{'='*60}", flush=True)
    print(f"LIVE LOOP: {iterations} iterations", flush=True)
    print(f"Initial recon: {current_recon} ({total_orders} orders)", flush=True)
    print(f"{'='*60}\n", flush=True)

    for i in range(iterations):
        print(f"\n--- Iteration {i+1}/{iterations} ---", flush=True)
        print(f"Recon: {current_recon} ({total_orders} orders)", flush=True)

        # Step 1: Evolutionary search
        plan_path, score = run_search(current_recon, pop=pop, gens=gens, workers=workers)
        if not plan_path:
            print("  Search failed, skipping iteration", flush=True)
            continue

        if score > best_score:
            best_score = score
            best_plan = plan_path
            print(f"  NEW BEST SIM SCORE: {best_score}", flush=True)

        # Step 2: Request game token
        ws_url = request_game_token(difficulty)
        if not ws_url:
            print("  Could not get game token, skipping live", flush=True)
            continue

        # Step 3: Run live with BotAdapter (max orders for recon capture)
        # BotAdapter scores 300+ and captures more orders than genome replay
        new_recon = run_live_game(ws_url, mapf_plan=plan_path, use_botadapter=True)
        if not new_recon:
            print("  No recon captured from live game", flush=True)
            continue

        # Step 4: Merge new orders
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        merged_path = f"logs/74001e7f_{timestamp}_extended_recon.json"
        total_orders = merge_recons(current_recon, new_recon, merged_path)
        current_recon = merged_path

        print(f"  Extended recon to {total_orders} orders", flush=True)

        # Cooldown between iterations (API enforces 60s between games)
        if i < iterations - 1:
            print(f"  Waiting 65s for API cooldown...", flush=True)
            time.sleep(65)

    print(f"\n{'='*60}", flush=True)
    print(f"DONE: {iterations} iterations", flush=True)
    print(f"Best sim score: {best_score}", flush=True)
    print(f"Best plan: {best_plan}", flush=True)
    print(f"Final recon: {current_recon} ({total_orders} orders)", flush=True)
    print(f"{'='*60}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Live optimization loop")
    parser.add_argument("--recon", required=True, help="Initial recon file")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--pop", type=int, default=30)
    parser.add_argument("--gens", type=int, default=30)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--difficulty", default="nightmare")
    args = parser.parse_args()

    live_loop(
        args.recon,
        iterations=args.iterations,
        pop=args.pop,
        gens=args.gens,
        workers=args.workers,
        difficulty=args.difficulty,
    )


if __name__ == "__main__":
    main()
