"""Night watch: auto-observe, submit, and collect scores.

Designed to be called from Claude Code cron. Prints ACTION markers that
the cron prompt interprets and acts on (e.g. sending Discord messages).

Actions:
- SUBMIT_OK:<round_num> — solver ran and submitted successfully
- NEW_SCORE:<round_num>:<score>:<vec> — new score available, GT saved
- IDLE — nothing to do
- WAITING:<round_num> — round active but budget spent
"""

import sys
import os
import json
import requests
import numpy as np
import subprocess
from datetime import datetime
from collections import defaultdict

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2MDFkN2QwMi0yZTViLTQxNjgtODZiZC02OGFlMjk0M2QzNDEiLCJlbWFpbCI6InN0aWFuNDJAZ21haWwuY29tIiwiZXhwIjoxNzc0MjAzOTQ0fQ.fK5N9Q-thmwwCTj1uYsGLJhtFGq-S0nA0XU6QhqjiU8"
BASE = "https://api.ainm.no"
LOG = "astar-island/data/night_log.txt"
CODE_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def make_session():
    s = requests.Session()
    s.cookies.set("access_token", TOKEN)
    s.headers["Origin"] = "https://app.ainm.no"
    return s

def analyze_round(s, round_info):
    """Analyze a completed round: compute transition vec, character description."""
    rn = round_info["round_number"]
    rid = round_info["id"]
    score = round_info.get("round_score", 0)
    seeds = round_info.get("seed_scores", [])

    gt_path = f"astar-island/data/ground_truth_r{rn}.json"
    rd_path = f"data/astar_round{rn}.json"

    if not os.path.exists(rd_path):
        return f"R{rn}={score:.1f} (no round data for analysis)"

    rd = json.load(open(rd_path))
    gt_data = json.load(open(gt_path))

    # Compute transition rates
    T = np.zeros((6, 6))
    for seed_str, gt_arr in gt_data.items():
        seed = int(seed_str)
        if seed >= len(rd["initial_states"]):
            continue
        ig = rd["initial_states"][seed]["grid"]
        gt_am = np.argmax(np.array(gt_arr), axis=-1)
        for y in range(40):
            for x in range(40):
                T[CODE_TO_CLASS.get(ig[y][x], 0), gt_am[y, x]] += 1

    e2s = 100 * (T[0, 1] + T[0, 2]) / T[0].sum() if T[0].sum() > 0 else 0
    s2s = 100 * T[1, 1] / T[1].sum() if T[1].sum() > 0 else 0
    s2e = 100 * T[1, 0] / T[1].sum() if T[1].sum() > 0 else 0
    f2s = 100 * (T[4, 1] + T[4, 2]) / T[4].sum() if T[4].sum() > 0 else 0

    # Character description
    if s2e > 80:
        char = "Ultra-brutal winter"
    elif s2e > 60:
        char = "Brutal winter"
    elif s2s > 50:
        char = "Stable"
    elif e2s > 10:
        char = "High expansion"
    else:
        char = "Moderate"

    vec = f"E->S={e2s:.1f}% S->S={s2s:.1f}% S->E={s2e:.1f}% F->S={f2s:.1f}%"
    seed_str = " ".join(f"{s:.1f}" for s in seeds) if seeds else "?"

    return f"R{rn}={score:.1f} ({char}). Seeds: [{seed_str}]. {vec}"

def main():
    s = make_session()

    # Check for active round
    rounds = s.get(f"{BASE}/astar-island/rounds").json()
    active = next((r for r in rounds if r["status"] == "active"), None)

    if active:
        round_num = active["round_number"]
        budget = s.get(f"{BASE}/astar-island/budget").json()
        remaining = budget["queries_max"] - budget["queries_used"]

        if remaining > 0:
            log(f"R{round_num} ACTIVE with {remaining} queries — running solver")

            # Save round data
            detail = s.get(f"{BASE}/astar-island/rounds/{active['id']}").json()
            rd_path = f"data/astar_round{round_num}.json"
            with open(rd_path, "w") as f:
                json.dump(detail, f)

            # Run solver + submit
            result = subprocess.run(
                [sys.executable, "astar-island/solver_v3.py", "--submit"],
                cwd="C:/Projects/Personlig/NmIAi",
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                log(f"R{round_num} submitted OK")
                print(f"SUBMIT_OK:{round_num}")
            else:
                log(f"R{round_num} FAILED: {result.stderr[-300:]}")
                print(f"SUBMIT_FAIL:{round_num}")
        else:
            print(f"WAITING:{round_num}")
        return

    # No active round — check for unsaved scores
    my = s.get(f"{BASE}/astar-island/my-rounds").json()
    new_scores = []

    for r in my:
        rn = r["round_number"]
        score = r.get("round_score")
        gt_path = f"astar-island/data/ground_truth_r{rn}.json"

        if score is not None and not os.path.exists(gt_path):
            log(f"R{rn} score={score:.1f} — saving GT")
            try:
                gt_all = {}
                for seed in range(5):
                    data = s.get(f"{BASE}/astar-island/analysis/{r['id']}/{seed}").json()
                    gt_all[seed] = np.array(data["ground_truth"])
                with open(gt_path, "w") as f:
                    json.dump({str(k): v.tolist() for k, v in gt_all.items()}, f)
                log(f"R{rn} GT saved")

                analysis = analyze_round(s, r)
                print(f"NEW_SCORE:{rn}:{score:.1f}:{analysis}")
                new_scores.append(rn)
            except Exception as e:
                log(f"R{rn} GT save failed: {e}")

    if not new_scores:
        print("IDLE")

if __name__ == "__main__":
    main()
