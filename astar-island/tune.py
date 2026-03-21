"""Auto-tune C++ simulator params against R1 ground truth."""

import json
import numpy as np
import subprocess
import os
import tempfile
from itertools import product

# Load ground truth
with open('astar-island/data/ground_truth_r1.json') as f:
    gt_all = json.load(f)

with open('data/astar_round1.json') as f:
    detail = json.load(f)

CODE_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def score_prediction(sim_pred, gt):
    """Calculate entropy-weighted KL score."""
    entropy = -np.sum(gt * np.log(gt + 1e-10), axis=-1)
    kl = np.sum(gt * np.log((gt + 1e-10) / (sim_pred + 1e-10)), axis=-1)
    mask = entropy > 0.01
    if mask.sum() == 0:
        return 100.0
    weighted_kl = np.sum(entropy[mask] * kl[mask]) / np.sum(entropy[mask])
    return max(0, min(100, 100 * np.exp(-3 * weighted_kl)))


def run_sim(params_override, seed_index=0, n_runs=2000):
    """Run C++ sim with modified params (edit source, rebuild, run)."""
    # Read current source
    with open('astar-island/sim.cpp', 'r') as f:
        src = f.read()

    # Replace params
    for key, value in params_override.items():
        # Find and replace the param value in source
        import re
        pattern = rf'(double {key}\s*=\s*)[\d.]+;'
        replacement = rf'\g<1>{value};'
        src_new = re.sub(pattern, src, count=1)
        if src_new != src:
            src = src_new
        else:
            # Try int params
            pattern = rf'(int {key}\s*=\s*)\d+;'
            replacement = rf'\g<1>{int(value)};'
            src = re.sub(pattern, replacement, src)

    # Write modified source
    with open('astar-island/sim_tune.cpp', 'w') as f:
        f.write(src)

    # Build
    result = subprocess.run(
        ['powershell.exe', '-NoProfile', '-Command',
         "cd 'C:\\Projects\\Personlig\\NmIAi\\astar-island'; "
         "& 'C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat' > $null 2>&1; "
         "cl /EHsc /O2 /std:c++17 /MT /openmp sim_tune.cpp /Fe:sim_tune.exe 2>&1"],
        capture_output=True, text=True, cwd='astar-island'
    )

    if not os.path.exists('astar-island/sim_tune.exe'):
        print(f"Build failed: {result.stderr[:200]}")
        return -1

    # Run
    out_file = 'astar-island/data/sim_tune_out.json'
    result = subprocess.run(
        ['astar-island/sim_tune.exe', 'data/astar_round1.json',
         '--runs', str(n_runs), '--seed-index', str(seed_index),
         '--output', out_file],
        capture_output=True, text=True, timeout=30
    )

    if not os.path.exists(out_file):
        return -1

    with open(out_file) as f:
        sim = json.load(f)

    sim_pred = np.array(sim['predictions'][str(seed_index)])
    gt = np.array(gt_all[str(seed_index)])

    return score_prediction(sim_pred, gt)


# Instead of rebuilding, let's just do a grid search by modifying the source directly
# Actually, faster: use the EXISTING sim.exe and just try different approaches

# Let's compute what the "ideal" prediction would be using the distance-based model
# from our analysis, without needing the C++ sim at all

print("=== Distance-based analytical model ===")
print("Using R1 ground truth to build optimal distance-based prediction")
print()

# Learn: for each (initial_class, distance_to_nearest_settlement), what's the GT distribution?
from collections import defaultdict

model = defaultdict(list)  # (init_cls, dist_bucket) -> [gt_probs]

for seed in range(5):
    init_grid = detail['initial_states'][seed]['grid']
    gt = np.array(gt_all[str(seed)])

    settlements = [(y, x) for y in range(40) for x in range(40)
                   if init_grid[y][x] in (1, 2)]

    for y in range(40):
        for x in range(40):
            init_cls = CODE_TO_CLASS.get(init_grid[y][x], 0)
            if not settlements:
                dist = 99
            else:
                dist = min(abs(y-sy) + abs(x-sx) for sy, sx in settlements)
            dist_bucket = min(dist, 10)
            model[(init_cls, dist_bucket)].append(gt[y, x])

# Compute average GT per bucket
avg_model = {}
for key, vals in model.items():
    avg_model[key] = np.mean(vals, axis=0)

# Now predict using this model
scores = []
for seed in range(5):
    init_grid = detail['initial_states'][seed]['grid']
    gt = np.array(gt_all[str(seed)])

    settlements = [(y, x) for y in range(40) for x in range(40)
                   if init_grid[y][x] in (1, 2)]

    pred = np.zeros((40, 40, 6))
    for y in range(40):
        for x in range(40):
            init_cls = CODE_TO_CLASS.get(init_grid[y][x], 0)
            if not settlements:
                dist = 99
            else:
                dist = min(abs(y-sy) + abs(x-sx) for sy, sx in settlements)
            dist_bucket = min(dist, 10)
            key = (init_cls, dist_bucket)
            if key in avg_model:
                pred[y, x] = avg_model[key]
            else:
                pred[y, x] = np.full(6, 1/6)

    # Floor and normalize
    pred = np.maximum(pred, 0.005)
    pred = pred / pred.sum(axis=-1, keepdims=True)

    s = score_prediction(pred, gt)
    scores.append(s)
    print(f"Seed {seed}: score={s:.1f}")

print(f"\nAverage: {np.mean(scores):.1f}")
print("(This is the ceiling for a distance-based model without per-cell variation)")

# Save this prediction for comparison
pred_data = {}
for seed in range(5):
    init_grid = detail['initial_states'][seed]['grid']
    settlements = [(y, x) for y in range(40) for x in range(40)
                   if init_grid[y][x] in (1, 2)]
    pred = np.zeros((40, 40, 6))
    for y in range(40):
        for x in range(40):
            init_cls = CODE_TO_CLASS.get(init_grid[y][x], 0)
            dist = min(abs(y-sy) + abs(x-sx) for sy, sx in settlements) if settlements else 99
            key = (init_cls, min(dist, 10))
            pred[y, x] = avg_model.get(key, np.full(6, 1/6))
    pred = np.maximum(pred, 0.005)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    pred_data[str(seed)] = pred.tolist()

with open('astar-island/data/sim_latest.json', 'w') as f:
    json.dump({"predictions": pred_data}, f)
print("\nSaved analytical model predictions to sim_latest.json")
