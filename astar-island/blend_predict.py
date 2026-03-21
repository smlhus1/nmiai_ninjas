"""Build blended predictions: weighted historical GT + current round observations."""

import json
import numpy as np
import glob
import requests
import time
from collections import defaultdict

CODE_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2MDFkN2QwMi0yZTViLTQxNjgtODZiZC02OGFlMjk0M2QzNDEiLCJlbWFpbCI6InN0aWFuNDJAZ21haWwuY29tIiwiZXhwIjoxNzc0MjAzOTQ0fQ.fK5N9Q-thmwwCTj1uYsGLJhtFGq-S0nA0XU6QhqjiU8"
BASE = "https://api.ainm.no"
s = requests.Session()
s.cookies.set("access_token", TOKEN)
s.headers["Origin"] = "https://app.ainm.no"


def extract_features(ig, y, x, settlements):
    ic = CODE_TO_CLASS.get(ig[y][x], 0)
    d = min((abs(y - sy) + abs(x - sx) for sy, sx in settlements), default=99)
    d = min(d, 8)
    ns = min(sum(1 for sy, sx in settlements if abs(y - sy) + abs(x - sx) <= 2), 2)
    nf = min(sum(1 for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                 if 0 <= y + dy < 40 and 0 <= x + dx < 40 and ig[y + dy][x + dx] == 4), 2)
    no = int(any(0 <= y + dy < 40 and 0 <= x + dx < 40 and ig[y + dy][x + dx] == 10
                 for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]))
    return (ic, d, ns, nf, no)


def get_transition_vector(obs_data, round_detail):
    """Compute transition vector [E->S, S->S, S->E, F->S] from observations."""
    T = np.zeros((6, 6))
    for obs in obs_data:
        seed = obs["seed"]
        ig = round_detail["initial_states"][seed]["grid"]
        grid = obs["grid"]
        vy, vx = obs.get("vy", 0), obs.get("vx", 0)
        for dy in range(len(grid)):
            for dx in range(len(grid[0])):
                ay, ax = vy + dy, vx + dx
                if ay >= 40 or ax >= 40:
                    continue
                start = CODE_TO_CLASS.get(ig[ay][ax], 0)
                end = CODE_TO_CLASS.get(grid[dy][dx], 0)
                T[start, end] += 1
    # Extract key rates
    e2s = 100 * T[0, 1] / T[0].sum() if T[0].sum() > 0 else 0
    s2s = 100 * T[1, 1] / T[1].sum() if T[1].sum() > 0 else 0
    s2e = 100 * T[1, 0] / T[1].sum() if T[1].sum() > 0 else 0
    f2s = 100 * T[4, 1] / T[4].sum() if T[4].sum() > 0 else 0
    return np.array([e2s, s2s, s2e, f2s])


def build_feature_model_from_gt(gt_data, round_detail):
    """Build feature model from ground truth."""
    model = defaultdict(list)
    for seed_str, gt_array in gt_data.items():
        seed = int(seed_str)
        if seed >= len(round_detail["initial_states"]):
            continue
        ig = round_detail["initial_states"][seed]["grid"]
        gt = np.array(gt_array)
        settlements = [(y, x) for y in range(40) for x in range(40) if ig[y][x] in (1, 2)]
        for y in range(40):
            for x in range(40):
                key = extract_features(ig, y, x, settlements)
                model[key].append(gt[y, x])
    return {k: np.mean(v, axis=0) for k, v in model.items() if len(v) >= 2}


# Historical rounds
historical = [
    {"name": "R1", "vec": np.array([0.5, 57.2, 41.3, 1.2]),
     "gt": "astar-island/data/ground_truth_r1.json", "rd": "data/astar_round1.json"},
    {"name": "R2", "vec": np.array([19.1, 43.2, 38.0, 19.8]),
     "gt": "astar-island/data/ground_truth_r2.json", "rd": "data/astar_round2.json"},
    # R3 skipped — brutal outlier (1.8% survival, 0 expansion)
    {"name": "R4", "vec": np.array([7.8, 27.0, 43.8, 8.5]),
     "gt": "astar-island/data/ground_truth_r4.json", "rd": None},
]

# Get active round
rounds = s.get(f"{BASE}/astar-island/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)
if not active:
    print("No active round!")
    exit(1)

round_id = active["id"]
print(f"Round {active['round_number']}: {round_id}")
detail = s.get(f"{BASE}/astar-island/rounds/{round_id}").json()

# Load observations
obs_files = sorted(glob.glob("astar-island/data/obs_r*.json"))
latest_obs = obs_files[-1] if obs_files else None
if not latest_obs:
    print("No observations found!")
    exit(1)
print(f"Using observations: {latest_obs}")
with open(latest_obs) as f:
    raw_obs = json.load(f)

# Compute current round transition vector
current_vec = get_transition_vector(raw_obs, detail)
print(f"Current round: E->S={current_vec[0]:.1f}% S->S={current_vec[1]:.1f}% S->E={current_vec[2]:.1f}% F->S={current_vec[3]:.1f}%")

# Compute weights for historical rounds
weights = {}
for h in historical:
    dist = np.sqrt(np.sum((current_vec - h["vec"]) ** 2))
    weights[h["name"]] = 1.0 / (dist + 1.0)
    print(f"  {h['name']}: distance={dist:.1f}, raw_weight={weights[h['name']]:.3f}")

total_w = sum(weights.values())
for name in weights:
    weights[name] /= total_w
    print(f"  {name}: normalized={weights[name]:.3f}")

# Build weighted historical model
model_blend = defaultdict(lambda: np.zeros(6))
model_count = defaultdict(float)

for h in historical:
    w = weights[h["name"]]
    try:
        with open(h["gt"]) as f:
            gt_data = json.load(f)
        if h["rd"]:
            with open(h["rd"]) as f:
                rd = json.load(f)
        else:
            my = s.get(f"{BASE}/astar-island/my-rounds").json()
            match = next(r for r in my if r["round_number"] == 4)
            rd = s.get(f"{BASE}/astar-island/rounds/{match['id']}").json()

        for seed_str, gt_array in gt_data.items():
            seed = int(seed_str)
            if seed >= len(rd["initial_states"]):
                continue
            ig = rd["initial_states"][seed]["grid"]
            gt = np.array(gt_array)
            settlements = [(y, x) for y in range(40) for x in range(40) if ig[y][x] in (1, 2)]
            for y in range(40):
                for x in range(40):
                    key = extract_features(ig, y, x, settlements)
                    model_blend[key] += w * gt[y, x]
                    model_count[key] += w
    except Exception as e:
        print(f"  {h['name']}: error {e}")

avg_blend = {k: model_blend[k] / model_count[k] for k in model_blend if model_count[k] > 0}

# Simple fallback
simple_blend = defaultdict(lambda: np.zeros(6))
simple_count = defaultdict(float)
for key in model_blend:
    sk = (key[0], key[1])
    simple_blend[sk] += model_blend[key]
    simple_count[sk] += model_count[key]
simple_avg = {k: v / simple_count[k] for k, v in simple_blend.items()}

print(f"Blended model: {len(avg_blend)} keys")

# Build current-round observation model
observations = defaultdict(list)
for obs in raw_obs:
    seed = obs["seed"]
    grid = obs["grid"]
    vy, vx = obs.get("vy", 0), obs.get("vx", 0)
    for dy in range(len(grid)):
        for dx in range(len(grid[0])):
            ay, ax = vy + dy, vx + dx
            if ay >= 40 or ax >= 40:
                continue
            observations[(seed, ay, ax)].append(CODE_TO_CLASS.get(grid[dy][dx], 0))

model_current = defaultdict(list)
for (seed, y, x), obs_list in observations.items():
    ig = detail["initial_states"][seed]["grid"]
    settlements = [(sy, sx) for sy in range(40) for sx in range(40) if ig[sy][sx] in (1, 2)]
    key = extract_features(ig, y, x, settlements)
    for cls in obs_list:
        one_hot = np.zeros(6)
        one_hot[cls] = 1.0
        model_current[key].append(one_hot)
avg_current = {k: np.mean(v, axis=0) for k, v in model_current.items() if len(v) >= 2}

# Build final predictions: 60% current obs + 40% blended historical
predictions = {}
for seed in range(5):
    ig = detail["initial_states"][seed]["grid"]
    settlements = [(y, x) for y in range(40) for x in range(40) if ig[y][x] in (1, 2)]
    pred = np.zeros((40, 40, 6))

    for y in range(40):
        for x in range(40):
            key = extract_features(ig, y, x, settlements)
            direct = observations.get((seed, y, x), [])

            # Current round prediction
            cur_pred = None
            if len(direct) > 0:
                counts = np.bincount(direct, minlength=6).astype(float)
                if key in avg_current:
                    prior = avg_current[key] * 3
                    cur_pred = (prior + counts) / (3 + len(direct))
                else:
                    cur_pred = (counts + 0.5) / (len(direct) + 3)
            elif key in avg_current:
                cur_pred = avg_current[key]

            # Blended historical prediction
            hist_pred = avg_blend.get(key, simple_avg.get((key[0], key[1]), np.full(6, 1 / 6)))

            if cur_pred is not None:
                pred[y, x] = 0.6 * cur_pred + 0.4 * hist_pred
            else:
                pred[y, x] = hist_pred

    pred = np.maximum(pred, 0.005)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    predictions[seed] = pred

    init_cls = np.array([[CODE_TO_CLASS.get(ig[y][x], 0) for x in range(40)] for y in range(40)])
    n_changed = np.sum(np.argmax(pred, axis=-1) != init_cls)
    n_settle = np.sum(np.argmax(pred, axis=-1) == 1)
    print(f"Seed {seed}: {n_changed} changed, {n_settle} settlements (blended)")

# Save
pred_data = {str(s): p.tolist() for s, p in predictions.items()}
with open("astar-island/data/r2_predictions.json", "w") as f:
    json.dump({"predictions": pred_data}, f)
print("\nBlended predictions saved (NOT submitted)")
