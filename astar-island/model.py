"""Analytical prediction model for Astar Island.

Builds feature-based lookup table from ground truth data.
For new rounds: uses R1 as prior, updates with observations.

Score on R1: ~87 (overfitted), expected R2: 50-70 without observations.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

CODE_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
K = 6
FLOOR = 0.005


def extract_features(init_grid, y, x, settlements):
    """Extract features for a cell."""
    init_cls = CODE_TO_CLASS.get(init_grid[y][x], 0)
    dist = min((abs(y-sy) + abs(x-sx) for sy, sx in settlements), default=99)
    dist = min(dist, 8)
    n_settle_r2 = min(sum(1 for sy, sx in settlements if abs(y-sy)+abs(x-sx) <= 2), 2)
    n_forest = min(sum(1 for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]
                       if 0 <= y+dy < 40 and 0 <= x+dx < 40 and init_grid[y+dy][x+dx] == 4), 2)
    near_ocean = int(any(0 <= y+dy < 40 and 0 <= x+dx < 40 and init_grid[y+dy][x+dx] == 10
                         for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]))
    return (init_cls, dist, n_settle_r2, n_forest, near_ocean)


class AnalyticalModel:
    def __init__(self):
        self.model = {}       # feature_key -> avg GT distribution
        self.simple_model = {} # (init_cls, dist) -> avg GT distribution

    def train(self, round_data, gt_data):
        """Train from ground truth data."""
        raw = defaultdict(list)

        for seed_str, gt_array in gt_data.items():
            seed = int(seed_str)
            if seed >= len(round_data["initial_states"]):
                continue
            ig = round_data["initial_states"][seed]["grid"]
            gt = np.array(gt_array)
            settlements = [(y, x) for y in range(40) for x in range(40) if ig[y][x] in (1, 2)]

            for y in range(40):
                for x in range(40):
                    key = extract_features(ig, y, x, settlements)
                    raw[key].append(gt[y, x])

        self.model = {k: np.mean(v, axis=0) for k, v in raw.items() if len(v) >= 2}

        simple_raw = defaultdict(list)
        for k, v in raw.items():
            simple_raw[(k[0], k[1])].extend(v)
        self.simple_model = {k: np.mean(v, axis=0) for k, v in simple_raw.items()}

    def predict_cell(self, key):
        """Get prediction for a feature key."""
        if key in self.model:
            return self.model[key].copy()
        simple = (key[0], key[1])
        if simple in self.simple_model:
            return self.simple_model[simple].copy()
        return np.full(K, 1/K)

    def predict_grid(self, init_grid, observations=None, prior_weight=3.0):
        """Predict full 40x40x6 tensor.

        observations: dict (y, x) -> list of class indices
        """
        settlements = [(y, x) for y in range(40) for x in range(40) if init_grid[y][x] in (1, 2)]
        pred = np.zeros((40, 40, K))

        for y in range(40):
            for x in range(40):
                key = extract_features(init_grid, y, x, settlements)
                prior = self.predict_cell(key)

                if observations and (y, x) in observations and len(observations[(y, x)]) > 0:
                    # Bayesian update: prior + observations
                    obs = observations[(y, x)]
                    counts = np.bincount(obs, minlength=K).astype(float)
                    n = len(obs)
                    pseudo = prior * prior_weight * K
                    pred[y, x] = (pseudo + counts) / (prior_weight * K + n)
                else:
                    pred[y, x] = prior

        # Floor and normalize
        pred = np.maximum(pred, FLOOR)
        pred = pred / pred.sum(axis=-1, keepdims=True)
        return pred

    def save(self, path):
        """Save model to JSON."""
        data = {
            "model": {str(k): v.tolist() for k, v in self.model.items()},
            "simple_model": {str(k): v.tolist() for k, v in self.simple_model.items()},
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path):
        """Load model from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.model = {eval(k): np.array(v) for k, v in data["model"].items()}
        self.simple_model = {eval(k): np.array(v) for k, v in data["simple_model"].items()}


def train_from_r1():
    """Train model from R1 ground truth and save."""
    with open('data/astar_round1.json') as f:
        round_data = json.load(f)
    with open('astar-island/data/ground_truth_r1.json') as f:
        gt_data = json.load(f)

    model = AnalyticalModel()
    model.train(round_data, gt_data)
    model.save('astar-island/data/model_r1.json')
    print(f"Model trained: {len(model.model)} feature keys, {len(model.simple_model)} simple keys")
    return model


if __name__ == "__main__":
    model = train_from_r1()

    # Validate on R1
    with open('data/astar_round1.json') as f:
        detail = json.load(f)
    with open('astar-island/data/ground_truth_r1.json') as f:
        gt_all = json.load(f)

    scores = []
    for seed in range(5):
        gt = np.array(gt_all[str(seed)])
        ig = detail["initial_states"][seed]["grid"]
        pred = model.predict_grid(ig)

        entropy = -np.sum(gt * np.log(gt + 1e-10), axis=-1)
        kl = np.sum(gt * np.log((gt + 1e-10) / (pred + 1e-10)), axis=-1)
        mask = entropy > 0.01
        wkl = np.sum(entropy[mask] * kl[mask]) / np.sum(entropy[mask])
        s = max(0, min(100, 100 * np.exp(-3 * wkl)))
        scores.append(s)
        print(f"Seed {seed}: {s:.1f}")

    print(f"\nAverage: {np.mean(scores):.1f}")
