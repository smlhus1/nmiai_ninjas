# NM i AI 2026 — Team Ninjas

Competition entry for [NM i AI 2026](https://ainm.no) (March 19-22). Three challenges, one repo.

## Challenges

| Challenge | Directory | Score | Description |
|-----------|-----------|-------|-------------|
| **Grocery Bot** | `bot/`, `Simulering/`, `solver/`, `cpp_solver/` | 381 (nightmare) | WebSocket bot navigating a grocery store grid |
| **Astar Island** | `astar-island/` | **avg ~91** | Terrain prediction with ABC-calibrated simulation |
| **Object Detection** | `obj_detect/` | **0.8857** | Detect grocery products on store shelves |

---

## Object Detection (NorgesGruppen Data)

**Score: 0.8857** | [Weights on HuggingFace](https://huggingface.co/smlhus/nmiai-grocery-detection) | [Details](obj_detect/README.md)

2x YOLOv8 ONNX ensemble (1280px + 640px) with OIV7 pretrain, tiled inference, and WBF fusion. Trained on 248 shelf images, 356 product categories.

```bash
# Training pipeline (Google Colab with A100)
# See obj_detect/colab_train_v7.py for full steps
```

---

## Grocery Bot

WebSocket bot for the grid-based grocery store challenge. Bots navigate aisles, pick items from shelves, and deliver orders against a 300-round clock.

| Map | Grid | Bots | Best Score |
|-----|------|------|------------|
| Easy | 12x10 | 1 | 124 |
| Medium | 16x12 | 3 | 151 |
| Hard | 22x14 | 5 | 139 |
| Expert | 22x14 | 10 | 118 |
| Nightmare | 30x18 | 20 | 393 |

Leaderboard = sum of best score across all difficulties.

```bash
py -m pip install -r requirements.txt
py main.py --url "wss://game.ainm.no/ws?token=<JWT>"

# Offline testing
py -m Simulering.offline.run_offline --latest easy
py -m pytest tests/ Simulering/ -v
```

### Architecture

```
main.py (WebSocket)
  -> Coordinator (bot/coordinator.py)
       +-- PathEngine (A* + BFS cache)
       +-- PIBTResolver (collision-free movement)
       +-- TaskPlanner (strategic assignment)
       |     +-- RouteBuilder (multi-item TSP routes)
       |     +-- Hungarian (optimal bot-to-route matching)
       +-- ActionResolver (tasks -> move/pick/drop)
       +-- Recon/Replay (two-pass optimization)

Simulering/offline/    Offline simulator (~1ms/game)
solver/                Time-space A* planner
cpp_solver/            C++ MAPF planner (zone partitioning)
ml/                    ML-based planning experiments
```

---

## Astar Island

Terrain prediction challenge. Given an island's initial state, predict what each cell becomes after 50 years of simulation. 50 observation queries per round, 5 seeds per round.

### Scores (R1–R21)

| Round | Type | Score |
|-------|------|-------|
| R17 | Stable (93% survival) | 91.9 |
| R18 | Expansive (97% survival, 32% E→S) | 88.3 |
| R19 | Ultra-brutal (100% S→E) | 94.5 |
| R20 | Ultra-brutal (100% S→E) | 92.0 |
| R21 | Ultra-brutal (97% S→E) | 91.7 |

### Approach

4-phase Dirichlet pipeline:
1. **Observe** — 50 viewport queries allocated across seeds
2. **ABC Parameter Estimation** — 200 rejection samples calibrate simulator params against observations (~8s)
3. **Cross-seed Model** — feature-based transfer learning across seeds and historical rounds
4. **Dirichlet Prediction** — Jeffreys prior + historical + cross-seed + ABC-calibrated MC sims → posterior

Key innovations:
- ABC-calibrated simulator as prediction source (SIM_SCALE=5, +0.2–0.4 across all rounds)
- Per-class bias correction multipliers (tuned on 9-round LOO)
- Automated night watch: observe → solve → submit → Discord notification

### Run

```bash
pip install -r astar-island/requirements.txt
export ASTAR_TOKEN="<your token>"   # or add to astar-island/.env

# Backtest against ground truth
py astar-island/solver_v3.py --backtest 4

# Submit to active round
py astar-island/solver_v3.py --submit

# Automated pipeline
py astar-island/night_watch.py
```

### Files

```
astar-island/
  solver_v3.py         Main solver (4-phase Dirichlet pipeline)
  simulator.py         Local game engine (ABC param estimation)
  night_watch.py       Auto observe/submit/score collection
  data/                Ground truth, observations, round data (R1-R21)
  experiment_*.py      60+ parameter tuning experiments
```

---

## Project Structure

```
bot/                   Grocery bot core
Simulering/            Offline simulator + optimizer
solver/                Python MAPF planner
cpp_solver/            C++ MAPF planner
ml/                    ML planning experiments
nightmare_lab/         Nightmare map optimizer
astar-island/          Astar Island solver
obj_detect/            Object detection pipeline
research/              Research reports (all challenges)
tests/                 Bot unit tests
logs/                  Game recon logs
```

## Environment

- Python 3.13 (Windows — use `py` not `python`)
- Dependencies: `websockets`, `scipy`, `numpy`, `pytest`
- Object detection: see `obj_detect/requirements.txt`
