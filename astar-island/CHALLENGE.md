# Astar Island — Viking Civilisation Prediction

## What
Observer a stochastic Norse civilisation simulator through limited viewports. Predict the probability distribution of 6 terrain classes across the entire 40x40 map after 50 simulated years.

## Key Numbers
| Param | Value |
|-------|-------|
| Map size | 40x40 |
| Seeds per round | 5 |
| Queries per round | 50 (shared across all seeds) |
| Max viewport | 15x15 |
| Classes | 6 (Empty, Settlement, Port, Ruin, Forest, Mountain) |
| Prediction format | H x W x 6 probability tensor (sums to 1.0 per cell) |
| Round duration | ~2h 45min |
| Scoring | Entropy-weighted KL divergence → exp(-3*kl) → 0-100 |

## Terrain Classes

| Index | Class | Internal codes | Behaviour |
|-------|-------|---------------|-----------|
| 0 | Empty | Ocean(10), Plains(11), Empty(0) | STATIC — never changes |
| 1 | Settlement | 1 | DYNAMIC — grows, expands, gets raided |
| 2 | Port | 2 | DYNAMIC — coastal settlement with harbour |
| 3 | Ruin | 3 | DYNAMIC — collapsed settlement |
| 4 | Forest | 4 | MOSTLY STATIC — can reclaim ruined land |
| 5 | Mountain | 5 | STATIC — never changes |

## Simulation Phases (per year, 50 years total)
1. **Growth** — settlements produce food, grow population, expand to nearby land, develop ports
2. **Conflict** — raids between settlements, longships extend range, conquered settlements change faction
3. **Trade** — ports trade food/wealth, technology diffuses
4. **Winter** — food loss, settlements can collapse → Ruins
5. **Environment** — ruins reclaimed by nearby settlements or overgrown by forest

## API Endpoints

Base URL: `https://api.ainm.no/astar-island`

Auth: Cookie `access_token=<JWT>` or `Authorization: Bearer <JWT>`

| Method | Path | Description |
|--------|------|-------------|
| GET | /rounds | List all rounds |
| GET | /rounds/{id} | Round details + initial states for all seeds |
| GET | /budget | Remaining queries for active round |
| POST | /simulate | Run one simulation, observe viewport (costs 1 query) |
| POST | /submit | Submit H×W×6 prediction tensor for one seed |
| GET | /my-rounds | Your scores, rank, budget per round |
| GET | /my-predictions/{id} | Your predictions with argmax/confidence |
| GET | /analysis/{id}/{seed} | Post-round ground truth comparison |
| GET | /leaderboard | Rankings |

### POST /simulate — Request
```json
{
  "round_id": "uuid",
  "seed_index": 0,       // 0-4
  "viewport_x": 10,
  "viewport_y": 5,
  "viewport_w": 15,      // 5-15
  "viewport_h": 15       // 5-15
}
```

### POST /simulate — Response
```json
{
  "grid": [[4, 11, 1, ...], ...],   // viewport_h x viewport_w
  "settlements": [
    {"x": 12, "y": 7, "population": 2.8, "food": 0.4, "wealth": 0.7,
     "defense": 0.6, "has_port": true, "alive": true, "owner_id": 3}
  ],
  "viewport": {"x": 10, "y": 5, "w": 15, "h": 15},
  "queries_used": 24, "queries_max": 50
}
```

### POST /submit — Request
```json
{
  "round_id": "uuid",
  "seed_index": 0,
  "prediction": [[[0.85, 0.05, 0.02, 0.03, 0.03, 0.02], ...], ...]
}
```
prediction[y][x][class] — H rows x W cols x 6 probabilities. Must sum to 1.0 per cell.

### Grid Cell Values (internal → class mapping)
| Value | Terrain | Class |
|-------|---------|-------|
| 10 | Ocean | 0 |
| 11 | Plains | 0 |
| 0 | Empty | 0 |
| 1 | Settlement | 1 |
| 2 | Port | 2 |
| 3 | Ruin | 3 |
| 4 | Forest | 4 |
| 5 | Mountain | 5 |

## Scoring

### Formula
```
KL(p || q) = sum(p_i * log(p_i / q_i))     # per cell
entropy(cell) = -sum(p_i * log(p_i))         # ground truth entropy

weighted_kl = sum(entropy * KL) / sum(entropy)  # only dynamic cells
score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
```

### Critical Rules
- **NEVER assign 0.0 probability** — KL goes to infinity. Floor at 0.01, renormalize.
- **Static cells (ocean, mountain) are excluded** from scoring — no entropy.
- **High-entropy cells matter most** — focus observations there.
- **Uniform prediction scores ~1-5** — any observation-based prediction beats it.
- Per-round score = average of 5 seed scores. Missing seed = 0.
- Leaderboard = weighted average across rounds.

## Quickstart Code

```python
import requests
import numpy as np

BASE = "https://api.ainm.no"
TOKEN = "YOUR_JWT_TOKEN"

session = requests.Session()
session.cookies.set("access_token", TOKEN)

# Get active round
rounds = session.get(f"{BASE}/astar-island/rounds").json()
active = next(r for r in rounds if r["status"] == "active")
round_id = active["id"]

# Get initial states
detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()

# Query simulator (costs 1 of 50 queries)
result = session.post(f"{BASE}/astar-island/simulate", json={
    "round_id": round_id,
    "seed_index": 0,
    "viewport_x": 0, "viewport_y": 0,
    "viewport_w": 15, "viewport_h": 15,
}).json()

# Submit prediction (H x W x 6 tensor)
prediction = np.full((40, 40, 6), 1/6)  # uniform baseline
# Floor at 0.01 and renormalize
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)

session.post(f"{BASE}/astar-island/submit", json={
    "round_id": round_id,
    "seed_index": 0,
    "prediction": prediction.tolist(),
})
```

## Current Round
- Round 1: `71451d74-be9f-471f-aacd-a41f3b68a9cd`
- Status: active
- Closes: 2026-03-19 20:42 UTC (21:42 CET)
- Budget: 50 queries, 0 used
- Seeds: 5 (30, 52, 35, 60, 31 settlements respectively)
