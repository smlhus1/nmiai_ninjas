# NM i AI 2026 — Grocery Bot

Competition bot for [NM i AI](https://ainm.no) pre-qualification. Grid-based grocery store where bots navigate aisles, pick items from shelves, and deliver orders against a 300-round clock.

## Scores

| Map | Grid | Bots | Best Score | Status |
|-----|------|------|------------|--------|
| Easy | 12x10 | 1 | 118 | Tuned |
| Medium | 16x12 | 3 | 81 | In progress |
| Hard | 22x14 | 5 | - | Not started |
| Expert | 28x18 | 10 | - | Not started |

Leaderboard = sum of all 4 maps. Score = `items_delivered + orders_completed * 5`.

## Quick Start

```bash
py -m pip install -r requirements.txt
py -m pytest tests/ Simulering/ -v          # 120 tests, all green
```

**Run against server** (requires token from game dashboard):
```bash
py main.py --url "wss://game.ainm.no/ws?token=<JWT>"
```

**Run offline** (no server needed):
```bash
py -m Simulering.offline.run_offline --scenario easy          # Run bot on built-in scenario
py -m Simulering.offline.run_offline --scenario easy --compare # Compare bot vs simple strategy
```

## Architecture

```
main.py (WebSocket)
  |
  v
Coordinator ---- owns all persistent state
  |
  +-- GameState (models.py)          Immutable, parsed per round
  +-- PathEngine (pathfinding.py)    A* + BFS cache
  +-- PIBTResolver (pibt.py)         Collision-free multi-bot movement
  +-- WorldModel (world_model.py)    Enriched queries per round
  +-- TaskPlanner (planner.py)       Strategic: what should each bot do?
  |     +-- RouteBuilder             Multi-item route candidates (TSP)
  |     +-- Hungarian                Optimal bot-to-route matching (scipy)
  +-- ActionResolver                 Tactical: tasks -> move/pick/drop via PIBT
  +-- GameLogger (recon/)            Records order sequence for offline analysis
```

**Per round:** Parse JSON -> merge shelf walls -> pathfinding -> plan tasks -> resolve actions -> respond. All within 2 seconds.

## Project Structure

```
bot/                        The actual competition bot
  models.py                   Immutable data models (GameState, Bot, Item, Order)
  coordinator.py              Central orchestrator
  config.py                   Tunable parameters (CoordinatorConfig)
  engine/
    pathfinding.py              A* + BFS distance cache
    pibt.py                     Priority Inheritance with Backtracking
    world_model.py              Enriched game state queries
  strategy/
    planner.py                  Task assignment (active, preview, endgame)
    route_builder.py            Multi-item route generation
    hungarian.py                Optimal assignment via scipy
    action_resolver.py          Tasks -> concrete actions
    task.py                     Task/Route/BotAssignment definitions
  recon/
    logger.py                   GameLogger: records orders + shelves
    analyzer.py                 OfflinePlanner: brute-force optimal plan
    replay.py                   ReplayPlanner: execute pre-computed plan

Simulering/offline/         Offline simulator + optimizer
  simulator.py                Full game engine (~1ms/game)
  bot_adapter.py              Wraps live bot for offline testing
  strategy.py                 Simple parameterized strategy (13 knobs)
  optimize.py                 Hill climbing optimizer for strategy params
  optimize_plan.py            Optimizer for CoordinatorConfig params
  test_collision_model.py     Server rule verification (20 tests)
  test_adapter_e2e.py         BotAdapter integration tests
  test_e2e.py                 Simulator E2E + determinism tests

tests/                      Bot unit + integration tests (78 tests)
```

## Key Game Rules

- **300 rounds**, score as much as you can
- **Sequential orders**: complete active before preview activates
- Bots have **3-item inventory** capacity
- `pick_up` from **adjacent** cell (Manhattan distance 1), shelves are not walkable
- `drop_off` delivers ALL items matching active order in one action
- Non-matching items **stay in inventory** (don't waste slots)
- **Auto-delivery**: when active order completes, matching inventory items for the new active order auto-deliver
- Items are **infinite** — respawn at same shelf with new IDs
- Collisions resolve in **bot ID order** (lower ID wins)
- Invalid actions silently become `wait`
- Game is **deterministic per day** — same seed = same game

## What Works Well

- **PIBT collision resolution** — no deadlocks even in narrow corridors
- **Multi-item routes** — bots plan 1-3 item routes, deliver in one trip
- **Hungarian assignment** — globally optimal bot-to-route matching
- **Preview pre-staging** — idle bots pre-pick items for upcoming orders
- **Endgame mode** — switches strategy in final ~40 rounds
- **Offline simulator** — test changes without server, ~1ms per 300-round game

## What Needs Work

### 1. CoordinatorConfig is not wired in (biggest win)

`bot/config.py` has 15+ tunable parameters with `mutate()` for optimization. `optimize_plan.py` has a hill-climbing optimizer. But **Coordinator/Planner/RouteBuilder don't read from config** — all values are hardcoded. Wire this up, run the optimizer, find optimal values per map.

### 2. Recon/replay end-to-end

Individual pieces work (GameLogger, OfflinePlanner, ReplayPlanner). But:
- `finalize_game()` doesn't write plans to disk
- Coordinator doesn't detect/load plans on startup
- The full two-pass flow (recon -> plan -> replay) is not connected

### 3. Hard + Expert maps

5 and 10 bots. PIBT scales, but strategy needs work:
- Zone splitting (divide store into bot zones)
- Dedicated deliverer patterns
- Pipeline: one picks, one delivers, one pre-picks

## Offline Simulator

The simulator reproduces server logic exactly. Use it to test changes without a server.

```python
from Simulering.offline.simulator import Simulator
from Simulering.offline.bot_adapter import BotAdapter

sim = Simulator.from_recon_data(recon_data)  # or make_easy_scenario()

# Run live bot through simulator
adapter = BotAdapter(suppress_logs=True)
result = sim.run(adapter, verbose=False)
print(f"Score: {result['score']}, Items: {result['items_delivered']}")

recon = adapter.finalize(result)  # Get recon data
adapter.reset()                   # Ready for next game
```

**Optimizer** (once config is wired):
```bash
py -m Simulering.offline.optimize_plan --recon logs/<fingerprint>_recon.json --iterations 500
```

## Environment

- Python 3.13, Windows — use `py` not `python`
- Dependencies: `websockets`, `scipy`, `pytest`
- Tests: `py -m pytest tests/ Simulering/ -v`
