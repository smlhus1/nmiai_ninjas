# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is this?
Competition bot for NM i AI's Grocery Bot challenge. WebSocket bot receives game state JSON, responds with actions within 2 seconds. Grid-based grocery store where bots navigate, pick up items from shelves, and deliver to a drop-off zone.

## Commands
```bash
py -m pip install -r requirements.txt   # Install dependencies
py main.py --url "wss://..."            # Connect to game server (NEVER run without explicit user request)
py -m pytest tests/ -v                  # Run all tests
py -m pytest tests/test_pibt.py -v      # Run single test file
py -m pytest tests/test_pibt.py -k "test_name" -v  # Run single test
```

## Architecture: Pipeline + Centralized Coordinator

```
WebSocket (main.py) — thin layer, drains buffered messages to prevent desync
    |
    v
Coordinator (bot/coordinator.py) — owns persistent state, orchestrates pipeline
    |
    +-- GameState (bot/models.py) — immutable, parsed per round
    |
    +-- PathEngine (bot/engine/pathfinding.py) — A* + BFS distance cache
    |
    +-- PIBTResolver (bot/engine/pibt.py) — collision-free movement
    |
    +-- WorldModel (bot/engine/world_model.py) — enriched queries (per-round, not persisted)
    |
    +-- TaskPlanner (bot/strategy/planner.py) — assigns tasks to bots (strategic brain)
    |   +-- RouteBuilder (bot/strategy/route_builder.py) — multi-item route candidates
    |   +-- Hungarian (bot/strategy/hungarian.py) — optimal bot-to-route matching via scipy
    |
    +-- ActionResolver (bot/strategy/action_resolver.py) — tasks -> actions via PIBT (tactical layer)
    |
    +-- Recon/Replay (bot/recon/) — two-pass optimization
        +-- GameLogger (logger.py) — records order sequence + shelf map
        +-- OfflinePlanner (analyzer.py) — brute-force optimal plan from recon data
        +-- ReplayPlanner (replay.py) — executes pre-computed plan with reactive fallback
```

### Data flow per round
1. Parse raw JSON -> immutable `GameState`
2. Merge shelf positions into grid as walls (cached once on first round)
3. Initialize `PathEngine` with enhanced grid
4. Build `WorldModel` (enriched view, created fresh each round)
5. `TaskPlanner.plan()` assigns/updates tasks (or `ReplayPlanner.plan()` in replay mode)
6. `_schedule_dropoff()` limits concurrent deliverers to prevent gridlock
7. `ActionResolver.resolve()` converts tasks to concrete actions via PIBT
8. Return JSON response

### Strategic vs Tactical separation
- **TaskPlanner** (strategy) decides WHAT each bot should do: which item to pick, when to deliver, endgame behavior
- **ActionResolver** (tactical) decides HOW: pathfinding, collision avoidance via PIBT, converting targets to move/pick/drop actions
- These must stay separate — mixing strategy into action resolution causes subtle bugs

## Game rules — critical gotchas
- `drop_off` only delivers items matching the **ACTIVE** order. Non-matching items STAY in inventory.
- Only 1 `drop_off` action per round, delivers ALL matching items at once.
- Invalid actions silently become `wait` — no error feedback from server.
- Items are INFINITE — they respawn at the same shelf positions with new IDs. Item count stays constant.
- Shelf positions are NOT in the server's `walls` list but ARE non-walkable. Coordinator adds them.
- Collisions resolve in bot ID order (low ID wins).
- Auto-delivery: when active order completes, preview becomes active and matching inventory items auto-deliver.
- 300 rounds max, 2s response time, 10s cooldown between games, 120s wall-clock limit per game.
- Score: +1 per item delivered, +5 per completed order.

## Key design decisions and invariants
- **Immutable GameState**: parsed fresh each round, NEVER mutated after creation
- **Sticky assignments**: bots keep tasks until completed or invalidated — prevents flip-flopping
- **No double-booking**: `claimed_items` set in TaskPlanner tracks ALL claimed items (PICK_UP + PRE_PICK + route items + blacklisted)
- **Navigation override**: staging bots use `BotAssignment.navigation_override`, NEVER mutate `Task.target_pos`
- **Bot ID priority**: low-ID bots get higher PIBT priority (collision right-of-way)
- **BFS distance cache**: cached per destination, shared across bots. Grid must stay identical between rounds for cache validity — that's why shelf positions are merged once and never changed.
- **Hungarian handles ONLY active order items**. Preview pre-picking is handled separately by `_assign_preview_tasks()`.
- **Route advancement in `_advance_routes()`**: detects pickup via inventory change (Counter comparison), NOT item disappearance alone (items respawn infinitely)
- **`_prev_inventory` must be saved every round** — including early returns in endgame. Missing this breaks stuck detection.

## Two-pass recon/replay system
Game is deterministic per day (same seed). First run logs orders+shelves (recon mode), offline planner brute-forces optimal pickup sequence, second run executes plan (replay mode) with reactive fallback on divergence. Plans stored in `logs/` keyed by fingerprint + date.

## Task types
- `PICK_UP`: go to item, pick it up (active order)
- `DELIVER`: go to drop-off, deliver inventory
- `PRE_PICK`: pre-pick preview order item (auto-delivery on order transition)
- `IDLE`: no useful work

## Offline Simulator (Simulering/)

Local game engine for testing strategies without a server connection.

```
Simulering/offline/
    simulator.py       — Full game engine (movement, pickup, dropoff, auto-delivery, collisions)
    strategy.py        — ParameterizedStrategy (13 tunable params, simple BFS-based)
    bot_adapter.py     — BotAdapter: wraps live Coordinator for use inside Simulator
    optimize.py        — Hill climbing + grid search for StrategyParams tuning
    planner.py         — Brute-force optimal pickup sequence planning
    run_offline.py     — CLI: run live bot offline (--recon, --scenario easy/medium, --compare)
    test_collision_model.py  — 34 tests verifying simulator matches server rules
    test_adapter_e2e.py      — BotAdapter integration tests (adapter + recon round-trip)
    test_e2e.py, test.py     — Existing simulator + strategy tests
```

### Workflows
```bash
# Run live bot on built-in easy scenario
py -m Simulering.offline.run_offline --scenario easy

# Compare live bot vs simple strategy
py -m Simulering.offline.run_offline --scenario easy --compare

# Replay a captured game (from a live recon run)
py -m Simulering.offline.run_offline --recon logs/abc12345_2026-03-01_recon.json

# Run collision model verification
py -m pytest Simulering/offline/test_collision_model.py -v
```

### Key factory methods on Simulator
- `Simulator.from_recon_data(recon_dict)` — build from live bot's recon JSON (GameLogger output)
- `Simulator.from_recon_file("path/to/recon.json")` — convenience: load from disk
- `Simulator.from_game_log("path/to/log.json")` — legacy: build from full round-by-round log
- `Simulator.from_analysis(analysis, grid)` — build from analyzer output

### BotAdapter
Wraps the live `Coordinator` as a Simulator-compatible callable. Same code path as a live game.
```python
from Simulering.offline.bot_adapter import BotAdapter
adapter = BotAdapter(suppress_logs=True)
result = sim.run(adapter)
recon = adapter.finalize(result)  # returns recon dict
adapter.reset()                   # ready for next game
```

## Environment
- Python 3.13, run with `py` (Windows Python Launcher)
- `websockets`, `scipy` (Hungarian), `numpy`, `pytest`
- `GAME_WS_URL` env var or `--url` flag for server address

## MCP documentation
Challenge docs available via the `nmiai` MCP server. Use `search_docs` tool or read resources like `challenge://scoring` for game mechanics details.
