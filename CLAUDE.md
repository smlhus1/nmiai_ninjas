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
- Auto-delivery: when active order completes, preview becomes active. ONLY the delivering bot's matching inventory auto-delivers. Other bots must manually drop_off.
- 300 rounds max, 2s response time, 10s cooldown between games, 120s wall-clock limit per game.
- Score: +1 per item delivered, +5 per completed order.

## Difficulty levels and map details

| Difficulty | Grid | Bots | Fingerprint | Drop-off | Spawn | Best Score |
|-----------|------|------|-------------|----------|-------|------------|
| Easy | 12x10 | 1 | `31642503` | — | — | 124 |
| Medium | 16x12 | 3 | `6fb8097b` | — | — | 151 |
| Hard | 22x14 | 5 | `8d88a034` | (1,12) | (20,12) | 139 |
| Expert | 22x14 | 10 | `515edd5d` | — | — | 118 |
| Nightmare | 30x18 | 20 | `74001e7f` | (1,16) | (28,16) | 357 |

Leaderboard = SUM of best score across ALL difficulties. Items/orders change daily (midnight UTC), grid structure is fixed per difficulty.

### Config presets
- `CoordinatorConfig.for_difficulty(n_bots)` auto-selects: 1 bot → easy, ≤3 → medium, ≥10 → nightmare, else hard
- All presets have `replay_enabled=False` — replay doesn't help multi-bot and regresses single-bot (124 vs 93)
- `TaskPlanner.maintain()` accepts `skip_route_abort` and `skip_time_check` kwargs for replay mode

### Per-difficulty bottlenecks
- **Easy** (1 bot): route optimization, delivery timing
- **Medium** (3 bots): drop-off congestion, PIBT collisions, inventory deadlock, bot coordination
- **Hard** (5 bots): same as medium + longer distances, spawn stacking (all 5 at same pos)
- **Nightmare** (20 bots): x=1 DOWN sole approach to drop-off → massive bottleneck, spawn stacking (20 bots at (28,16)), preview items create dead weight

### Nightmare map corridors
- Cross-corridors: y=1, y=9, y=15, y=16
- Vertical aisles: x=1,4,8,12,16,20,24,27,28
- Motorway: y=15/16 RIGHT, y=1/9 LEFT, vertical alternate DOWN/UP (counter-clockwise flow)

### CRITICAL: Conveyor belt is HARMFUL for 5+ bots
Auto-delivery only fires for the DELIVERING bot. Other bots' preview items become dead weight they can't deliver. Merging active+preview orders (conveyor belt strategy) causes bots to fill inventory with preview items → can't deliver active items. This was masked by a simulator bug (fixed 2026-03-08) where auto-delivery fired for ALL bots.

- **NEVER change shared planner/strategy code without testing BOTH `--latest easy` AND `--latest medium`**
- Multi-bot replay was attempted (VRP, distributed batches, type budgets) but all approaches scored lower than reactive due to drop-off congestion, pre-pick dead weight, and coordination overhead.

## PIBT urgency tiers
Collision priority in PIBTResolver (lower number = higher priority):
- `ESCAPE` (-1): bot at drop-off but NOT doing DROP_OFF — must get out of the way
- `DELIVER` (0): heading to drop-off with items
- `PICK_UP` (1): heading to pick up active order item
- `PRE_PICK` (2): pre-picking preview order item
- `IDLE` (3): no task
- Pushed bots (depth > 0) yield unless they are AT their target position

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
    recon_utils.py     — Find latest recon files by difficulty
    run_offline.py     — CLI: run live bot offline (--latest, --recon, --scenario, --compare)
    test_collision_model.py  — 34 tests verifying simulator matches server rules
    test_adapter_e2e.py      — BotAdapter integration tests (adapter + recon round-trip)
    test_e2e.py, test.py     — Existing simulator + strategy tests
```

### Workflows
```bash
# RECOMMENDED: Auto-find latest recon for difficulty (realistic testing)
py -m Simulering.offline.run_offline --latest easy
py -m Simulering.offline.run_offline --latest medium --compare

# Replay a specific captured game
py -m Simulering.offline.run_offline --recon logs/6fb8097b_2026-03-04_recon.json

# Built-in scenario (smoke test ONLY — NOT representative of live games)
py -m Simulering.offline.run_offline --scenario easy

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

## Nightmare Experiment Log
**File: `memory/nightmare_experiments.md`** (in Claude auto-memory directory)
- **READ BEFORE every nightmare optimization attempt** — don't repeat failed experiments
- **UPDATE AFTER every experiment** with result (score, what changed, notes)
- Contains all tested parameter combinations, scores, and key learnings

## CRITICAL OPTIMIZATION RULES — READ BEFORE CHANGING ANYTHING

1. **Optimaliser for RECON, ikke random sim-ordrer.** Simulatoren genererer nye ordrer med `Random(42)` når reconen er utbrukt. Disse random-ordrene finnes IKKE i live-spill. Optimizerresultater basert på genererte ordrer er verdiløse — de overfitter til sim.
2. **Replay er DEAKTIVERT.** `replay_enabled=False` er default. Replay divergerer umiddelbart på medium (daglig endrede ordrer) og kaster bort 5+ runder. Optimizer skal ALDRI sette `replay_enabled=True`.
3. **Sim-live gap betyr at sim er feil, ikke at koden er feil.** Når sim gir 118 men live gir 52, er problemet at sim har annerledes ordresekvens og kollisjonsoppløsning — ikke at koden trenger mer tuning.
4. **Ikke legg til aggressive heuristikker uten å måle.** Blacklisting, avstandsstraffer og korte stuck-terskler kan gjøre ting VERRE. Mål alltid før/etter med sim mot SAMME recon.
5. **Gjenbruk recon fra live-runs.** Ny recon = ny ordresekvens. Optimaliser mot ferskeste recon, og test live med den configen.

### Hva som HAR fungert
- pickup_positions grid fix: +15 (medium 84→99)
- Single-bot preview keep fix: +7 (easy 107→114)
- Step 1.8 route preservation: +10 (medium 101→111)
- PIBT navigation_override priority: +7 (medium 111→118)
- Blacklist stuck items i coordinator (hindrer infinite re-assign loop)
- Auto-delivery bug fix in simulator: sim now matches live scores (2026-03-08)
- PIBT yield-on-push: pushed bots yield unless AT target — fixes drop-off deadlock
- PIBT grid bug fix: ActionResolver now passes shelf-merged grid, not raw state.grid
- One-way aisles (vertical + horizontal): essential for 5+ bots

### Hva som IKKE fungerte (IKKE gjenta)
- Type-claim deduplication (3 varianter): ALLE crasher til 19
- Aggressiv PRE_PICK blacklist + 3-runde terskel + drop-off penalty: 63→29 regresjon
- Preview pre-picking threshold tuning: ZERO effect
- Distance-based batching: Pipeline stall
- IDLE bots at drop-off: Gridlock (118→12)
- Disable _schedule_dropoff: Catastrophic crash til 64
- Hungarian param sweeps: ZERO effect
- Multi-bot replay (VRP, distributed batches): ALLE lavere enn reaktiv
- 10000 optimizer-iterasjoner med genererte sim-ordrer: ZERO fremgang
- Preview cap for 5+ bots: regression
- Evict to staging: catastrophic gridlock
- PRE_PICK urgency 2→1: catastrophic (all same priority)
- y=15 LEFT approach highway (nightmare): 75→75 (loop problem)
- Token delivery system (wait at position): 92→49
- Conveyor belt (merge active+preview) for 5+ bots: dead weight (based on sim bug)

## Time-Space A* Planner (AKTIV UTVIKLING)

### Status
- `solver/planner_v2.py` — planlegger 39 ordrer i 291 runder (est 401+)
- `solver/time_space.py` — A* med (x, y, timestep) state + reservation table
- `solver/scripted_strategy.py` — spiller pre-computed plan i sim
- **BUG**: sim score 1 — collision model mismatch
- **FIX**: spawn stacking, ID-priority, pickup/dropoff timing

### Validert workflow (FUNGERER)
```bash
py _capture_reactive.py <recon> <output> [config]  # capture BotAdapter plan
py main.py --mapf <plan> --url "wss://..."          # replay live (381 score)
```

### Time-Space workflow (UNDER UTVIKLING)
```bash
py -m solver.planner_v2 --recon <recon>             # plan + validate i sim
# Når score > 381: capture som MAPF plan og replay live
```

## C++ MAPF Planner (`cpp_solver/mapf.cpp`)

### Status
- Round-by-round sequential planner with reactive item assignment
- Zone partitioning for nightmare (3 zones, 7/7/6 bots)
- LNS infrastructure for trip optimization (multi-threaded)
- **Works**: easy (55), medium (27), nightmare (20) — all sim-verified
- **Broken**: hard (2) — 5 bots at same spawn with single zone congestion

### Build & Run
```bash
cd cpp_solver && build_mapf.bat
mapf.exe --recon <file> --greedy                    # fast: no LNS
mapf.exe --recon <file> --iterations 1000           # LNS search
```

### Known Issues & Learnings
- No PIBT: uses sequential BFS step — works for 1-3 bots and nightmare (with zones/stagger), fails for hard (5 bots, 1 zone)
- One-way rules DISABLED — cause oscillation loops with sequential planner
- Spawn stacking bug fixed: sim does NOT allow stacking, planner must match
- Claimed positions (not shelves) prevent multi-bot target conflicts; claims released on pickup/stuck
- Stuck detection: 25 rounds without distance progress → reassign
- TODO: Port PIBTResolver from solver.cpp for better multi-bot collision resolution

## MCP documentation
Challenge docs available via the `nmiai` MCP server. Use `search_docs` tool or read resources like `challenge://scoring` for game mechanics details.
