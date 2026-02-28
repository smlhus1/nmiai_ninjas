# NMIAI - Grocery Bot Challenge

## What is this?
Competition bot for a grid-based grocery store game. WebSocket bot receives game state JSON, responds with actions within 2 seconds.

## Quick start
```bash
py -m pip install -r requirements.txt
py main.py                              # Connect to game server
py -m pytest tests/ -v                  # Run tests
```

## Architecture: Pipeline + Centralized Coordinator

```
WebSocket (main.py)
    |
    v
Coordinator (bot/coordinator.py) --- owns persistent state
    |
    +-- GameState (bot/models.py) --- immutable, parsed per round
    |
    +-- PathEngine (bot/engine/pathfinding.py) --- A* + BFS cache
    |
    +-- PIBTResolver (bot/engine/pibt.py) --- collision-free movement
    |
    +-- WorldModel (bot/engine/world_model.py) --- enriched queries
    |
    +-- TaskPlanner (bot/strategy/planner.py) --- assigns tasks to bots
    |   +-- RouteBuilder (bot/strategy/route_builder.py) --- multi-item route candidates
    |   +-- Hungarian (bot/strategy/hungarian.py) --- optimal bot-to-route matching
    |
    +-- ActionResolver (bot/strategy/action_resolver.py) --- tasks -> actions via PIBT
```

## File structure
```
nmiai/
  main.py                           # WebSocket client, entry point
  bot/
    models.py                       # Immutable data models (GameState, Bot, Item, etc.)
    coordinator.py                  # Central orchestrator, owns all state
    engine/
      pathfinding.py                # A* + BFS distance cache
      pibt.py                       # PIBT collision resolution
      world_model.py                # Enriched world queries + endgame detection
    strategy/
      task.py                       # Task/BotAssignment/Route/RouteStop definitions
      route_builder.py              # Multi-item route candidate generation
      planner.py                    # Strategic task assignment (active, preview, endgame, route tracking)
      hungarian.py                  # Optimal bot-to-route matching via scipy
      action_resolver.py            # Tactical action generation via PIBT
  tests/
    test_models.py
    test_pathfinding.py
    test_coordinator.py
    test_pibt.py
    test_hungarian.py
    test_routes.py
```

## Key design decisions
- **Immutable GameState**: parsed fresh each round, never mutated
- **Sticky assignments**: bots keep tasks until completed or invalidated
- **No double-booking**: TaskPlanner tracks claimed items globally (PICK_UP + PRE_PICK)
- **PIBT collision resolution**: cooperative movement eliminates deadlocks in narrow corridors
- **Multi-item routes**: bots plan routes (1-3 items) and deliver in one trip instead of per-item
- **Hungarian assignment**: globally optimal bot-to-route matching via scipy
- **Navigation override**: staging bots use override, never mutate Task.target_pos
- **Bot ID priority**: low-ID bots get critical tasks (collision right-of-way)
- **Endgame mode**: last ~40 rounds, abandon incomplete orders, maximize items/round
- **Preview pre-staging**: idle bots pre-pick items for upcoming orders (auto-delivery on transition)
- **BFS distance cache**: cached per destination, shared across bots (many go to same drop-off)

## Task types
- `PICK_UP`: go to item, pick it up (active order)
- `DELIVER`: go to drop-off, deliver inventory
- `PRE_PICK`: pre-pick preview order item (auto-delivery when order transitions)
- `IDLE`: no useful work

## Game rules summary
- 300 rounds max, 2 second response time
- Grid-based, bots navigate + pick up items + deliver to drop-off
- Score: +1 per item delivered, +5 per completed order
- Collisions: actions resolve in bot ID order
- Sequential orders: active + preview, infinite supply
- Auto-delivery: items in inventory matching new active order are auto-delivered on transition

## Environment
- Python 3.13
- `websockets` for server connection
- `scipy` for Hungarian assignment
- `pytest` for testing
- Run with `py` (Windows Python Launcher)

## Config
- `GAME_WS_URL` env var for server address (default: ws://localhost:8765)
- `TEAM_NAME` env var for team name (default: nmiai)
