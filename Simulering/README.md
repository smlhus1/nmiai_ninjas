# Offline Optimization Toolkit for Grocery Bot

> Drop hele `offline/`-mappen i prosjekt-rooten. Null eksterne dependencies (bare stdlib Python).

## Filer

| Fil | Hva | Linjer |
|-----|-----|--------|
| `grid.py` | Grid, BFS, A* — delt mellom alle moduler | ~120 |
| `analyzer.py` | Analyserer game logs. Ordresekvens, items, determinisme-sjekk | ~150 |
| `planner.py` | Brute-force optimal planner med cross-order pipelining | ~300 |
| `replay.py` | Kjører pre-beregnet plan mot live game med fallback | ~200 |
| `simulator.py` | **Lokal game simulator** — reproduserer server-logikk, ~5ms/game | ~350 |
| `strategy.py` | Parameterisert bot-strategi med 13 tunable parametre | ~400 |
| `optimize.py` | Hill climbing / grid search over strategi-parametre | ~200 |
| `run.py` | CLI entry point | ~100 |
| `test.py` | Planner self-test | ~150 |
| `test_e2e.py` | Full integration test: simulator + optimizer | ~200 |

## Quick Start

### Alt 1: Optimaliser fra game log (anbefalt)

```bash
# 1. Kjør recon-game, logg til JSON
py main.py --url "wss://game.ainm.no/ws?token=<JWT>"

# 2. Analyser loggen
py -m offline.run analyze logs/game_log.json

# 3. Sjekk determinisme (kjør to ganger samme dag)
py -m offline.run check-determinism logs/run1.json logs/run2.json

# 4. Optimaliser strategi-parametre (5000 iter ≈ 25 sek)
py -m offline.optimize logs/game_log.json --iterations 5000

# 5. Grid search (systematisk, ~500 kombinasjoner)
py -m offline.optimize logs/game_log.json --grid-search

# 6. Sammenlign strategier
py -m offline.optimize logs/game_log.json --compare
```

### Alt 2: Generer og replay optimal plan

```bash
# Generer optimal pickup-rekkefølge
py -m offline.run plan logs/game_log.json

# For multi-bot:
py -m offline.run plan logs/game_log.json --bots 3
```

## Hvordan det funker

### Simulatoren (~5ms per 300-runde game)

Reproduserer server-logikk nøyaktig:
- Actions i bot-ID-rekkefølge
- Ugyldige actions → stille wait
- pick_up: adjacent (Manhattan 1), inventory < 3
- drop_off: på drop-off-cellen, leverer ALLE matchende active-order items
- Auto-delivery ved ordreovergang
- Items respawner på samme hylle

```python
from offline.simulator import Simulator
sim = Simulator.from_game_log("logs/game_log.json")
result = sim.run(my_strategy)  # {"score": 132, "items_delivered": 52, ...}
```

### Parameterisert strategi (13 tunable knobs)

Hver beslutning styres av et numerisk parameter:

| Parameter | Default | Hva den gjør |
|-----------|---------|-------------|
| `min_items_to_deliver` | 1 | Min items i inventory før levering |
| `force_deliver_matching` | 2 | Lever alltid med ≥N matchende items |
| `deliver_if_close` | 4 | Lever med færre items hvis nær drop-off |
| `cross_order_threshold` | 0 | Start preview-picking når active har ≤N igjen |
| `cross_order_max` | 1 | Maks preview-items å pre-plukke |
| `w_distance` | 1.0 | Vekt: avstand til item |
| `w_completion` | 5.0 | Vekt: bonus for items som fullfører ordre |
| `w_cluster` | 0.5 | Vekt: bonus for items nær andre items |
| `w_dropoff_proximity` | 0.3 | Vekt: bonus for items nær drop-off |
| `endgame_threshold` | 30 | Runder igjen → switch til greedy |
| `endgame_greedy` | true | Lever alt i endgame |
| `zone_penalty` | 3.0 | Straff for items utenfor egen sone |
| `staging_distance` | 2 | Avstand for staging-posisjoner |

### Optimizeren

Hill climbing med restarts + simulated annealing:
1. Start med default-parametre
2. Muter 1-3 parametre tilfeldig
3. Kjør simulator
4. Behold forbedringer
5. Restart fra topp-10 hvis stuck

## Resultater (test-grid)

```
Easy-grid (12x10, 1 bot, 20 ordrer):
  Default params:     99 poeng  (39 items, 12 ordrer)
  Patient deliver:   107 poeng  (42 items, 13 ordrer)
  Optimized (5000i): 132 poeng  (52 items, 16 ordrer)  ← +33%!

Optimized params:
  deliver_if_close:      4 → 1   (fyll inventory, ikke lever tidlig)
  cross_order_threshold: 0 → 3   (alltid pre-plukk preview)

Speed: ~200 games/sec med shared BFS cache
  10k iterasjoner ≈ 50 sek
  100k iterasjoner ≈ 8 min
```

## Integrasjon med eksisterende bot

### Bruk optimaliserte parametre i live-bot

```python
from offline.optimize import load_params
from offline.strategy import ParameterizedStrategy

params = load_params("logs/best_params.json")
# Bruk params.min_items_to_deliver, params.w_completion etc.
# i eksisterende TaskPlanner/RouteBuilder
```

### Bruk replay-executor

```python
from offline.replay import ReplayExecutor, load_plan

plan = load_plan(fingerprint)
if plan:
    replayer = ReplayExecutor(plan, fallback_fn=reactive_bot)
    # I game loop:
    response = replayer.execute(state)
```
