## Nightmare Offline Optimizer

### Mål
Score 1300+ på nightmare. Ordresekvensen er KJENT. Deterministisk optimaliseringsproblem.

### Tilnærming
Bruk validert sim (`Simulering/offline/simulator.py` + BotAdapter) som sannhetskilde.
Kjør hundrevis/tusenvis av varianter. Loop: optimaliser → live for flere ordrer → optimaliser.

### Validert sim
- `py -m Simulering.offline.run_offline --recon <fil>` — BotAdapter i simulator
- 34 collision model tester, matcher server-oppførsel
- BotAdapter sim ≈ live score (297-319 vs 260-314)

### Gjenbrukbare filer i solver/
- `grid.py` — Grid, GameMap, pickup_positions
- `pathfinding.py` — BFS DistanceCache
- `orders.py` — OrderQueue, ShelfIndex
- `trips.py` — TripPlanner, TSP for pickup-rekkefølge

### Filer som er FEIL/unyttige
- `sim.py` — fake PIBT (0 kollisjoner), ga 350 som ikke matcher live
- `live_coordinator.py` — mangler PIBT, scorer 3 live
- `scheduler.py`, `assignment.py` — statisk uten collision-awareness

### Nåværende status
- 34 ordrer tilgjengelig (180 items, maks score 350)
- BotAdapter sim: 297-319
- Live: 260-314
- Trenger 100+ ordrer for å teste 1300+ potential
