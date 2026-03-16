# 5 Whys: Nightmare velocity ceiling — 0.73/round vs 2.5 target
**Dato:** 2026-03-16
**Prosjekt:** NM i AI Grocery Bot

## Symptom
Nightmare scorer 131 @ R180 (0.73/round). LNS viser theoretical 264 @ R180 (1.47/round).
Mål er 2.5/round (1250 score, ~114 ordrer i 500 runder). Nåværende: 35 ordrer/500r.
For 114 ordrer trenger vi 4.4 runder per ordre. Nåværende snitt: 14.3r/ordre.

## Analyse

### Nivå 1: Hvorfor er velocity begrenset til 0.73/round?
**Funn:** 24% av alle bot-runder (2434/10000) er IDLE. Bots gjør ingenting.
Ytterligere 23% er WAIT (PIBT-congestion). Kun ~53% av bot-kapasiteten brukes produktivt.
**Bevis:** Direkte telling av task-typer per runde over 500 runder.
**Confidence:** HØY

### Nivå 2: Hvorfor er 24% av bot-runder IDLE?
**Funn:** 100% av IDLE bots har FULL INVENTORY (3/3 items). De er IDLE fordi
items ikke matcher active eller preview ordre. Koden sjekker bare N (active) og N+1 (preview).
Items er "dead weight" fra kodens perspektiv.
**Bevis:** `planner.py:1485-1506` — `if len(bot.inventory) >= 3: ... else: IDLE`
**Confidence:** HØY

### Nivå 3: Hvorfor er "dead weight" items faktisk verdifulle?
**Funn:** 99% av IDLE bot-runder har items som matcher fremtidige ordrer (N+2..N+8).
6610 items holdes i "dead weight" som faktisk er pipeline-ready for kommende ordrer.
49 items var allerede i IDLE inventories da ordren de matcher startet — men ble IKKE levert.
**Bevis:** Kryssreferanse mellom IDLE inventory og fremtidige ordrer fra recon.
Ordre 10: 4/5 items allerede i IDLE bots. Ordre 20: 5/6. Ordre 30: 5/6.
**Confidence:** HØY

### Nivå 4: Hvorfor leverer ikke bots items de allerede har for fremtidige ordrer?
**Funn:** `_nightmare_queue_strategy` linje 1485-1506 har en 3-veis if:
```python
if active_mc > 0:     → DELIVER (target drop-off)
elif preview_mc > 0:  → STAGE near drop-off
else:                 → IDLE at queue position  ← ROTÅRSAKEN
```
Koden sjekker bare active (N) og preview (N+1). Bots med items for N+2, N+3, ..., N+8
klassifiseres som "no match" og sendes til IDLE-kø.

**Konsekvens:** Bots med 3/3 items som matcher ordre 3-6 ordrer frem:
1. Sitter IDLE i korridorer (y=15: 926 bot-runder!)
2. BLOKKERER andre bots som prøver å levere/plukke
3. Kan ikke plukke nye items (full inventory)
4. Når deres ordre endelig aktiveres, må de navigere fra IDLE-posisjon til drop-off (5-15r ekstra)

**Bevis:** `planner.py:1503-1505`, IDLE position data (y=15 dominerer)
**Confidence:** HØY

### Nivå 5: ROTÅRSAK — Hva mangler arkitekturelt?
**Funn:** Systemet mangler en **3+ stage pipeline**.

**Nåværende arkitektur (2-stage):**
```
Active (N):  pick → deliver
Preview (N+1): pre-pick → [wait as IDLE]
Future (N+2+): IGNORED
```

**Nødvendig arkitektur (3-4 stage pipeline):**
```
Stage DELIVER:  Bots med active-match → deliver          (6 bots, ~3r)
Stage STAGE:    Bots med N+1..N+3 match → near drop-off  (7 bots, ~2r transit)
Stage PICK:     Empty bots → pick N+3..N+5 items         (7 bots, ~12r cycle)
```

Throughput med pipeline: 1 ordre per 4-5 runder (vs 14.3r nå).
4.4r/ordre → 114 ordrer/500r → 1250 score → 2.5/round.

**Bevis:**
- 2434 IDLE bot-runder med full inventory (24% kapasitet bortkastet)
- 49 items allerede i IDLE inventory ved ordrestart (490r potensiale)
- IDLE bots klynger y=15 (926 bot-runder) og blokkerer drop-off approach
- Koden har INGEN mekanisme for N+2+ staging
**Confidence:** HØY

## Rotårsak
**Beskrivelse:** `_nightmare_queue_strategy` har en 2-stage pipeline (active + preview)
men trenger en 3-4 stage pipeline. Bots med full inventory som matcher N+2..N+8 ordrer
klassifiseres som IDLE og parkeres i korridorer, der de blokkerer trafikk OG kaster bort
sine pre-picked items. Ordresekvensen er KJENT fra recon men brukes ikke for staging.

**Kategori:** `wrong-assumption` — antatt at bare N og N+1 er relevante for bot-assignment

**Confidence:** HØY — 99% av IDLE inventory matcher fremtidige ordrer, 49 items
per spill er "gratis" leveranser som aldri skjer

## Anbefalt fix

### Umiddelbar fix (planner.py:1503-1505):
Erstatt IDLE-assignment for full-inventory bots med:
```python
# Check future orders N+2..N+8
future_match = check_future_orders(bot.inventory, order_sequence, current_idx + 2)
if future_match:
    # Stage near drop-off — ready for instant delivery when order activates
    nearest_do = world.nearest_drop_off(bot.position, bot.id)
    a.task = Task(task_type=TaskType.PRE_PICK, target_pos=nearest_do)
else:
    # Truly dead weight — dump inventory and pick new items
    a.task = Task(task_type=TaskType.IDLE, target_pos=queue_pos)
```

### Full pipeline fix:
1. Bots med N+2..N+4 match i inventory → stage near drop-off (PRE_PICK target)
2. Bots som er staged og nær drop-off → vente stille (IKKE blokkere korridorer)
3. Empty bots → aggressivt plukke N+3..N+6 items (utvidet pre-pick)
4. Staging-posisjoner bør være UTENFOR korridorer (ikke y=15!)
5. Bruk recon ordresekvens til å SEKVENSERE staging (N+2 bots nærmest drop-off, N+4 bots lenger unna)

### Kapasitetsberegning med pipeline:
- 20 bots / 3 stages = ~7 bots per stage
- Stage 1 (deliver): 6 items / 7 bots = 1 runde + 2r transit = 3r
- Stage 2 (stage): 7 bots allerede nær drop-off, venter
- Stage 3 (pick): 7 bots picking N+3 items, ~12r cycle
- Throughput: 1 ordre per max(3r, 12r/2) ≈ 6r
- 500r / 6r = 83 ordrer × 11 score = 913 score (1.83/round)
- Med 4 stages og tighter packing: ~4.5r/ordre → 111 ordrer → 1221 (2.44/round)

## Preventivt tiltak
- [ ] Legg til test: "IDLE bots med full inventory som matcher N+2..N+8 skal ALDRI forekomme"
- [ ] Instrumenter pipeline-stage fordeling per runde (deliver/stage/pick/idle) for kontinuerlig monitoring
- [ ] CLAUDE.md: dokumenter 3-stage pipeline som arkitekturmål

## Nøkkeldata

### IDLE bot inventory match mot fremtidige ordrer
| Metric | Verdi |
|--------|-------|
| IDLE bot-runder totalt | 2434 (24% av 10000) |
| IDLE med inventory | 2434 (100%) |
| IDLE med future match | 2426 (99%) |
| Items i IDLE som matcher N+2..N+8 | 6610 |
| Items "gratis" ved ordrestart (ikke levert) | 49 per spill |
| Estimert bortkastet potensiale | ~490 runder |

### IDLE bot posisjon (y-fordeling)
| y-rad | Bot-runder | Note |
|-------|-----------|------|
| y=15 | 926 | DROP-OFF APPROACH — blokkerer leveranser |
| y=16 | 336 | DROP-OFF RAD — blokkerer drop-off |
| y=14 | 193 | Nær drop-off |
| y=9 | 268 | Cross-corridor |

### Per-ordre "gratis" items (allerede i IDLE inventory)
| Ordre | Items i IDLE / Totalt | Note |
|-------|----------------------|------|
| 10 | 4/5 | Nesten komplett — 1 item fra instant |
| 20 | 5/6 | Nesten komplett |
| 30 | 5/6 | Nesten komplett |
| 8 | 4/7 | Over halvparten |
| 17 | 4/7 | Over halvparten |
| 6 | 3/4 | 75% |
