# Work Breakdown: ML Planner MVP — NM i AI Nightmare

> Planlagt: 2026-03-16
> Basert på: `tech-eval/ml-planner-mvp-spec-2026-03-16.md`, `tech-eval/ml-planner-architecture-2026-03-16.md`, `tech-eval/ml-planner-risks-2026-03-16.md`

---

## Scope

Erstatt `V2TaskPlanner` med en lært scorer (MLP) + beam search for nightmare-kartet (20 bots).
Mål: sim-score > 393 på nyeste `74001e7f`-recon. Sekundært: inference < 100ms live.

Arkitekturen er drop-in: `MLPlanner` implementerer identiske `plan()` og `maintain()`-signaturer.
PIBT, ActionResolver, BotAdapter, Simulator — alt uendret.

**REVIDERT etter Spike 1 (2026-03-16):**
Originalgate (Oracle assignment > 600) var feil stilt. Standalone greedy oracle scorer 136
vs baseline 354 — ikke fordi assignment er irrelevant, men fordi V2TaskPlanners fulle
pipeline (routes, pre-pick, OrderSolver, stuck handling) er like viktig som assignment.

**Ny framing:** ML planner erstatter HELE V2TaskPlanner-stacken, ikke bare assignment.
Beam search + MLP scorer over (bot, action)-par erstatter OrderSolver + PipelineManager +
all håndskrevet heuristikk. Gate-beslutning tas av bruker basert på spike-resultater.

---

## Estimat-oppsummering

|  | Tradisjonell | AI-assistert |
|---|---|---|
| **Pre-gate (Spike 1)** | 1 dag (8t) | 4 timer |
| **Post-gate (Spike 2+3 + MVP)** | 12 dager (96t) | 5 dager (40t) |
| **Totalt** | ~13 dager | ~5.5 dager |
| **Kritisk sti** | 13 dager | 5.5 dager |
| **Team** | 1 person | 1 person |

---

## Spikes

Spike 1 er GO/NO-GO-gate og MÅ kjøres og evalueres før post-gate arbeid begynner.
Spike 2, 3 og 4 kjøres etter GO-beslutning — Spike 4 kan kjøres parallelt med Spike 2.

---

## Forutsetninger

- PyTorch >= 2.2 med CUDA 11.8+ tilgjengelig (`torch.cuda.is_available()` skal returnere `True`)
- Siste nightmare-recon tilgjengelig i `logs/74001e7f_2026-03-16_*.json`
- `V2TaskPlanner` fungerer som baseline i sim (`py -m Simulering.offline.run_offline --latest medium`)
- `data/` og `models/` mapper opprettes som del av TASK-1-1

---

## ===================================================================
## GATE A: PRE-GATE — Kjøres alltid, uavhengig av GO/NO-GO
## ===================================================================

### Epic 1: Oracle Spike (Gate-beslutning)

**Mål:** Svar på ett spørsmål: kan perfekt assignment gi > 600 i nightmare-sim?
Resultatet bestemmer om hele ML-planner-prosjektet fortsetter.

**Avhenger av:** ingen
**Estimat:** Tradisjonell: 1 dag | AI-assistert: 4 timer

---

#### TASK-1-1: Implement OracleAssigner and measure theoretical max assignment score

**Type:** Spike / Feature
**Estimat:** Tradisjonell: 8 timer | AI-assistert: 4 timer
**Avhenger av:** ingen
**Filer:**
- `oracle/oracle_assigner.py` (ny)
- `oracle/__init__.py` (ny)
- `oracle/run_oracle.py` (ny — CLI-entry)

**Beskrivelse:**
Skriv en `OracleAssigner` som er et drop-in replacement for `V2TaskPlanner` i `BotAdapter`.
Hvert runde har orakel full fremtidskunnskap: den vet hvilke ordrer som kommer og velger alltid
ledige bots med korteste total-vei (BFS pick + BFS dropoff) til items som matcher neste ordre.
Kjør via `BotAdapter` mot nyeste nightmare-recon (`74001e7f_2026-03-16_*.json`).
Sammenlikn score med `V2TaskPlanner`-baseline på eksakt samme recon.

Orakelet er en øvre grense for hva assignment-forbedring kan gi. Det er IKKE en produksjonsplanner —
det er et måleapparat. Implementer `plan()` og `maintain()` med identisk signatur som `V2TaskPlanner`.

**Akseptansekriterier:**
- [x] `OracleAssigner` implementerer `plan(world, assignments)` og `maintain(world, assignments)` med identiske signaturer som `V2TaskPlanner`
- [x] Kjøring mot `74001e7f_2026-03-16_*.json` via BotAdapter fullfører uten unntak
- [x] Oracle-score og V2TaskPlanner-baseline-score logges tydelig til stdout
- [x] Score-differansen (oracle − baseline) er dokumentert (se Gate A Evaluering ovenfor)
- [x] Gate-evaluering er revidert basert på faktiske resultater (se Gate A Evaluering)

**GO/NO-GO-beslutning basert på output:**
```
Oracle > 600  → GO:     Fortsett med Epic 2 (Spike 2+3) og Epic 3+
Oracle 450-600 → GULT:  Bruker bestemmer — skriv ut begge mulige veier
Oracle < 450  → NO-GO:  Stopp. Bottleneck er PIBT/korridor, ikke assignment.
                        Invester i taktisk lag (MAPF, one-way forbedringer) i stedet.
```

**AI-hint:**
OracleAssigner er ren Python med BFS-kall — Claude Code kan generere mesteparten fra spec
(grådige BFS-prioriterte assignments, samme interface som V2TaskPlanner). Kritisk at du
validerer at BFS-kallet bruker den shelf-mergede gridet (ikke rådata), og at claimed_items-settet
oppdateres for å unngå double-booking. Kjør med eksisterende BotAdapter via
`Simulering.offline.bot_adapter.BotAdapter`.

---

## ===================================================================
## GATE A EVALUERING — Revidert etter spike-resultater
## ===================================================================
##
## Resultat fra TASK-1-1 (standalone greedy oracle):
##
##   | Difficulty       | Baseline | Oracle | Gap     |
##   |------------------|----------|--------|---------|
##   | Easy (1 bot)     |       99 |     21 |  -79%   |
##   | Medium (3 bots)  |      172 |     45 |  -74%   |
##   | Nightmare (20)   |      354 |    136 |  -62%   |
##
## Originalgate (Oracle > 600 → GO) er FEIL STILT.
## En standalone greedy oracle mangler V2TaskPlanners sofistikerte logikk
## (routes, pipeline, OrderSolver, stuck handling, dead-weight management).
## Oracle-scoren måler IKKE assignment-ceiling — den måler greedy vs sophisticated.
##
## REVIDERT FRAMING:
## ML planner erstatter HELE V2TaskPlanner-stacken (ikke bare assignment).
## Gate-spørsmålet endres fra "er assignment bottlenecken?" til
## "kan en lært planner erstatte 700 linjer håndskrevet logikk?"
##
## Nøkkelinnsikter fra spike:
## 1. V2TaskPlanners pipeline (pre-pick, routes, OrderSolver) er ESSENSIELT
## 2. IDLE bots MUST park (not target=current_pos) — PIBT yield-on-push
## 3. Pre-pick for 20 bots krever dead-weight management (order-transition clearing)
## 4. Future order knowledge SKADER nightmare (-17%) — overaggressiv pre-pick
##
## Din beslutning:
##   [ ] GO   — ML planner erstatter hele planner-stack. Større scope enn originalt.
##   [ ] NO-GO — V2TaskPlanner er allerede godt optimert. Fokuser på PIBT/MAPF.
##
## ===================================================================

---

## ===================================================================
## POST-GATE — Kjøres KUN etter GO-beslutning
## ===================================================================

### Epic 2: Validation Spikes (Proof-of-concept)

**Mål:** Bevis at (a) en MLP faktisk kan lære noe meningsfull fra recon-data,
og (b) sim-to-live-gap er < 10% for V2TaskPlanner (treningssignalet er gyldig).
Disse to spikene kan kjøres parallelt.

**Avhenger av:** TASK-1-1 med GO-resultat
**Estimat:** Tradisjonell: 2 dager | AI-assistert: 6 timer

---

#### TASK-2-1: Spike — MLP proof-of-concept on historical recon data

**Type:** Spike
**Estimat:** Tradisjonell: 1 dag | AI-assistert: 3 timer
**Avhenger av:** TASK-1-1 (GO)
**Filer:**
- `ml/spike_mlp_poc.py` (ny — throwaway spike-script, ikke produksjonskode)
- `data/` (mappe opprettes)

**Beskrivelse:**
Generer 50 sim-kjøringer med `BotAdapter` mot nyeste nightmare-recon. For hvert runde,
logg (state_features_placeholder, reward_5) per bot som dummy-encoding
(bruk 48 tilfeldige floats per par som placeholder — featuren valideres skikkelig i TASK-3-1).
Tren en minimal `nn.Module` (Linear(48, 64) → ReLU → Linear(64, 1)) i 10 epoker med AdamW.
Del 80/20 train/val. Plot (eller print) train- vs val-loss per epoke.

Målet er IKKE å bygge produksjonskode — det er å bekrefte at treningspipelinen fungerer
end-to-end, og at val-loss konvergerer (ikke divergerer).

**Akseptansekriterier:**
- [ ] 50 sim-kjøringer mot recon fullføres og produserer >= 900K (state, reward_5)-par
- [ ] Val-loss etter 10 epoker er lavere enn initial val-loss (konvergerer, ikke divergerer)
- [ ] Ingen CUDA OOM under trening (verifiser `torch.cuda.is_available()` + batch-størrelse 256)
- [ ] Script kjøres med `py -m ml.spike_mlp_poc --recon <fil>` og printer loss-kurve til stdout

**AI-hint:**
Standard PyTorch supervised learning-boilerplate — Claude Code kan generere hele scriptet fra
arkitekturspeksen. Viktig: bruk HuberLoss (ikke MSE), og verifiser at reward_5-labelen er
normalisert til [0, 1] (del på 10.0 som i spec). Spike-scriptet skal være selvstendig og
kastes etter valideringen er gjort.

---

#### TASK-2-2: Spike — Sim-to-live gap calibration

**Type:** Spike
**Estimat:** Tradisjonell: 4 timer | AI-assistert: 2 timer
**Avhenger av:** TASK-1-1 (GO)
**Filer:**
- `oracle/sim_live_calibration.py` (ny — engangsscript)

**Beskrivelse:**
Kjør `V2TaskPlanner` via `BotAdapter` mot 5 ulike nightmare-recon-filer fra siste 7 dager
(velg fra `logs/74001e7f_*_recon.json`). Sammenlikn sim-score med `final_score`-feltet i
recon-JSONen (live-score). Beregn gjennomsnittlig prosentvis avvik og standardavvik.

Hvis avvik > 20%: treningssignal fra sim er biased — ML-modell trent på sim vil underpreste live.
Da trengs kalibrering (tren på live-logget data i stedet for bare sim-rollouts).

**Akseptansekriterier:**
- [ ] Script kjører mot minst 5 ulike recon-filer og skriver én linje per fil: `[fil] sim=X live=Y avvik=Z%`
- [ ] Gjennomsnittlig avvik og standardavvik skrives til stdout
- [ ] Konklusjon skrives eksplisitt: "Sim-to-live gap: OK (<10%)" eller "ADVARSEL: gap > 20% — vurder live-logging"
- [ ] Ingen krasj hvis `final_score`-feltet mangler i recon (håndter gracefully)

**AI-hint:**
Enkel Python-scripting som leser JSON og kjører eksisterende BotAdapter. Claude Code håndterer
dette raskt. Pass på å bruke eksakt samme recon-fil for sim-kjøringen som `final_score`-kilden —
ikke to ulike recon-filer for samme dag.

---

### Epic 3: FeatureExtractor

**Mål:** `GameState` → 48 float-tensor per (bot, item)-par, med unit-tester mot kjente recon-states.
Dette er den høyeste stille-feil-risikoen i hele pipelinen — en feil her ødelegger trening
uten å krasje.

**Avhenger av:** TASK-2-1 og TASK-2-2 (begge OK)
**Estimat:** Tradisjonell: 3 dager | AI-assistert: 1 dag

---

#### TASK-3-1: Implement FeatureExtractor — bot features (14 floats)

**Type:** Feature
**Estimat:** Tradisjonell: 1 dag | AI-assistert: 3 timer
**Avhenger av:** TASK-2-1, TASK-2-2
**Filer:**
- `ml/__init__.py` (ny)
- `ml/feature_extractor.py` (ny — klasse `FeatureExtractor`, metode `extract_bot_features`)
- `tests/ml/test_feature_extractor.py` (ny)

**Beskrivelse:**
Implementer bot-features-delen av `FeatureExtractor` (14 floats per bot som beskrevet i
arkitekturspec seksjon 5.1): posisjon normalisert, BFS-distanse til drop-off, inventory-status
(3 floats), task one-hot (4 floats), congestion, runde-normalisert, bot_id_norm.

Kritisk: bruk BFS-distanse (fra `PathEngine`), IKKE Manhattan. Nightmare-kartet har smalpassasjer
der Manhattan er fundamentalt feil.

**Akseptansekriterier:**
- [ ] `extract_bot_features(bot, state, path_engine)` returnerer `torch.Tensor` med shape `(14,)` og alle verdier i `[0.0, 1.0]`
- [ ] Unit-test: gitt kjent bot-posisjon (5, 8) og kjent BFS-dist 12, er `dist_to_dropoff == 12 / max_dist` (ikke Manhattan-avstand)
- [ ] Unit-test: bot med tom inventory → `inv_size == 0.0`, `inv_active_match == 0.0`
- [ ] Unit-test: task one-hot er sum 1.0 for alle gyldige task-typer
- [ ] Ingen import av `torch` feiler (verifiser PyTorch-installasjon som del av første kjøring)

**AI-hint:**
Claude Code kan generere encoding direkte fra spec seksjon 5.1. Viktigste valideringspunkt for
deg: at `bfs_dist()`-kallet bruker den shelf-mergede gridet fra `PathEngine`, ikke `state.grid`
direkte. Se `bot/engine/pathfinding.py` for eksisterende BFS-API.

---

#### TASK-3-2: Implement FeatureExtractor — item features (12 floats)

**Type:** Feature
**Estimat:** Tradisjonell: 1 dag | AI-assistert: 3 timer
**Avhenger av:** TASK-3-1
**Filer:**
- `ml/feature_extractor.py` (utvid med `extract_item_features`)
- `tests/ml/test_feature_extractor.py` (utvid)

**Beskrivelse:**
Legg til item-features (12 floats per item som i seksjon 5.2): posisjon, BFS-distanser (bot→item,
item→dropoff), is_active_needed, active_count_needed, is_preview_needed, demand_score,
is_claimed, bots_closer, item_type_norm, item_congestion.

Kritisk edge case: `demand_score` er basert på fremtidige ordre fra recon-data. Hvis recon ikke
er tilgjengelig (første runde), fall tilbake til 0.0. Ikke krasj.

`is_claimed` må sjekke `claimed_items`-settet fra `V2TaskPlanner` — dette settet er ikke direkte
tilgjengelig i `GameState`, men kan passeres inn som parameter.

**Akseptansekriterier:**
- [ ] `extract_item_features(bot, item, state, path_engine, claimed_items, active_order, preview_order, demand)` returnerer `torch.Tensor` med shape `(12,)` og alle verdier i `[0.0, 1.0]`
- [ ] Unit-test: item av type som finnes i aktiv ordre → `is_active_needed == 1.0`; item som IKKE finnes → `0.0`
- [ ] Unit-test: item i `claimed_items`-settet → `is_claimed == 1.0`
- [ ] Unit-test: ingen fremtidige ordrer tilgjengelig → `demand_score == 0.0` (ikke krasj)
- [ ] Unit-test: `dist_bot_to_item` bruker BFS (test med en posisjon der BFS >> Manhattan pga. vegg)

**AI-hint:**
Mye av koden er analog med TASK-3-1. Pass spesielt på `bots_closer`-beregningen — den teller
antall bots med kortere BFS-distanse til item enn gjeldende bot. Bruk eksisterende
BFS-distansecache i `PathEngine` for effektivitet.

---

#### TASK-3-3: Implement FeatureExtractor — global features (22 floats) and full pair encoding

**Type:** Feature
**Estimat:** Tradisjonell: 1 dag | AI-assistert: 3 timer
**Avhenger av:** TASK-3-2
**Filer:**
- `ml/feature_extractor.py` (utvid med `extract_global_features` og `encode_pair`)
- `tests/ml/test_feature_extractor.py` (utvid med integrasjonstest)

**Beskrivelse:**
Implementer global features (22 floats, seksjon 5.3): ordre-progress, pipeline-status (4 floats),
next_order_types (7 floats), collective_next_match, dropoff_congestion, zone-occupancy (3 floats),
score_velocity.

Legg til `encode_pair(bot, item, state, path_engine, context) -> Tensor[48]` som concatenerer
alle tre feature-vektorer. Skriv en integrasjonstest som laster en recon-fil, parser første
game-state, og kaller `encode_pair` for alle (bot, item)-kombinasjoner — verifiser shape `(N_bots * N_items, 48)` og ingen NaN/Inf-verdier.

**Akseptansekriterier:**
- [ ] `extract_global_features(state, path_engine, context)` returnerer `Tensor[22]`, alle verdier i `[0.0, 1.0]`
- [ ] `encode_pair(bot, item, ...)` returnerer `Tensor[48]` = concat av `[14 | 12 | 22]`
- [ ] Integrasjonstest: last recon-fil, kjør `encode_pair` for alle par i runde 1 — ingen NaN, ingen Inf, ingen shape-feil
- [ ] `score_velocity` beregnes fra siste 10 runder — returnerer 0.0 i runde < 10 (ikke krasj)
- [ ] `next_order_types` (7 floats) representerer type-distribusjon for neste ordre — verifiser at summen er <= 1.0

**AI-hint:**
`zone1/2/3_occupied`-features er nightmare-spesifikke. Hardkod nightmare drop-off sone-posisjoner
basert på CLAUDE.md (drop-off på (1,16)). For andre difficulties, returner 0.0. Bruk
`state.map_fingerprint` for å detektere nightmare (`74001e7f`).

---

### Epic 4: ScorerMLP og Treningsinfrastruktur

**Mål:** PyTorch-modell + treningsloop som produserer et gyldig checkpoint på < 5 minutter.

**Avhenger av:** Epic 3 (TASK-3-3)
**Estimat:** Tradisjonell: 3 dager | AI-assistert: 8 timer

---

#### TASK-4-1: Implement ScorerMLP (nn.Module)

**Type:** Feature
**Estimat:** Tradisjonell: 4 timer | AI-assistert: 1 time
**Avhenger av:** TASK-3-3
**Filer:**
- `ml/scorer.py` (ny — klasse `ScorerMLP`)
- `tests/ml/test_scorer.py` (ny)

**Beskrivelse:**
Implementer `ScorerMLP` som beskrevet i arkitekturspec seksjon 4.1:
`Linear(48, 256) → LayerNorm(256) → ReLU → Linear(256, 256) → LayerNorm(256) → ReLU →
Linear(256, 128) → ReLU → Linear(128, 1) → sigmoid`.
Ca. 144K parametere, output i [0, 1].

`ScorerMLP.forward(x: Tensor) -> Tensor` tar batch `(N, 48)` og returnerer `(N, 1)`.

**Akseptansekriterier:**
- [ ] `ScorerMLP()` opprettes uten feil, `sum(p.numel() for p in model.parameters())` er 140K–150K
- [ ] `forward(torch.randn(1200, 48))` returnerer shape `(1200, 1)` med alle verdier i `(0.0, 1.0)`
- [ ] Modellen kan flyttes til GPU: `model.cuda()` uten feil (hvis CUDA er tilgjengelig)
- [ ] Unit-test: `forward` på CPU og GPU gir numerisk ekvivalente resultater (maks diff < 1e-5)

**AI-hint:**
Standard PyTorch `nn.Sequential` eller `nn.Module` — Claude Code genererer dette direkte fra spec.
Viktig: LayerNorm, ikke BatchNorm (BatchNorm oppfører seg dårlig med små batches under inference).

---

#### TASK-4-2: Implement TrainingLogger — log (state, assignment, reward_5) during sim

**Type:** Feature
**Estimat:** Tradisjonell: 1 dag | AI-assistert: 3 timer
**Avhenger av:** TASK-3-3
**Filer:**
- `ml/training_logger.py` (ny — klasse `TrainingLogger`)
- `tests/ml/test_training_logger.py` (ny)

**Beskrivelse:**
`TrainingLogger` intercepter BotAdapter-output og logger (state_features, bot_id, item_id,
task_type, runde, game_id) per beslutning. Reward_5 beregnes i etterbehandling (ikke online)
fordi den avhenger av fremtidige runder.

Implementer:
- `TrainingLogger.on_round(round_num, score, decisions: list[BotDecision])` — logger runde-data
- `TrainingLogger.finalize(final_game_id) -> list[TrainingPoint]` — beregner reward_5 i ettertid
  og returnerer liste med komplette datapunkter
- `TrainingLogger.save(path: Path)` — lagrer som pickle

Edge case: siste 5 runder har < 5 fremtidige runder. Bruk tilgjengelige runder og normaliser
tilsvarende (ikke kast disse datapunktene).

**Akseptansekriterier:**
- [ ] `TrainingLogger.finalize()` produserer korrekt `reward_5` for runde T: `(score[T+5] - score[T]) / 10.0`
- [ ] Unit-test: spill med 10 runder → siste 5 runder har `reward_5` basert på tilgjengelige runder (ikke 0.0 og ikke krasj)
- [ ] `TrainingLogger.save()` produserer pickle som kan lastes med `pickle.load()` og itereres
- [ ] Ingen off-by-one på runde-indeks: verifiser eksplisitt at runde 0 reward_5 bruker score[5], ikke score[4] eller score[6]

**AI-hint:**
Reward_5-beregningen er den vanligste kilden til off-by-one-feil. Be Claude Code skrive
en spesifikk enhetstestfor nøyaktig denne beregningen med hardkodet score-sekvens.
`BotDecision` er et enkelt dataclass — la CC generere det.

---

#### TASK-4-3: Implement collect_training_data CLI and TrainingDataset

**Type:** Feature
**Estimat:** Tradisjonell: 4 timer | AI-assistert: 2 timer
**Avhenger av:** TASK-4-2
**Filer:**
- `ml/collect_training_data.py` (ny — CLI-entry: `py -m ml.collect_training_data`)
- `ml/dataset.py` (ny — klasse `TrainingDataset(torch.utils.data.Dataset)`)

**Beskrivelse:**
CLI kjører `N` sim-spill via `BotAdapter` mot angitt recon-fil, samler `TrainingLogger`-output
fra hvert spill, og lagrer til `data/training_<fingerprint>_<dato>.pkl`.

`TrainingDataset` wrapper pickle-filen for PyTorch DataLoader.

```
py -m ml.collect_training_data \
  --recon logs/74001e7f_2026-03-16_score274_recon.json \
  --n-games 50 \
  --output data/training_74001e7f_2026-03-16.pkl
```

**Akseptansekriterier:**
- [ ] 50 spill fullføres og pickle lagres — ingen krasj ved timeout-runder
- [ ] `len(dataset)` >= 800K for 50 spill (300 runder × 20 bots × ~3 kandidater per bot × 50 spill)
- [ ] `dataset[0]` returnerer `(features: Tensor[48], reward: Tensor[1])` med korrekte shapes
- [ ] CLI printer progress: "Game X/50 done, score=Y, total_samples=Z" per spill
- [ ] Eksisterende `data/*.pkl` overskrives IKKE hvis `--output` ikke er spesifisert (defaultnavn inkluderer dato)

**AI-hint:**
BotAdapter-integrasjon er allerede etablert. Sett `suppress_logs=True` for å unngå støy.
Kall `adapter.reset()` mellom hvert spill. Claude Code kan generere dette raskt fra
eksisterende `run_offline.py` som referanse.

---

#### TASK-4-4: Implement training loop (train.py)

**Type:** Feature
**Estimat:** Tradisjonell: 1 dag | AI-assistert: 2 timer
**Avhenger av:** TASK-4-1, TASK-4-3
**Filer:**
- `ml/train.py` (ny — CLI: `py -m ml.train`)
- `models/` (mappe — opprettes av scriptet)

**Beskrivelse:**
Treningsloop som laster `TrainingDataset`, trener `ScorerMLP` 20 epoker med AdamW og
CosineAnnealingLR, og lagrer checkpoint.

```
py -m ml.train \
  --data data/training_74001e7f_2026-03-16.pkl \
  --output models/ \
  --epochs 20 \
  --batch-size 256 \
  --lr 1e-3
```

Checkpoint: `models/scorer_<fingerprint>_<dato>.pt`.
Bruk 90/10 train/val-split. Print train-loss og val-loss per epoke.

**Akseptansekriterier:**
- [ ] Trening på 900K samples, 20 epoker fullfører på < 10 minutter på RTX A3000
- [ ] Val-loss er lavere ved epoke 20 enn ved epoke 1 (verifiser at modellen faktisk lærer)
- [ ] Checkpoint lagres med `torch.save(model.state_dict(), ...)` og kan lastes med `torch.load()`
- [ ] Ingen CUDA OOM: batch-størrelse 256 med 48-float input er langt innenfor 6GB VRAM
- [ ] Trening avbrytes gracefully ved Ctrl+C og checkpoint lagres fra siste fullførte epoke

**AI-hint:**
Kanonisk PyTorch treningsloop — Claude Code genererer dette rett fra spec.
Bruk `torch.utils.data.random_split` for train/val-split, ikke manuell indeksering.
Husk `model.train()` og `model.eval()` rundt train- og val-loops.

---

### Epic 5: CandidateGenerator og BeamSearch

**Mål:** Global assignment via beam search — ingen double-booking, korrekte constraints,
< 10ms per runde.

**Avhenger av:** Epic 3 (TASK-3-3), Epic 4 (TASK-4-1)
**Estimat:** Tradisjonell: 3 dager | AI-assistert: 1 dag

---

#### TASK-5-1: Implement CandidateGenerator — top-K items per bot

**Type:** Feature
**Estimat:** Tradisjonell: 4 timer | AI-assistert: 2 timer
**Avhenger av:** TASK-3-3
**Filer:**
- `ml/candidate_generator.py` (ny — klasse `CandidateGenerator`)
- `tests/ml/test_candidate_generator.py` (ny)

**Beskrivelse:**
`CandidateGenerator` tar `GameState`, `WorldModel` og `claimed_items`-sett og returnerer
`dict[bot_id, list[str | Literal["DELIVER", "IDLE"]]]` — topp-K kandidater per bot.

Filtreringshierarki:
1. Ikke allerede claime items (sjekk `claimed_items`-settet)
2. BFS-reachable innen rimelig distanse (maks 2 × gjennomsnittlig BFS-dist til nærmeste item)
3. Sorter gjenværende etter BFS-distanse bot → item
4. Returner topp-5 items + alltid legg til DELIVER (hvis inventory ikke er tomt) og IDLE

K=5 er standard. CandidateGenerator håndterer inventory-cap: DELIVER legges til kun hvis
inventory-size > 0.

**Akseptansekriterier:**
- [ ] Returnerer `dict` med entry for ALLE bots i `state.bots` — ingen bot mangler
- [ ] Ingen candidate-liste inneholder et item som er i `claimed_items`-settet
- [ ] "DELIVER" er i kandidat-listen til en bot hvis og bare hvis `len(bot.inventory) > 0`
- [ ] K=5 er default, konfigurerbar som konstruktørparameter
- [ ] Unit-test: bot med tomt inventory → DELIVER er IKKE i kandidatliste

**AI-hint:**
BFS-filtrering bruker eksisterende `PathEngine.bfs_distance()`. Claude Code kan generere
sortering og filtrering raskt. Pass på at BFS-kall bruker shelf-merged grid — se
`bot/engine/pathfinding.py` for API.

---

#### TASK-5-2: Implement BeamSearch over global assignment

**Type:** Feature
**Estimat:** Tradisjonell: 2 dager | AI-assistert: 6 timer
**Avhenger av:** TASK-5-1, TASK-4-1
**Filer:**
- `ml/beam_search.py` (ny — klasse `BeamSearch`)
- `tests/ml/test_beam_search.py` (ny — korrekthetssuite)

**Beskrivelse:**
`BeamSearch` tar `CandidateGenerator`-output og `ScorerMLP`, og produserer global assignment
som `dict[bot_id, str | Literal["DELIVER", "IDLE"]]`.

Algoritme (fra arkitekturspec seksjon 6):
```
beam = [(score=0.0, assignment={}, claimed=set())]
for bot in bots_sorted_by_pibt_priority:
    new_beam = []
    for partial_score, partial_assign, partial_claimed in beam:
        for action in candidates[bot.id]:
            if action is item_id and action in partial_claimed:
                skip  # double-booking
            features = encode_pair(state, bot, action, partial_assign)
            action_score = scorer.forward(features)
            new_beam.append((partial_score + action_score, ...))
    beam = top_20(new_beam)
return beam[0].assignment
```

`claimed`-settet akkumuleres per beam-node (ikke delt mellom noder).

**Akseptansekriterier:**
- [ ] Korrekthetssuite: 100 tilfeldige game-states fra recon → ingen item claimes av to bots
- [ ] Korrekthetssuite: alle assignments er enten gyldig `item_id`, `"DELIVER"` eller `"IDLE"`
- [ ] Timing: < 10ms for 20 bots, topp-5 kandidater, beam width=20 på CPU
- [ ] Beam-bredde=20 er default, konfigurerbar som konstruktørparameter
- [ ] Fallback: hvis beam er tom (ingen gyldige kandidater), assign IDLE til gjeldende bot

**AI-hint:**
Beam search med domenespesifikke constraints er der feil er mest sannsynlige. Be Claude Code
skrive korrekthetstestene FØR implementasjonen (TDD). Kritisk invariant: `claimed`-settet er
PER BEAM-NODE, ikke globalt — feil her gir double-booking som er vanskelig å debugge.

---

### Epic 6: MLPlanner og Coordinator-integrasjon

**Mål:** Drop-in replacement for V2TaskPlanner, med fallback og live-test.

**Avhenger av:** Epic 5 (TASK-5-2), Epic 4 (TASK-4-4)
**Estimat:** Tradisjonell: 2 dager | AI-assistert: 6 timer

---

#### TASK-6-1: Implement MLPlanner — drop-in replacement for V2TaskPlanner

**Type:** Feature
**Estimat:** Tradisjonell: 1 dag | AI-assistert: 3 timer
**Avhenger av:** TASK-5-2
**Filer:**
- `bot/strategy/ml_planner.py` (ny — klasse `MLPlanner`)
- `tests/test_ml_planner.py` (ny)

**Beskrivelse:**
`MLPlanner` implementerer identiske `plan(world, assignments, events)` og
`maintain(state, assignments)` signaturer som `V2TaskPlanner`.

Intern flyt:
1. `FeatureExtractor` enkoder alle (bot, item)-par
2. `CandidateGenerator` genererer topp-K kandidater
3. `ScorerMLP.forward()` batchscorer kandidater
4. `BeamSearch` produserer global assignment
5. Konverter til `BotAssignment`-dict (samme format som `V2TaskPlanner` output)

Fallback: hvis ingen checkpoint er tilgjengelig i `models/`, bruk `V2TaskPlanner` og logg advarsel.

```python
class MLPlanner:
    def plan(self, world, assignments, events) -> None: ...
    def maintain(self, state, assignments) -> None: ...
    def set_future_orders(self, order_sequence: list[dict]) -> None: ...
```

**Akseptansekriterier:**
- [ ] `MLPlanner` med gyldig checkpoint kjøres via BotAdapter uten unntak mot nightmare-recon
- [ ] `MLPlanner` uten checkpoint faller tilbake til V2TaskPlanner og logger `WARNING: no checkpoint found`
- [ ] Sim-score med MLPlanner er >= 200 på nightmare-recon (verifiserer at integrasjonen ikke er fundamentalt ødelagt, ikke at den er god)
- [ ] `plan()` returnerer innen 100ms for alle runder (mål tid i logg)
- [ ] `BotAssignment`-output har korrekte `task.type`-verdier: PICK_UP, PRE_PICK, DELIVER, eller IDLE

**AI-hint:**
Interface-matching er kritisk. Studer V2TaskPlanner-signaturen i
`bot/strategy/v2/planner.py` nøye. Claude Code kan generere selve orkestreringen raskt,
men du må validere at `BotAssignment`-formatet matcher det `ActionResolver` forventer.
Se `bot/strategy/task.py` for `BotAssignment`-dataklassen.

---

#### TASK-6-2: Wire MLPlanner into Coordinator with feature flag

**Type:** Feature
**Estimat:** Tradisjonell: 4 timer | AI-assistert: 1 time
**Avhenger av:** TASK-6-1
**Filer:**
- `bot/coordinator.py` (minimal endring — legg til `use_ml_planner: bool = False` i `CoordinatorConfig`)
- `bot/config.py` (minimal endring — ny config-parameter)
- `tests/test_coordinator.py` (utvid eksisterende test med ML-planner-flag)

**Beskrivelse:**
Legg til `use_ml_planner: bool = False` i `CoordinatorConfig`. Når `True`, instansieres
`MLPlanner` i stedet for `TaskPlanner`. Standard er `False` — ingen endring i nåværende oppførsel.

Endringen i `coordinator.py` er én linje i `__init__`:
```python
if self._config and self._config.use_ml_planner:
    self._planner = MLPlanner(model_dir=Path("models/"))
else:
    self._planner = TaskPlanner()
```

**Akseptansekriterier:**
- [ ] `use_ml_planner=False` (default) gir identisk oppførsel som før — ingen regresjon
- [ ] `use_ml_planner=True` instansierer MLPlanner uten krasj
- [ ] Eksisterende Coordinator-tester passerer uendret
- [ ] `py -m Simulering.offline.run_offline --latest medium` gir samme score som før endringen

**AI-hint:**
Minimal-endring-prinsippet gjelder her. La Claude Code gjøre kun det som er beskrevet — ikke
refaktorere resten av Coordinator. Verifiser eksisterende tester med `py -m pytest tests/ -v`
etter endringen.

---

### Epic 7: Validering og Daglig Retraining CLI

**Mål:** Bekreft at MLPlanner slår V2TaskPlanner-baseline i sim, og bygg daglig workflow.

**Avhenger av:** Epic 6 (TASK-6-2)
**Estimat:** Tradisjonell: 3 dager | AI-assistert: 6 timer

---

#### TASK-7-1: Validate MLPlanner vs V2TaskPlanner baseline in sim

**Type:** Test / Validering
**Estimat:** Tradisjonell: 2 dager | AI-assistert: 1 dag
**Avhenger av:** TASK-6-2, TASK-4-4 (ferdigtrent checkpoint)
**Filer:**
- `ml/validate.py` (ny — CLI: `py -m ml.validate`)

**Beskrivelse:**
Kjør MLPlanner og V2TaskPlanner begge mot samme recon-fil N ganger (N=5 for konsistens).
Sammenlign gjennomsnittlig score og standardavvik.

```
py -m ml.validate \
  --recon logs/74001e7f_2026-03-16_score274_recon.json \
  --checkpoint models/scorer_74001e7f_2026-03-16.pt \
  --n-runs 5
```

Output:
```
V2TaskPlanner:  mean=393.2 std=2.1
MLPlanner:      mean=XXX.X std=Y.Y
Delta:          +XX.X points (+Z.Z%)
Verdict:        GO / NEEDS MORE TRAINING / REGRESSION
```

Suksesskriterium (fra MVP-spec): MLPlanner mean > 393 → MVP-målene er nådd.

**Akseptansekriterier:**
- [ ] Script kjører uten krasj og skriver sammenligning til stdout
- [ ] V2TaskPlanner-baseline replikerer forventet score +/- 5 poeng (verifiserer at testoppsettet er korrekt)
- [ ] Hvis MLPlanner mean < 393: skriv "REGRESSION" og logg de 5 dårligste rundene for diagnostikk
- [ ] Timing rapporteres: gjennomsnittlig `plan()`-tid per runde for MLPlanner

**AI-hint:**
Mesteparten av dette er å kjøre eksisterende kode. Claude Code kan generere script-strukturen.
Den manuelle delen er: analysere resultatet og bestemme neste steg (mer trening? feil i features?).
Bruk `suppress_logs=True` på BotAdapter for å unngå støy.

---

#### TASK-7-2: Implement daily retraining CLI

**Type:** Feature
**Estimat:** Tradisjonell: 1 dag | AI-assistert: 2 timer
**Avhenger av:** TASK-7-1
**Filer:**
- `ml/daily_retrain.py` (ny — CLI: `py -m ml.daily_retrain`)

**Beskrivelse:**
One-command workflow for daglig retraining etter daily reset:
```
py -m ml.daily_retrain \
  --recon logs/74001e7f_<DATO>_*.json \
  --n-games 50 \
  --epochs 20
```

Rekkefølge internt:
1. Kjør `collect_training_data` (50 spill)
2. Kjør `train` (20 epoker)
3. Kjør `validate` (5 runs, sammenlign med siste kjente baseline)
4. Skriv: "New checkpoint ready: models/scorer_<dato>.pt — Score: X (baseline: Y)"

Scriptet er idempotent: hvis pickle allerede finnes for dagens dato, hopp over datafangst.

**Akseptansekriterier:**
- [ ] Hele workflow (50 spill + trening + validering) kjøres med én kommando
- [ ] Idempotent: kjøring to ganger samme dag hopper over datafangst-steget
- [ ] Total kjøretid < 30 minutter (50 offline spill ~10 min + 5 min trening + 5 min validering)
- [ ] Checkpoint-filnavn inkluderer dato: `models/scorer_74001e7f_<YYYY-MM-DD>.pt`
- [ ] Feil i ett steg stopper workflow og printer tydelig feilmelding med steg-navn

**AI-hint:**
Shell-scripting/Python-subprocess er AI sweet spot. Claude Code genererer dette raskt.
Pass på filsti-mønster for recon-glob (`logs/74001e7f_<dato>_*.json`) — bruk `pathlib.Path.glob()`.

---

## Kritisk sti

```
TASK-1-1 (Oracle Spike)
    |
    | [Gate A: GO/NO-GO beslutning]
    v
TASK-2-1 (MLP PoC) ─── parallelt ─── TASK-2-2 (Sim-live gap)
    |
    v
TASK-3-1 (Bot features)
    |
    v
TASK-3-2 (Item features)
    |
    v
TASK-3-3 (Global features + encode_pair)
    |
    +─── TASK-4-1 (ScorerMLP)   TASK-4-2 (TrainingLogger)
    |         |                       |
    |         v                       v
    |    TASK-4-3 (collect_training_data + Dataset)
    |         |
    |         v
    |    TASK-4-4 (train.py)
    |
    v
TASK-5-1 (CandidateGenerator)
    |
    v
TASK-5-2 (BeamSearch)
    |
    v
TASK-6-1 (MLPlanner)
    |
    v
TASK-6-2 (Coordinator-integrasjon)
    |
    v
TASK-7-1 (Validering)
    |
    v
TASK-7-2 (Daily retrain CLI)
```

**Minimum tid (tradisjonell):** ~13 dager (1 person, lineær)
**Minimum tid (AI-assistert):** ~5.5 dager (1 person med Claude Code aktivt)

---

## Anbefalt arbeidsrekkefølge

### Dag 1 (Pre-gate): Oracle Spike

- [ ] TASK-1-1: OracleAssigner + måle theoretical max — 4 timer (AI-assistert)
- **STOPP.** Evaluer resultatet. Ta GO/NO-GO-beslutning.

---

### Dag 2 (Post-gate, kun ved GO): Validation Spikes

- [ ] TASK-2-1: MLP proof-of-concept — 3 timer (parallelt med 2-2)
- [ ] TASK-2-2: Sim-to-live gap kalibrering — 2 timer (parallelt med 2-1)

---

### Dag 3: FeatureExtractor (del 1)

- [ ] TASK-3-1: Bot features (14 floats) — 3 timer
- [ ] TASK-3-2: Item features (12 floats) — 3 timer

---

### Dag 4: FeatureExtractor (del 2) + ScorerMLP

- [ ] TASK-3-3: Global features + encode_pair — 3 timer
- [ ] TASK-4-1: ScorerMLP — 1 time
- [ ] TASK-4-2: TrainingLogger — 3 timer (kan delvis parallelliseres med 3-3)

---

### Dag 5: Treningspipeline + BeamSearch

- [ ] TASK-4-3: collect_training_data CLI + Dataset — 2 timer
- [ ] TASK-4-4: train.py — 2 timer
- [ ] TASK-5-1: CandidateGenerator — 2 timer

---

### Dag 6: BeamSearch + MLPlanner

- [ ] TASK-5-2: BeamSearch + korrekthetssuite — 6 timer (største enkelttask)

---

### Dag 7 (siste): Integrasjon + Validering + CLI

- [ ] TASK-6-1: MLPlanner — 3 timer
- [ ] TASK-6-2: Coordinator-integrasjon — 1 time
- [ ] TASK-7-1: Validering mot baseline — manuell vurdering
- [ ] TASK-7-2: Daily retrain CLI — 2 timer

---

## Parallelliseringsmuligheter

Etter GO-beslutning kan disse taskene kjøres parallelt av to utviklere eller to AI-agenter:

| Parallell gruppe | Tasks | Forutsetning |
|---|---|---|
| Gruppe A | TASK-2-1 + TASK-2-2 | TASK-1-1 GO |
| Gruppe B | TASK-4-1 (ScorerMLP) + TASK-4-2 (TrainingLogger) | TASK-3-3 ferdig |
| Gruppe C | TASK-5-1 (CandidateGenerator) starter | TASK-3-3 ferdig (uavhengig av trening) |

`CandidateGenerator` (TASK-5-1) kan bygges og testes uavhengig av treningspipelinen
(TASK-4-1 til 4-4) — begge avhenger bare av `FeatureExtractor`.

---

## Fase 2 (PPO) — Aktiveres KUN hvis

MLPlanner scorer stabilt > 500 i sim **og** live, og platåer over 2–3 dager.

Ikke start Fase 2 som en "kanskje det hjelper"-investering. Reward shaping for 20 bots
med sparse signal er dokumentert vanskelig (se `memory/nightmare_experiments.md`).

Estimat Fase 2: tradisjonell 7 dager, AI-assistert 4 dager.
Involverer: PettingZoo/TorchRL wrapper, MAPPO centralized critic, PPO fine-tuning.
