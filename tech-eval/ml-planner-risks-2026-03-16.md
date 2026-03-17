# Risiko- og kompleksitetsvurdering: GPU-Based ML Planner

> Vurdert: 2026-03-16 | Basert på: `tech-eval/ml-planner-architecture-2026-03-16.md`, `research/ml-planner-nightmare-2026-03-16.md`, `CLAUDE.md`

---

## Oppsummering

Arkitekturen er teknisk solid og veldesignet — supervised learning + beam search er det riktige valget for dette problemet, og VRAM/timing-estimatene er realistiske med god margin. De to reelle showstopperne er: (1) **supervised learning kan per definisjon ikke overstige heuristikkens eget tak** uten DAgger-iterasjon, og det er uklart om heuristikken nærmer seg et tak rundt 400-500 i nightmare eller om gapet mot 1361 er mye større; (2) **sim-to-live-gapet er større enn arkitekturen antar** — heuristikken scorer 393 live, men sim-score og live-score kan divergere betydelig for nye treningspunkter som ikke er recon-data. Anbefalingen er å kjøre oracle-assignment-spiken FØR noe som helst ML-kode skrives. Den tar 1 dag og forteller om ML overhodet kan tette gapet.

---

## Overordnet risikoprofil

- **Teknisk risiko:** Gul
- **Avhengighetsrisiko:** Gronn
- **Skaleringsrisiko:** Gronn (inference) / Gul (trening hvis Fase 2)
- **Kompetanserisiko:** Gul (PyTorch nytt for utvikler, men AI kompenserer godt)

---

## Kompleksitetskart

| Komponent | Kompleksitet | Usikkerhet | Begrunnelse |
|-----------|-------------|------------|-------------|
| ScorerMLP (nn.Module) | Gronn Lav | Gronn Lav | Standard PyTorch MLP, AI-sweet spot, 20 linjer kode |
| Treningsloop (AdamW, HuberLoss) | Gronn Lav | Gronn Lav | Kanonisk supervised learning, ingen RL-overhead |
| FeatureExtractor (GameState → 48 floats) | Gul Middels | Gronn Lav | Mye boilerplate, men alle inputs er kjente felt. Feil her gir stille dårlig trening |
| TrainingLogger (datafangst i sim) | Gronn Lav | Gronn Lav | BotAdapter-integrasjon er allerede etablert. Reward-5-beregning trenger litt etterbehandling |
| CandidateGenerator (top-K per bot) | Gronn Lav | Gronn Lav | BFS allerede implementert og cachet, filtrering er enkel |
| BeamSearch over global assignment | Gul Middels | Gul Middels | Domenespesifikk kombinatorikk med constraints (inventory caps, claimed_items, sticky assignments). Lettest å subtile feil |
| MLPlanner drop-in integrasjon | Gronn Lav | Gronn Lav | Interface er veldefinert (plan/maintain), fallback til V2TaskPlanner gjor rollback trivielt |
| Coordinator-integrasjon | Gronn Lav | Gronn Lav | En linje i __init__, nothing downstream endres |
| reward_5-labeling (etterbehandling) | Gul Middels | Gul Middels | Off-by-one på runde-indeks, edge case ved spillslutt (< 5 runder igjen), potential-based shaping er valgfritt men nyttig |
| Daglig retraining workflow | Gronn Lav | Gronn Lav | Shell-scripting, AI-sweet spot |
| Oracle-assignment test (spike) | Gronn Lav | Gul Middels | Enkel sim-kjoring med "all-knowing" planner, men krever domeneforstaelse for korrekt implementasjon |
| Fase 2: PettingZoo/TorchRL wrapper | Rod Hoy | Gul Middels | Simulator bruker Python-objekter/dicts, TorchRL krever TensorDict-basert API. Ikke trivielt |
| Fase 2: MAPPO med centralized critic | Rod Hoy | Rod Hoy | Reward shaping for 20 bots med sparse signal er dokumentert vanskelig. Credit assignment-problem er fundamentalt |
| Fase 2: PPO fine-tuning | Gul Middels | Rod Hoy | Hyperparameter-rom er stort, og iterasjonstid med CPU-sim er 4-12 timer per kjoring |

---

## Spike-liste (prioritert)

### Spike 1: Oracle-assignment — hva er theoretical max? (1 dag)

**Risikoen:** Gapet mot 1361 er 3.46x. Hvis optimal assignment i sim gir ~500, er ML-planner rette veien. Hvis den gir ~400, er bottlenecken PIBT/koordinering — og ML hjelper lite. Denne spiken kan drepe hele Fase 1-argumentet.

**Hypotesen:** Vi antar at suboptimal assignment (feil bot til feil item) er ansvarlig for minst 200-300 poeng av gapet mot toppscore.

**Eksperimentet:** Skriv en `OracleAssigner` som hvert runde kjorer BFS-optimaliseringen med full fremtidskunnskap: gi alltid den ledige boten med korteste total-vei (pick + dropoff) til items som matcher naeste ordre. Kjor mot nyeste nightmare-recon (`74001e7f_2026-03-16_*`). Sammenlikn med heuristikkens score pa samme recon.

**Suksesskriterium:** Oracle-score > 600 i sim = bevis for at assignment er bottlenecken, ML er rett investering.

**Hvis det feiler (oracle ~ 400-450):** Bottlenecken er PIBT/drop-off-kapasitet/korridor-gjennomstromning, ikke assignment. ML-planner er feil retning. Plan B: se pa trafikkstyring, one-way aisle-forbedringer, eller MAPF-solver for taktisk lag.

**AI-assistert estimat:** Claude Code kan generere OracleAssigner fra spec pa < 30 min. Manuell validering av output tar 2-3 timer. Total: ~4 timer med AI-hjelp vs 1 dag uten.

---

### Spike 2: MLP proof-of-concept pa historisk data (0.5 dag)

**Risikoen:** Supervised MLP kan overfit til spesifikke recon-sekvenser og laere "item X er alltid god" heller enn generelle heuristikker. Hvis valideringsloss ikke avtar med treningsdata, er feature-encodingen for svak.

**Hypotesen:** En 3-layer MLP trent pa (state, assignment, reward_5)-tripletter fra 50 sim-kjorninger mot kjent recon oppnar validation loss < 0.05 (Huber) og scorer hoyere enn random assignment pa held-out data.

**Eksperimentet:** Generer 50 sim-kjorninger med BotAdapter mot nyeste nightmare-recon. Logg (state_features, reward_5) per (bot, item)-par. Del 80/20 train/val. Tren ScorerMLP 20 epochs. Plot train- vs val-loss. Verifiser at scorer rangerer items korrekt (item narmere drop-off + riktig type bor score hoyere enn feil type langt unna).

**Suksesskriterium:** Val-loss konvergerer (ingen gap mot train-loss som tyder pa overfitting), og manuell inspeksjon av top-5 scorer per bot gir logiske assignments.

**Hvis det feiler:** FeatureExtractor mangler kritisk informasjon, eller reward_5 er for noisig til supervised signal. Plan B: bytt til reward_1 (kortere horisont, taettere signal) eller legg til potential-based shaping som supplementert reward.

**AI-assistert estimat:** Claude Code genererer trening + logg-kode raskt. Domenevalidering (er scorene logiske?) er manuell. Total: ~3 timer med AI mot 1 dag uten.

---

### Spike 3: BeamSearch korrekthet og timing (0.5 dag)

**Risikoen:** BeamSearch er den eneste nye komponenten med ikke-triviell kombinatorisk logikk. Feil her kan gi: double-booking (to bots assignet til samme item), violation av inventory-cap, eller brutt sticky-assignment-invariant. Disse feilene er subtile og gir ikke krasj — de gir bare darliger score.

**Hypotesen:** BeamSearch med width=20 over 20 bots og topp-5 kandidater per bot produserer gyldige assignments (ingen double-booking, claims-sett korrekt) og fullfores pa < 10ms.

**Eksperimentet:** Skriv BeamSearch, kjor mot 100 tilfeldige game states fra recon. Assert: ingen item claimet av to bots, alle assignments har gyldig item_id eller DELIVER/IDLE, timing < 10ms pa CPU (GPU er raskere). Legg til debug-logging for beam-bredde ved pruning.

**Suksesskriterium:** Alle 100 states passerer assertions, ingen constraint-violations, < 10ms pa CPU.

**Hvis det feiler:** Beam-ekspansjon respekterer ikke claimed_items korrekt. Plan B: forenkle til greedy-decode (velg beste action per bot sekvensielt) som korrekthetsfallback mens beam-logikken fikses.

**AI-assistert estimat:** Claude Code er flink pa kombinatorisk logikk med klare constraints. Test-generering er AI-sweet spot. Total: ~3 timer med AI mot 1 dag uten.

---

### Spike 4: Sim-to-live-gap kalibrering (0.5 dag, parallelt med Spike 1)

**Risikoen:** Modellen trenes pa sim-rollouts. Hvis sim-score for MLPlanner avviker sterkt fra live-score (slik det har skjedd tidligere, f.eks. 118 sim vs 52 live), er treningssignalet ugyldig for live-bruk.

**Hypotesen:** Dagens heuristikk (V2TaskPlanner) gir < 10% avvik mellom sim og live pa nyeste recon (post-kollisjonsmodell-fix 2026-03-08).

**Eksperimentet:** Kjor V2TaskPlanner mot 5 ulike recon-filer fra de siste 7 dagene. Sammenlikn sim-score med live-score logget i recon-JSON. Beregn gjennomsnittlig avvik.

**Suksesskriterium:** Avvik < 10% konsistent. Da er sim trygt nok som treningskilde.

**Hvis det feiler (avvik > 20%):** Treningsdata fra sim er biased. ML-modell trent pa sim vil underpreste live. Mitigering: tren primert pa live-logget data (logg faktiske beslutninger og utfall fra live-games), ikke bare sim-kjorninger.

**AI-assistert estimat:** Datainnhenting og analyse er enkel scripting. Total: 2 timer.

---

## Kritiske avhengigheter

| Avhengighet | Type | Risiko | Alternativ |
|-------------|------|--------|------------|
| PyTorch >= 2.2 med CUDA 11.8+ | Kritisk (trening + inference) | Lav — RTX A3000 er Ampere (compute 8.6), bredt stottet. Enkel a verifisere. | CPU-inference er mulig (<1ms), CUDA er kun for trening |
| BotAdapter + Simulator (eksisterende) | Kritisk (datafangst) | Lav — allerede validert og i bruk. | Ingen — bruk av live-games som fallback for datafangst |
| Recon-data (daglig oppdatering) | Kritisk (treningssignal) | Middels — recon samles ved forste live-game etter daglig reset. Manglende recon = ingen trening den dagen. | Bruk gaarsdagens modell som fallback (scorer generaliserer over item-types, ikke IDs) |
| GameLogger recon-format | Kritisk (FeatureExtractor-input) | Lav — format er stabilt. | N/A |
| scipy (Hungarian, eksisterende) | Viktig | Lav — allerede i bruk. | Ingen — beholdes uendret |
| Daglig reset kl 00:00 UTC | Viktig (retraining-vindu) | Middels — 30 min workflow ma fullfores for forste live-game. Sen oppstart = ingen modell den dagen. | Gaarsdagens checkpoint som fallback (V2TaskPlanner er alltid tilgjengelig) |
| PyTorch checkpoint-format | Nice-to-have | Lav — .pt-format er stabilt mellom PyTorch-versjoner. | torch.jit.script som alternativ serialisering |

**Vurdering av enkeltpunktsrisiko:**
- Ingen kritisk avhengighet er ekstern sky-tjeneste eller betalings-API — alt er lokalt eller open source
- Den eneste reelle flaskehals er recon-innsamling etter daglig reset — dette er prosedyremessig risiko, ikke teknisk

---

## Skaleringsanalyse

### 10x (5 difficulties, ikke bare nightmare)

MLPlanner slik den er beskrevet er nightmare-spesifikk (20 bots, 30x18 grid, spesifikke features som `dropoff_congestion` for nightmare-korridor). Pa easy (1 bot) og medium (3 bots) er heuristikken allerede sterk (124 og 151 poeng), og MLPlanner risikerer regresjon der siden treningsdata er fra nightmare.

**Konkret risiko:** `global_features`-vektoren inneholder `zone1/2/3_occupied` og `dropoff_congestion` som er nightmare-spesifikke. Pa easy er det ingen flaskehals ved drop-off. En felles modell vil laere feil representasjon for enkle kart.

**Mitigering:** Tren separate modeller per difficulty, eller ha difficulty-betingede features. Alternativt: behold V2TaskPlanner for easy/medium (der den allerede er OK) og bruk MLPlanner kun for nightmare.

**Inference-timing:** 10x i antall par (200 bots × 600 items = 120,000 forward passes) er relevant for andre konkurransen med mye storre kart. For navaerende maps er marginen >99% — dette er ikke et problem.

### 100x (tenkt: 200 bots, 300x180 grid)

BeamSearch med width=20 og 200 bots har O(bots × items × beam_width) vekst som ville blitt ~2.4M forward passes per runde. Dette er ikke innenfor 2-sekunders-budsjett pa CPU, men GPU-batching loser det.

BFS-distansecachen vokser kvadratisk med kartareal. Pa 300x180 = 54,000 noder med 200 destinasjoner = 10.8M BFS-operasjoner. Dette er utenfor rekkevidden av navaerende arkitektur.

### 1000x

Irrelevant for NM i AI-konkurransen. Konkurransen har faste mapstorrelser.

---

## Kompetansegap

| Omrade | Niva trengs | Tilgjengelig via AI? | Kommentar |
|--------|-------------|---------------------|-----------|
| PyTorch MLP + treningsloop | Junior | Ja, fullt ut | Standard boilerplate, AI genererer korrekt kode |
| CUDA-oppsett og verifisering | Junior | Ja, fullt ut | `torch.cuda.is_available()` + enkel test |
| Feature engineering (spilldomene) | Senior | Delvis | AI kan foreslaa features, men "er dette en god feature?" krever domeneforstaelse av spillets bottlenecks |
| BeamSearch med spillspesifikke constraints | Mid | Delvis | AI er flink pa kombinatorikk, men claimed_items/sticky-invarianter krever korrekthetskontroll av noen som kan spillreglene |
| Reward-shaping (Fase 2) | Senior/Spesialist | Nei | Iterasjonskrevende, domeneintuitiv prosess. AI kan foreslaa varianter, men riktig valg krever eksperimentell forstaelse |
| MAPPO/PPO fine-tuning (Fase 2) | Senior ML | Delvis | TorchRL-dokumentasjon er god, AI kjenner APIet, men hyperparameter-valg og diagnose av "policy collapsed" krever erfaring |
| Validering av treningssignal (er reward_5 korrekt?) | Mid | Delvis | AI kan lage unit tests, men logisk korrekthet av off-by-one i runde-indeks krever manuell gjennomgang |
| Spillregler og edge cases | Senior (domenekunnskap) | Nei | "Auto-delivery fires only for delivering bot", "items respawn med ny ID" — disse reglene er ikke i PyTorch-dokumentasjonen |

**AI-kompensasjonsoppsummering:**
- Kan kompensere godt (>70% tidsbesparelse): ScorerMLP, treningsloop, CLI-scripting, BeamSearch-skjelett, unit tests
- Kan kompensere delvis (40-60%): FeatureExtractor (genererer kode, men utvikler validerer at features er meningsfulle), BeamSearch-constraints, reward-labeling
- Kan ikke kompensere: Domeneforstaelse av spillbottlenecks, diagnose av "modellen laerer ikke" uten ML-erfaring, reward shaping for Fase 2

---

## "Hva om"-scenarier

### Hva om: MLP laerer ikke bedre enn greedy nearest

**Sannsynlighet:** Middels

**Konsekvens:** Fase 1 gir < 393 i sim. Prosjektet har brukt 5-6 dager pa infrastruktur uten score-fremgang.

**Mitigering:**
1. Oracle-spike (Spike 1) avslorer om ML i det hele tatt kan hjelpe
2. Sjekk validation loss — konvergerer den? Hvis loss er lav men score er ikke bedre, er det beam search som ikke utnytter scorer korrekt
3. Bytt til greedy-decode som debugging-steg — beam search legger til kompleksitet som kan maskere scorer-problemer
4. Sjekk feature-distribusjoner: er reward_5 for sparsom/noisig til supervised signal?

**Deteksjonstidspunkt:** Etter Spike 2 (dag 1-2). Tidlig nok til a endre kurs.

---

### Hva om: Supervised learning platayer under heuristikken

**Sannsynlighet:** Middels-Hoy

**Konsekvens:** Modellen laerer a etterligne heuristikken men ikke a overgaa den. Fase 1 scorer ~370-390 — marginalt darligere fordi beam search nar en suboptimal kombinasjon.

**Aarsak:** Supervised learning pa heuristikkens egne beslutninger laerer bare hva heuristikken gjor — ikke hva den burde gjort. Dette er det klassiske imitation-learning-taket. For a bryte gjennom trengs DAgger-iterasjon (scorer genererer egne beslutninger, samler ny data, retrainer) eller RL (Fase 2).

**Mitigering:** Design treningsdatagenereringen med tilfeldig perturbering: med 20% sannsynlighet velg et suboptimalt assignment og logg det med faktisk reward_5. Dette gir scorer informasjon om hva som er darlig, ikke bare hva heuristikken gjor.

**Deteksjonstidspunkt:** Etter 1-2 dager med live-testing. Planlegg for DAgger i Fase 1.5 hvis dette treffer.

---

### Hva om: Gap mot 1361 er fundamentalt — PIBT er bottlenecken, ikke assignment

**Sannsynlighet:** Hoy (dette er den mest sannsynlige forklaringen pa gapet)

**Konsekvens:** Selv med perfekt assignment scorer vi < 550 i nightmare. ML-planner forbedrer ingenting vesentlig. Konkurransen vinnes av et team med MAPF-solver (LaCAM, CBS, EECBS) eller fullstendig pre-computed plan.

**Mitigering:**
1. Kjor Oracle-spike FORst — hvis oracle gir < 550, bekreft at bottlenecken er taktisk
2. Undersok om toppscore pa 1361 er mulig med navaerende spillregler (300 runder, 20 bots, score 70+ poeng/runde krever nesten perfekt parallell levering hvert 4. runde)
3. Vurder taktisk forbedring som alternativ: bedre PIBT-prioritering, dynamisk sone-tildeling, pre-reservasjon av drop-off-posisjon

**Deteksjonstidspunkt:** Oracle-spike (dag 1). Absolutt tidligst mulig deteksjon.

---

### Hva om: FeatureExtractor introduserer stille feil

**Sannsynlighet:** Middels (dette er den vanligste typen feil i ML-pipelines)

**Konsekvens:** Modellen trener fint, loss konvergerer, men scorer er feil fordi f.eks. `dist_bot_to_item` bruker Manhattan-avstand i stedet for BFS (ikke korrekt for nightmare-korridor med vegger). Resultatet er at scorer laerer a optimalisere feil metrisk.

**Typiske feil a se etter:**
- BFS-distanse vs Manhattan — nightmare-kartet har smalpassasjer der Manhattan-avstand er fundamentalt feil
- Normalisering feil (dele pa `grid.width - 1` i stedet for `max_bfs_dist`) — komprimerer signal unnodig
- `item_id` vs `item_type` forvirring — scorer ma generalisere over IDs (de endres daglig), ikke huske dem
- `claimed_items`-sjekk mangler i FeatureExtractor — scorer laerer at claimede items er "gode" fordi heuristikken velger dem

**Mitigering:** Unit-test FeatureExtractor pa kjente game-states fra recon-filer med hardkodede forventede verdier. Dette er en kritisk test som bor skrives for noe som helst trening.

**Deteksjonstidspunkt:** Under Spike 2-validering (dag 2). Kan fikses uten a kaste treningspipeline.

---

### Hva om: Daglig reset-workflow brekker (manglende recon = ingen trening)

**Sannsynlighet:** Lav-Middels (human-in-the-loop workflow)

**Konsekvens:** Dagens MLPlanner bruker gaarsdagens checkpoint. Checkpoint er trent pa gaarsdagens item-typer og ordre-sekvens. Generaliseringsevnen avhenger av om item-distribusjonen er stabil dag-til-dag.

**Mitigering:**
1. V2TaskPlanner er alltid tilgjengelig som fallback — ingen score-tap fra null
2. Undersok om item-type-distribusjon er stabil (er det de samme 5-8 varetypene daglig, bare i ny rekkefolge?) — hvis ja, er generalisering OK
3. Automatiser recon-innsamling: kjor et recon-game automatisk kl 00:15 UTC, start trening umiddelbart

---

## Showstoppere

### Showstopper 1: PIBT-kapasitetstank er under 550 (sannsynlighet: hoy)

Hvis Oracle-spiken viser at selv perfekt assignment gir < 500-550 i nightmare-sim, er bottlenecken ikke assignment men taktisk gjennomstromning (PIBT-kollisjonskontroll, drop-off-korridor med x=1, spawn-stacking ved (28,16)). I sa fall kan MLPlanner maksimalt gi marginale forbedringer (+20-40 poeng) — ikke de +200-400 som er nodvendig for a naerme seg 1361. Prosjektet ma da pivotere til MAPF-solver-tilnaerming eller fundamentalt annerledes taktisk lag. **Dette er den viktigste spiken a kjore for noe som helst ML-kode skrives.**

### Showstopper 2: Supervised learning-tak er under heuristikk (sannsynlighet: middels)

Modellen trent pa heuristikkens egne beslutninger kan per definisjon ikke overgaa dem uten DAgger-iterasjon. Hvis heuristikkens tak er 393 og scorer laerer a etterligne den, gar vi ingenting. Dette er ikke en teknisk svikt — det er en iboende begrensning i "imitate the teacher"-supervised learning. Mitigering krever eksplisitt datainnsamling fra suboptimale valg (se Scenario 2).

---

## Anbefalt rekkefølge

1. **Kjor Oracle-assignment-spike FORst (Spike 1, 1 dag).** Bevis at assignment-forbedring i det hele tatt kan gi > 550 i sim. Hvis det ikke kan, er ML-planner feil retning og tid bor brukes pa taktisk lag i stedet. Ikke skriv en linje ML-kode for dette er besvart.

2. **Kjor Sim-to-live kalibrering parallelt (Spike 4, 2 timer).** Verifiser at dagens V2TaskPlanner har < 10% avvik mellom sim og live. Treningssignal fra sim er ugyldig ellers.

3. **MLP proof-of-concept pa historisk data (Spike 2, 0.5 dag).** Bevis at en scorer-MLP kan laere meningsfull rangering fra eksisterende recon-data. Dette avslorer feature-encoding-problemer tidlig.

4. **Bygg FeatureExtractor med unit-tester (dag 2-3).** Denne komponenten er den stille-feil-risikoen. Tester mot kjente game-states er nodvendig for trening starter.

5. **ScorerMLP + treningsloop (dag 3-3.5).** Standard PyTorch, AI-assistert, ingen overraskelser.

6. **TrainingLogger + datafangst (dag 3.5-4).** Integrer mot BotAdapter, valider reward_5-beregning med off-by-one-sjekk.

7. **BeamSearch med korrekthetssuite (Spike 3, dag 4-4.5).** Skriv assertions for constraint-violations, kjor mot 100 game-states, bekreft ingen double-booking.

8. **MLPlanner drop-in + live-test (dag 5).** Minimal endring i Coordinator, kjor mot nyeste nightmare-recon. Sammenlikn sim-score med V2TaskPlanner-baseline.

9. **Aktiver Fase 2 KUN hvis:** Fase 1 scorer stabilt > 500 i sim og live, og platayer. Ikke start MAPPO-integrasjon som en "kanskje det hjelper"-investering.
