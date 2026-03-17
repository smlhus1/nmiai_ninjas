# MVP Specification: GPU-Based ML Planner for NM i AI

> Basert på: `tech-eval/ml-planner-architecture-2026-03-16.md` + `tech-eval/ml-planner-risks-2026-03-16.md`

## MVP-mål

Bevise at en lært scorer (MLP) + beam search gir **høyere nightmare-score enn nåværende heuristikk (393)** på recon-data i validert sim. Sekundært: at inference er rask nok for live-spill (<100ms).

## Scope: Inn og ut

### ✅ Inn (disse 5 features)
1. **FeatureExtractor** (48 floats per bot-item-par) — kjernen som gjør game state maskinlesbart
2. **ScorerMLP** (144K params, 3-layer MLP) — scorer kvaliteten av en (bot, item)-assignment
3. **TrainingLogger + datafangst** — genererer (state, assignment, reward_5) fra sim via BotAdapter
4. **BeamSearch** (width=20) — kombinatorisk søk over global bot-assignment
5. **MLPlanner** (drop-in for V2TaskPlanner) — orkestrerer inference, fallback til heuristikk

### ❌ Ut (bevisst utelatt)
1. **PPO/RL fine-tuning (Fase 2)** — avhenger av at Fase 1 platåer over 500
2. **GNN-arkitektur (Fase 3)** — kun hvis MLP-features er for svake
3. **Sim-akselerasjon (JAX/CUDA-port)** — supervised learning trenger ikke rask sim
4. **Multi-difficulty modeller** — behold V2TaskPlanner for easy/medium/hard, ML kun for nightmare
5. **Automatisert daglig retraining** — manuelt workflow er tilstrekkelig for MVP
6. **DAgger-iterasjon** — først etter at vanilla supervised learning er evaluert

## Teknisk MVP-stack

| Komponent | MVP-valg | Prod-valg (senere) | Begrunnelse |
|-----------|----------|-------------------|-------------|
| ML framework | PyTorch 2.2+ (CUDA) | Samme | Eneste reelle valg for lokal GPU-trening |
| Modell | MLP 48→256→256→128→1 | GNN (Fase 3) | Raskest å bygge, beviser hypotesen |
| Treningsdata | Pickle-filer (`data/*.pkl`) | Samme | <500MB, ingen database nødvendig |
| Checkpoint | `torch.save()` → `models/*.pt` | Samme | ~800KB per checkpoint |
| Search | BeamSearch (width=20) | Samme, evt. MCTS | God nok for 20 bots × 5 kandidater |
| Sim | Eksisterende `Simulator` + `BotAdapter` (CPU) | Evt. vectorized (Fase 2+) | Allerede validert, 50 spill ~10 min |
| Integrasjon | Drop-in MLPlanner i Coordinator | Samme | Ett-linje endring i `__init__` |

## Effort-estimat

### Tradisjonell utvikling
| Oppgave | Estimat | Forutsetning |
|---------|---------|--------------|
| Oracle-spike (Spike 1) | 1 dag | Avklarer om ML er rett retning |
| FeatureExtractor + unit tests | 3 dager | 48 floats, edge cases, BFS vs Manhattan |
| ScorerMLP + treningsloop | 2 dager | Standard PyTorch |
| TrainingLogger + datafangst | 1 dag | BotAdapter-integrasjon |
| BeamSearch + korrekthetstester | 2 dager | Constraint-logikk, double-booking |
| MLPlanner + Coordinator-integrasjon | 1 dag | Drop-in replacement |
| Testing + validering mot recon | 2 dager | Sim-score sammenligning |
| Daglig retraining CLI | 1 dag | Shell-scripting |
| **Totalt** | **~13 dager, 1 person** | |

### AI-assistert utvikling (Claude Code)
| Oppgave | Estimat | AI-gevinst | Forutsetning |
|---------|---------|------------|--------------|
| Oracle-spike (Spike 1) | 0.5 dag | ~50% | AI genererer, bruker validerer |
| FeatureExtractor + unit tests | 1 dag | ~67% | Boilerplate-encoding, AI sweet spot |
| ScorerMLP + treningsloop | 0.5 dag | ~75% | Kanonisk PyTorch |
| TrainingLogger + datafangst | 0.5 dag | ~50% | Integrasjon mot eksisterende |
| BeamSearch + korrekthetssuite | 1 dag | ~50% | Domenespesifikk kombinatorikk |
| MLPlanner + Coordinator-integrasjon | 0.5 dag | ~50% | Interface-matching |
| Testing + validering mot recon | 1 dag | ~50% | Domenevalidering er manuell |
| Daglig retraining CLI | 0.25 dag | ~75% | Script-generering |
| **Totalt** | **~5-6 dager, 1 person** | **~57% reduksjon** | |

### Viktige caveats
- Oracle-spike (dag 1) kan drepe hele prosjektet — hvis oracle < 500, er assignment ikke bottlenecken
- AI-estimater forutsetter utvikler som kan validere PyTorch-output og spilldomenet
- Estimater inkluderer IKKE: live-testing mot server, iterasjon på features etter første resultat

## Spikes å kjøre FØR MVP-utvikling

1. **Oracle-assignment spike** — ⏱️ 0.5 dag — Hva er theoretical max med perfekt assignment + nåværende PIBT? Hvis < 500: ML er feil retning. Hvis > 600: grønt lys.

2. **MLP proof-of-concept** — ⏱️ 0.5 dag — Kan en MLP lære meningsfull ranking fra 50 sim-kjøringer? Avslører feature-encoding-problemer tidlig.

3. **Sim-to-live kalibrering** — ⏱️ 2 timer — Er avviket mellom sim og live < 10%? Hvis > 20%: treningssignalet er ugyldig.

## Proof Points

Når MVPen er ferdig, kan vi svare på:
- ✅ Kan en lært scorer slå greedy nearest assignment i sim? (score > 393)
- ✅ Er inference rask nok for live-spill? (<100ms per runde)
- ✅ Generaliserer scorer til nye ordresekvenser? (test på annen recon enn treningsdata)
- ✅ Hva er gap mellom scorer og oracle? (rom for Fase 2 forbedring)

Vi kan IKKE svare på ennå:
- ❌ Kan RL (Fase 2) bryte gjennom supervised learning-taket?
- ❌ Er GNN bedre enn MLP for dette problemet?
- ❌ Kan vi nå 1361 med assignment-forbedring alene, eller trengs taktisk forbedring?
- ❌ Fungerer daglig retraining pålitelig over tid?
