# MVP Specification: NorgesGruppen Object Detection

> Basert på: norgesgruppen-architecture-2026-03-19.md + norgesgruppen-risks-2026-03-19.md

## MVP-mal

Levere en fungerende submission som scorer >0.50 combined mAP (0.7 x detection + 0.3 x classification) innen dag 2 kveld. Bevise at two-stage pipeline (class-agnostic detektor + DINOv2 nearest-neighbor) fungerer i sandbox med alle constraints.

## Scope: Inn og ut

### Inn (disse 5 features)
1. **Class-agnostic YOLOv8m detektor** — 1-klasse "product" pa alle 22,300 annotations. Maks utnyttelse av begrenset data.
2. **DINOv2-base embedding-klassifikator** — frozen feature extractor med k-NN mot 327 referanseprodukt-prototyper. Ingen treningssteg for klassifikatoren.
3. **run.py inferenspipeline** — load weights, detect, crop, embed, classify, COCO JSON output. Alle sandbox-constraints overholdt.
4. **COCO format-validering** — lokalt validerings-script som verifiserer output-format for innlevering.
5. **Baseline augmentasjon** — YOLOv8 built-in (mosaic, mixup, HSV, flip) uten custom copy-paste.

### Ut (bevisst utelatt)
1. **Copy-paste augmentasjon** — hoy ROI (+5-10 AP) men krever 50-100 linjer custom kode. Legges til etter forste submission.
2. **Multi-modell ensemble (WBF)** — krever 2 detektorer og timeout-budsjett-bekreftelse. Kun med DINOv2-small (lavere klassif.kvalitet).
3. **Test-Time Augmentation (TTA)** — 3x overhead pa timeout. Risikabelt for ukjent antall testbilder.
4. **DINOv2 fine-tuning** — krever treningsloop for feature extractor. Kun relevant hvis k-NN klassifikasjon < 20% mAP.
5. **Pseudo-labeling** — krever tilgang til testbilder. Fase 2.

## Teknisk MVP-stack

| Komponent | MVP-valg | Iterasjons-valg (senere) | Begrunnelse |
|-----------|----------|--------------------------|-------------|
| Detektor | YOLOv8m, 1-klasse, built-in aug | YOLOv8m + copy-paste aug + YOLOv8l ensemble | Raskeste vei til fungerende submission |
| Klassifikator | DINOv2-base frozen + k-NN | Fine-tuned DINOv2 pa crops + referansebilder | Null treningssteg, fungerer out-of-box |
| Fallback klassifikator | EfficientNet-B3 (12MB) | — | Brukes KUN hvis DINOv2 offline loading feiler (Spike 1) |
| Embedding-store | torch tensor (357 x 768 float32) | — | ~1MB, ingen fancy vector DB nodvendig |
| Output-format | COCO JSON med validerings-script | — | Feil format = 0 score, validering er kritisk |
| Packaging | pathlib-basert, ingen os/subprocess | — | Sandbox-krav |

## Effort-estimat

### Tradisjonell utvikling
| Oppgave | Estimat | Forutsetning |
|---------|---------|--------------|
| Spike 1: DINOv2 offline loading | 4 timer | Krever manuell eksperimentering |
| Spike 2: Timeout-budsjett | 2 timer | Timer modell-load + inferens |
| Data setup (COCO konvertering, split) | 1 dag | Standard JSON-manipulasjon |
| YOLOv8m baseline trening | 0.5 dag | 40-60 min GPU-tid + validering |
| DINOv2 embedding-pipeline | 1.5 dager | Last vekter, generer prototyper |
| run.py inferenspipeline | 1 dag | Two-stage pipeline + COCO output |
| Validering + packaging | 0.5 dag | Format-sjekk + zip |
| **Totalt** | **~5 dager, 1 person** | |

### AI-assistert utvikling (Claude Code)
| Oppgave | Estimat | AI-gevinst | Forutsetning |
|---------|---------|------------|--------------|
| Spike 1: DINOv2 offline loading | 1.5 timer | 60% | AI skriver test-script, manuell kjoering |
| Spike 2: Timeout-budsjett | 1 time | 50% | AI skriver timing-harness |
| Data setup (COCO konvertering, split) | 2 timer | 75% | Standard COCO, AI genererer kode |
| YOLOv8m baseline trening | 0.5 dag | 0% | GPU-tid er flaskehalsen |
| DINOv2 embedding-pipeline | 2 timer | 85% | Boilerplate timm-kode |
| run.py inferenspipeline | 3 timer | 75% | Pipeline-kode, AI genererer |
| Validering + packaging | 1 time | 80% | Format-sjekk er mekanisk |
| **Totalt** | **~1.5 dager, 1 person** | **~70% reduksjon** | AI-erfaren utvikler |

### Viktige caveats
- GPU-treningskjoering (40-60 min per YOLOv8-run) er IKKE pavirket av AI-assistanse
- DINOv2 offline loading-spike er risikofylt — kan ta 30 min eller 4 timer avhengig av bugs
- 300s timeout i sandbox er ukjent — kan tvinge arkitekturendringer
- Estimater forutsetter at treningsdata allerede er lastet ned
- AI-estimater forutsetter utvikler med erfaring i AI-assistert utvikling
- Estimater inkluderer IKKE: treningsdatanedlasting, GPU-trening-ventetid, submission-koo

## Spikes a kjore FOR MVP-utvikling

1. **DINOv2 offline loading** — 1.5t med AI — Avgjor om kjernearkitekturen fungerer eller om vi ma bytte til EfficientNet-B3. KJORES FORST.
2. **Sikkerhetsscanner-regler** — 30 min — Les docs/Discord for a forsta hva som blokkeres. Avgjor om timm/ultralytics imports er trygge.
3. **300s timeout-budsjett** — 1t — Timer modell-load + inferens pa N bilder. Avgjor om ensemble/TTA er mulig.
4. **COCO bbox-format** — 30 min — Skriv validerings-script. Forhindrer 0-score pa forste submission.

## Proof Points

Nar MVPen er ferdig, kan vi svare pa:
- Fungerer DINOv2 offline loading i sandbox-lignende environment?
- Hva er realistisk detection mAP@0.5 med 254 bilder og class-agnostic YOLOv8m?
- Hva er classification mAP@0.5 med k-NN mot referanseprototyper?
- Holder 300s for two-stage pipeline pa testbildene?
- Hva er combined score pa public leaderboard?

Vi kan IKKE svare pa enna:
- Om copy-paste augmentasjon gir +5-10 AP pa akkurat dette datasettet
- Om ensemble (m+l) gir bedre score enn single model nar vektbudsjett tvinger DINOv2-small
- Om fine-tuning av DINOv2 pa crops forbedrer klassifikasjonen
- Om pseudo-labeling pa testbilder er mulig og nyttig
- Hva private test set score er (endelig ranking)
