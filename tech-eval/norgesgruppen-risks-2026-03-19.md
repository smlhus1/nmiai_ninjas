# Risiko- og kompleksitetsvurdering: NorgesGruppen Object Detection

> Vurdert: 2026-03-19 | Basert på: tech-eval/norgesgruppen-architecture-2026-03-19.md

---

## Oppsummering

Dette er en teknisk godt gjennomtenkt arkitektur med ett dominerende showstopper-risikoomrade: DINOv2 offline loading via timm er dokumentert ustabilt og krever ikke-opplagte workarounds som MÅ verifiseres dag 1. Sekundarrisikoen er 300s timeout — modell-load alene kan ta 30-60s, og med DINOv2-batch-inferens og eventuell TTA er marginen tynn. Det tredje kritiske risikoområdet er sikkerhetsscanneren: blokkerte imports som `os` og `subprocess` er standard biblioteksavhengigheter som timm og ultralytics bruker internt — kandidaten tror feilaktig at dette bare gjelder sin egen kode. Med 4 dager og 3 submissions/dag er det null rom for å oppdage showstoppere i innleveringsrunden. Spik DINOv2-loading og timeout-budsjettet dag 1 før noe annet.

---

## Overordnet risikoprofil

- **Teknisk risiko:** Rød
- **Avhengighetsrisiko:** Rød
- **Skaleringsrisiko:** Gronn (irrelevant — konkurranseformat)
- **Kompetanserisiko:** Gul

---

## Kompleksitetskart

| Komponent | Kompleksitet | Usikkerhet | Begrunnelse |
|-----------|-------------|------------|-------------|
| YOLOv8m trening (1-klasse detektor) | Gul Middels | Gronn Lav | Kjent API, pre-installert, COCO-vekter gir god startpunkt. Augmentasjonsparametre krever iterasjon men ikke ekspertise. |
| COCO-konvertering (22k annotations -> 1 klasse) | Gronn Lav | Gronn Lav | Standard JSON-manipulasjon. Claude Code skriver dette pa minutter. |
| DINOv2 offline loading i sandbox | Gul Middels | Rod Hoy | Dokumentert GitHub-issue: `custom_load=True` default for ViT forarsakar KeyError pa offline PyTorch weights. Krever `pretrained_cfg_overlay` eller `checkpoint_path` workaround. Vil uten tvil feile med naive `load_state_dict()`. |
| DINOv2 state_dict eksport (classify_model.pt) | Gul Middels | Gul Middels | `torch.save(model.state_dict())` er ikke nok — modell-arkitekturen ma ogsa re-instansieres korrekt uten nettverkstilgang. Kombinert med `custom_load`-problemet er dette ikke trivielt. |
| Nearest-neighbor klassifikasjon (cosine similarity) | Gronn Lav | Gronn Lav | 357 x 768 float32-matrise, torch.cosine_similarity + argmax. 10 linjer kode. |
| Copy-paste augmentasjon (custom implementasjon) | Gul Middels | Gronn Lav | 50-100 linjer PIL/numpy-kode. ADR-5 er korrekt: YOLOv8 copy_paste virker kun med segmenteringsmasker. Mye boilerplate, lav konseptuell usikkerhet. |
| run.py uten forbudte imports | Gul Middels | Rod Hoy | Sikkerhetsscanneren blokkerer `os`, `subprocess`, `socket` etc. MEN timm og ultralytics bruker disse internt. Det er uklart om scanneren sjekker kun din kode eller rekursivt alle imports. Ingen dokumentasjon funnet. |
| Submission packaging (420MB grense) | Gronn Lav | Gronn Lav | YOLOv8m (49.7MB) + DINOv2-base (346MB) + embeddings (1.1MB) = ~397MB. 23MB buffer. Lav risiko sa lenge DINOv2-base brukes. DINOv2-small som fallback (88MB) er klart alternativ. |
| 300s timeout-budsjett | Gul Middels | Rod Hoy | Usikkert hvor mye tid modell-loading tar pa L4 i sandbox. DINOv2-base er 346MB — disk-to-GPU load kan ta 5-15s. YOLO warmup 2-5s. Med 100-500 testbilder og to-stegs pipeline er marginen liten. Ensemble og TTA kan vaere umulig. |
| Stratified 80/20 split (254 bilder) | Gronn Lav | Gronn Lav | Standard sklearn stratify. Klarer seg med 51 val-bilder. |
| WBF ensemble (2 detektorer) | Gul Middels | Gronn Lav | Vektbudsjett tvinger DINOv2-base -> DINOv2-small ved ensemble. Kvalitetstap i klassifikasjon mot timeout-gevinst er usikker trade-off. |
| albumentations 1.3.1 API-kompatibilitet | Gronn Lav | Gul Middels | Eldre versjon — `CoarseDropout` er i 1.3.1, `CLAHE` er der, men noen transforms endret API mellom 1.3 og 1.4. `RandomShadow` med `shadow_roi` syntaks kan vaere anderledes. |
| COCO bbox output-format i run.py | Gul Middels | Gul Middels | YOLO returnerer xyxy-koordinater, COCO-format krever xywh. YOLO returnerer normaliserte koordinater [0,1], COCO vil ha pikselverdier. Feil her gir 0 score uten feilmelding. |
| Overfitting med 254 bilder | Gul Middels | Gul Middels | YOLOv8m med 25.9M params pa 254 bilder er risikabelt. Mosaic + mixup + erasing er aggressiv nok regularisering til at det sannsynligvis holder, men val-kurven ma monitoreres. |

---

## Spike-liste (prioritert)

### Spike 1: DINOv2 offline loading i isolert environment — DAG 1, MORGEN
**Estimert tid:** 2-4 timer

**Risikoen:** timm 0.9.12 har dokumentert `custom_load=True` default for ViT-modeller som forarsakar KeyError ved offline PyTorch weight loading. Arkitekturforslaget sier "torch.save(model.state_dict())" og "pretrained=False + load_state_dict()" — men dette er sannsynligvis utilstrekkelig alene og vil feile i sandbox.

**Hypotesen:** `timm.create_model('vit_base_patch14_dinov2', pretrained=False)` + `model.load_state_dict(torch.load('classify_model.pt'))` feiler med KeyError pga. `custom_load=True`.

**Eksperimentet:**
1. Last ned DINOv2-base vekter lokalt: `timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)`
2. Eksporter pa ulike mater: (a) `state_dict()`, (b) full model `torch.save(model)`, (c) HuggingFace `save_pretrained()`
3. Blokker nettverk (sett `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`)
4. Test alle tre lasting-metoder i isolert Python-prosess uten nettverkstilgang
5. Test spesifikt `pretrained_cfg_overlay=dict(file='path', custom_load=False)` workaround

**Suksesskriterium:** Model laster, produserer identiske embeddings som online-versjon (sammenlign cosine similarity pa et testbilde).

**Hvis det feiler:** Bytt til EfficientNet-B3 (12MB, ingen custom_load-problem, direkte state_dict) eller ConvNeXt-base. Begge er tilgjengelig i timm 0.9.12. Lavere embedding-kvalitet, men garantert fungerer.

**AI-assistert estimat:** Claude Code kan skrive test-scriptet og feilsoke feilmeldingene pa minutter. Selve eksperimenteringen krever manuell kjoering. Estimat: 1.5 timer med AI vs 4 timer uten.

---

### Spike 2: 300s timeout-budsjett maaling — DAG 1, ETTERMIDDAG
**Estimert tid:** 1-2 timer

**Risikoen:** Timeout-budsjettet er ukjent. Modell-load + warmup + inferens pa ukjent antall testbilder kan overskride 300s, spesielt med DINOv2-batch-embedding og eventuell ensemble/TTA.

**Hypotesen:** To-stegs pipeline pa 100-500 bilder tar 120-250s, noe som gir lite rom for ensemble eller TTA.

**Eksperimentet:**
1. Simuler sandbox: load YOLOv8m fra disk (ikke cache), load DINOv2 fra disk, kjor inferens pa 100/200/500 bilder
2. Timer: (a) model load, (b) YOLO detect alle bilder, (c) DINOv2 embed alle crops (estimat: 10-50 crops per bilde)
3. Maal total tid for ulike batch-stoerreiser og antall bilder
4. Test med og uten TTA (3x overhead)

**Suksesskriterium:** Tydelig budsjett: "X bilder ferdig pa Y sekunder" for de ulike konfigurasjonene.

**Hvis det feiler (>250s for baseline):** Deaktiver TTA og ensemble. Vurder ONNX-eksport av DINOv2 for raskere inferens (onnxruntime-gpu er pre-installert).

**AI-assistert estimat:** Claude Code skriver timing-harness pa 10 minutter. Eksperimenteringen er manuell. Estimat: 1 time med AI.

---

### Spike 3: Sikkerhetsscanner — hva blokkeres egentlig?
**Estimert tid:** 30 minutter

**Risikoen:** Arkitekturnotatet sier "ingen os/subprocess/socket i run.py" — men timm og ultralytics bruker begge disse internt. Er scanneren AST-basert pa din kode alene, eller rekursiv pa alle imports?

**Hypotesen:** Scanneren sjekker kun run.py direkte, ikke transitive imports (ellers ville ingen ML-submission fungere).

**Eksperimentet:**
1. Les konkurranse-dokumentasjonen grundig — se etter eksakte regler for forbudte imports
2. Sjekk om andre deltakere har postet om dette i Discord/forum
3. Les nmiai MCP-dokumentasjon for sandbox-regler

**Suksesskriterium:** Tydelig forstaelse av hva som er tillatt og ikke.

**Hvis det feiler (rekursiv scan):** Hele tilnaermingen er blokkert. Plan B: ren PyTorch-modell uten timm-wrapper.

**AI-assistert estimat:** Claude Code kan lese og analysere all tilgjengelig dokumentasjon umiddelbart.

---

### Spike 4: COCO bbox output-format validering
**Estimert tid:** 1 time

**Risikoen:** YOLO returnerer xyxy normalisert [0,1]. COCO forventer xywh i piksler. Feil koordinat-format gir mAP = 0 uten feilmelding fra serveren.

**Hypotesen:** run.py-koden ma eksplisitt konvertere: `x = x_center_norm * width`, `w = w_norm * width` etc.

**Eksperimentet:**
1. Skriv et lite validerings-script som sammenligner output mot COCO ground truth-format
2. Sjekk at category_id i output matcher konkurransen sine category_ids (ikke YOLO-klassindeks)
3. Valider at bounding boxes har riktig koordinatsystem (piksel, ikke normalisert)

**Suksesskriterium:** Output matcher forventet COCO JSON-schema eksakt.

**AI-assistert estimat:** Claude Code skriver validerings-scriptet og format-konverteringen umiddelbart.

---

## Kritiske avhengigheter

| Avhengighet | Type | Risiko | Alternativ |
|-------------|------|--------|------------|
| timm 0.9.12 + DINOv2-base | Kritisk | Offline loading er dokumentert ustabilt for ViT-modeller. Uten nettverkstilgang i sandbox kan loading feile pa overraskende mater. | EfficientNet-B3 (12MB, ingen custom_load-issue) eller ConvNeXt-base. Begge lavere embedding-kvalitet. |
| ultralytics 8.1.0 | Kritisk | Spesifikk versjon pinned i sandbox. 8.1.0 er fra januar 2024 — noen nyere augmentasjonsfeatures ikke tilgjengelig. Ikke kritisk for kjernebruken. | Ingen direkte alternativ, men alle nodvendige features finnes i 8.1.0. |
| albumentations 1.3.1 | Viktig | API-endringer mellom 1.3 og 1.4 kan gjore at copy-paste augmentasjon-kode skrevet mot nyere versjon ikke fungerer. | Bruk kun features som er bekreftet i 1.3.1. Torchvision transforms som fallback. |
| ensemble-boxes 1.0.9 (WBF) | Viktig | Stabilt bibliotek. Minimal risiko. | Standard NMS via ultralytics som fallback. |
| NVIDIA L4 (24GB VRAM) | Kritisk | Sandbox-infrastruktur vi ikke kontrollerer. Hvis VRAM er delt eller begrenset kan DINOv2-base + YOLOv8m sammen kreve mer enn tilgjengelig. | Reduser batch size. DINOv2-small som fallback. |
| 300s sandbox timeout | Kritisk | Ukjent antall testbilder. Ingen mulighet til a se timeoutfeil for submission. Timeout = 0 score pa hele submission. | Fjern TTA og ensemble. ONNX-eksport for hurtigere inferens. |

---

## Skaleringsanalyse

Dette er et konkurranseformat, ikke et produksjonssystem. Skaleringsrisiko er irrelevant utover timeout-budsjett.

**10x (test images):** L4 med 24GB VRAM har rikelig kapasitet for hundrevis av bilder i batch. Flaskehalsen er 300s timeout, ikke GPU-kapasitet.

**100x / 1000x:** Ikke relevant for dette prosjektet.

---

## Kompetansegap

| Omrade | Niva trengs | Tilgjengelig via AI? | Kommentar |
|--------|-------------|---------------------|-----------|
| YOLOv8 trening og validering | Junior-Mid | Ja, godt | Ultralytics API er godt dokumentert, Claude Code kan skrive all kode. Domeneforstaelse (augmentasjonsparametre, overfitting-signaler) trengs for iterasjon. |
| DINOv2 / timm embedding-pipeline | Mid | Delvis | Boilerplate er enkelt med AI, men debugging av offline loading-problemer (custom_load, checkpoint_path) krever forstaelse av hva som skjer under pansen. |
| COCO annotation-format | Junior | Ja, godt | Standard JSON-format, Claude Code kan konvertere og validere. |
| Computer vision debugging | Mid | Delvis | Visualisering av deteksjoner, confusion matrices, feilanalyse — AI kan hjelpe, men krever domeneforstaelse for a tolke resultatene. |
| Copy-paste augmentasjon (custom) | Mid | Ja, godt | 50-100 linjer PIL/numpy-kode, tydelig spesifikasjon i ADR-5. Claude Code skriver dette raskt. |
| Sandbox-spesifikke constraints | Ukjent | Nei | Ingen i teamet vet sikkert hva scanneren tillater og blokkerer. Krever eksperimentering. |
| mAP@0.5 tolkning og iterasjonsstrategi | Mid | Delvis | AI kan forklare metrikken, men "hva gjores na for a fa +5 mAP" krever domeneerfaring. |

**AI-kompensasjonsvurdering:**

Claude Code er meget nyttig for all boilerplate og kjent kode (data-konvertering, pipeline-assembly, format-validering), men kompenserer IKKE for:
- Tolkning av val-kurver under trening (er dette overfitting eller bare variasjon?)
- Beslutning om augmentasjonsstyrke er riktig for disse spesifikke bildene
- Debugging av modell-atferd (hvorfor detekterer den ikke varer i hoyrekant av bilde?)
- Forstaelse av om DINOv2-embeddings faktisk skiller mellom NorgesGruppen-produkter i praksis

---

## "Hva om"-scenarier

### Hva om: DINOv2 offline loading feiler i sandbox
**Sannsynlighet:** Hoy (dokumentert GitHub-issue, ingen enkel workaround)
**Konsekvens:** Klassifikasjonssteg feiler. Enten: (a) hele run.py krasjer = 0 score, (b) fallback til "alle produkter er klasse 0" = only detection mAP bidrar.
**Mitigering:** Spike 1 DAG 1. Ha EfficientNet-B3 klar som drop-in-erstatning (12MB, ingen custom_load-problem, direkte state_dict).
**Deteksjonstidspunkt:** Oppdages i Spike 1 hvis kjort korrekt. Hvis ikke spiket: oppdages pa submission 1 = 1/3 av dagens submissions bortkastet.

---

### Hva om: 300s timeout overskrides
**Sannsynlighet:** Middels (avhenger av antall testbilder — ukjent)
**Konsekvens:** Submission "tidsavbrytes" = 0 score. Ingen feilmelding, bare 0.
**Mitigering:** Maale timeout-budsjett i Spike 2 dag 1. Deaktiver TTA og ensemble fra start — legg dem til kun hvis budsjett bekreftes. Vurder TensorRT/ONNX-eksport for DINOv2 (onnxruntime-gpu er pre-installert, kan gi 2-4x speedup).
**Deteksjonstidspunkt:** Oppdages pa forste submission hvis ikke pre-testet. Deretter diagnostisk submission som kaster en dag.

---

### Hva om: Sikkerhetsscanneren blokkerer timm eller ultralytics internt
**Sannsynlighet:** Lav (men ukjent — ingen dokumentasjon funnet)
**Konsekvens:** Hele submission blokkeres ved innlevering (foer sandbox kjoering). 0 score.
**Mitigering:** Les konkurranse-dokumentasjon grundig. Sjekk Discord/forum for presedenser. Spike 3.
**Deteksjonstidspunkt:** Umiddelbart ved submission-validering. Men da er det for sent med 3 submissions/dag.

---

### Hva om: DINOv2 embedding-kvalitet er for darlig for NorgesGruppen-produkter
**Sannsynlighet:** Lav-Middels (litteraturen viser 93.94% accuracy pa retail, men det var treningsdata inkludert)
**Konsekvens:** Klassifikasjon mAP 10-20% i stedet for 25-40%. Kombinert score 37-46% i stedet for 47-62%.
**Mitigering:** Maale klassifikasjon mAP isolert pa val-crops dag 3 (GT crops mot prototyper). Hvis < 20%: vurder fine-tuning av DINOv2 pa crops + referansebilder (krever mer tid men er gjennomforbart).
**Deteksjonstidspunkt:** Dag 3 i treningsplanen — tidlig nok til a iterere.

---

### Hva om: COCO bbox-format er feil i output
**Sannsynlighet:** Middels (lett a gjore feil — YOLO bruker normalisert xyxy, COCO vil ha piksel xywh)
**Konsekvens:** mAP = 0 pa hele submission uten forklarende feilmelding. Vanskelig a diagnostisere.
**Mitigering:** Spike 4 — skriv et validerings-script som sammenligner output-format mot forventet COCO-schema og sjekker at koordinater er i riktig range (piksler, ikke 0-1).
**Deteksjonstidspunkt:** Kan passere ubemerket hvis ikke eksplisitt testet. Oppdages forst pa submission.

---

### Hva om: Treningskjoring overskrider lokal GPU-kapasitet
**Sannsynlighet:** Lav (RTX 3090 anbefalt i arkitektur, YOLOv8m batch=16 er ~ 14GB VRAM)
**Konsekvens:** Treningskjoring feiler med OOM, mister tid dag 1-2.
**Mitigering:** Start med batch=8, skaler opp. Bruk `amp=True` (FP16 halverer VRAM-bruk).
**Deteksjonstidspunkt:** Umiddelbart ved forste treningskjoring.

---

## Showstoppere

### Showstopper 1: DINOv2 offline loading
Sannsynlighet: Hoy. timm 0.9.12 har en dokumentert bug/adferd der ViT-modeller med `custom_load=True` feiler ved offline PyTorch weight loading. Arkitekturens naive loesning (`pretrained=False` + `load_state_dict`) vil mest sannsynlig feile med `KeyError: 'embedding/kernel is not a file in the archive'` eller lignende. Dette MÅ spikes dag 1.

**Workaround som er dokumentert a fungere:**
```python
model = timm.create_model(
    'vit_base_patch14_dinov2.lvd142m',
    pretrained=True,
    pretrained_cfg_overlay=dict(file='classify_model.pt', custom_load=False)
)
```
Eller alternativt: bytt til EfficientNet-B3 som ikke har dette problemet.

### Showstopper 2: Feil COCO output-format
Sannsynlighet: Middels. Koordinatfeil gir 0 mAP uten feilmelding. Lavere sannsynlighet fordi det er lett a teste lokalt, men fort gjort a oversee hvis submission pakkes raskt.

### Ingen andre identifiserte showstoppere
300s timeout er en risikorisiko men ikke en absolutt stopper — kan mitigeres ved a fjerne ensemble/TTA.

---

## Anbefalt rekkefølge

1. **Dag 1 morgen: Spike 1 (DINOv2 offline loading)** — dette er det eneste som kan drepe hele arkitekturen. Fa dette bekreftet fungerende eller bytt til EfficientNet-B3 FOR NOE ANNET. Bruk nettverksblokk (`TRANSFORMERS_OFFLINE=1`) for realistisk test.

2. **Dag 1 morgen parallelt: Spike 3 (sikkerhetsscanner)** — les all tilgjengelig dokumentasjon (nmiai MCP, Discord, konkurranse-regler) for a forsta hva scanneren faktisk sjekker.

3. **Dag 1 ettermiddag: Spike 2 (timeout-budsjett)** — timer modell-load + inferens pa 100/200/500 bilder. Beslut om TTA og ensemble er mulig eller ma skjares bort fra start.

4. **Dag 1 ettermiddag: COCO-konvertering og data setup** — dette er lav risiko og kan kjoeres parallelt med treningskjoring. Claude Code skriver all kode her.

5. **Dag 1 kveld / Dag 2: YOLOv8m baseline-trening** — start treningskjoring, valider val-mAP > 30%. Dette er flaskehalsen (40-60 min per run). Kjoer nattlig dersom mulig.

6. **Dag 2: Copy-paste augmentasjon + forbedret trening** — Claude Code implementerer dette raskt. Spesifikk augmentasjonsstyrke ma itereres mot val-mAP.

7. **Dag 3: To-stegs pipeline assembly og Spike 4 (COCO format-validering)** — sett sammen hele pipeline i run.py, valider output-format eksplisitt.

8. **Dag 3 kveld: Forste submission** — med YOLOv8m + DINOv2 (eller EfficientNet-B3 fallback), ingen ensemble/TTA, verifisert COCO-format.

9. **Dag 4: Iterasjon basert pa leaderboard-feedback** — legg til ensemble/TTA kun hvis timeout-budsjett bekreftes og score-gevinst er tydelig.

---

## Vekttabell for hurtigreferanse

| Konfigurasjon | detect_model | classify_model | Total | Status |
|---------------|--------------|----------------|-------|--------|
| Primær (anbefalt) | YOLOv8m 49.7MB | DINOv2-base 346MB | ~397MB | OK (23MB buffer) |
| Ensemble (2 detektorer) | YOLOv8m+l 137MB | DINOv2-small 88MB | ~227MB | OK, men lavere klassif.kvalitet |
| Minimal fallback | YOLOv8m 49.7MB | EfficientNet-B3 12MB | ~63MB | OK, enklest a fa til a fungere |

**Anbefaling for forste submission:** Primær-konfigurasjon (YOLOv8m + DINOv2-base) — men kun hvis Spike 1 bekrefter at DINOv2 offline loading fungerer. Ellers: minimal fallback med EfficientNet-B3 for a fa noe inn pa leaderboard dag 1.
