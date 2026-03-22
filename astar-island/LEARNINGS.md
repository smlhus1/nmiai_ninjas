# Astar Island — Retrospektiv

**NM i AI 2026, Runde 1–22 (mars 2026)**

## Hva var oppgaven

Astar Island var en terrengprediksjons-challenge: gitt en 40x40 øy med vikingbosettinger, skog, fjell og hav, skulle vi predikere hva hver celle ble etter 50 år med stokastisk simulering. Vi fikk 5 seeds per runde og 50 observasjonsqueries totalt (15x15 viewport per query). Svaret var en sannsynlighetsfordeling over 6 klasser per celle — scoret med entropi-vektet KL-divergens.

Utfordringen: vi ser bare ~56% av kartet per seed med perfekt dekning (9 viewports), simuleringen er stokastisk, og bosettingene kan alt fra blomstre til bli utryddet av vinter.

## Tilnærmingen vår

Vi endte opp med en 4-fase pipeline i `solver_v3.py`:

1. **OBSERVE** — 5 seeds x 10 queries (9 grid + 1 bosettings-tett viewport). Gir ~8000 observerte celler med full kartdekning.
2. **ESTIMATE** — ABC rejection sampling: 200 parameterkombinasjoner kjøres gjennom en lokal simulator, de 10 nærmeste til observert survival/expansion/ruin-rate beholdes.
3. **CALIBRATE** — Dirichlet-prior bygd fra vektede historiske runder. Softmax-vekter basert på transition-vektor-avstand (E→S%, S→S%, S→E%, F→S%).
4. **PREDICT** — Alle kilder bidrar pseudo-counts til en Dirichlet-posterior: Jeffreys-prior (0.5), historisk prior, cross-seed-modell, ABC-kalibrert MC-sim, og direkte observasjoner. Ingen hardkodede blandingsvekter — alt er pseudo-counts.

Feature-nøkkelen per celle: `(init_class, settlement_dist, food_bin, ocean_dist, frontier, density_bin)`.

## Hva som fungerte

**Cross-seed feature-modell (+25 pts).** Den desidert viktigste innsikten. Alle 5 seeds deler de samme skjulte simuleringsparametrene. En skog-celle 3 steg fra en bosetting oppfører seg likt uansett seed. Ved å bygge en feature→distribusjon-lookup fra ALLE observasjoner på tvers av seeds, fikk vi dramatisk bedre prediksjoner for uobserverte celler.

**Dirichlet pseudo-count framework.** Elegant og robust. I stedet for `if obs: use obs, elif cross_seed: blend 70/30, else: historical` ble alt pseudo-counts i en felles Dirichlet. Flere observasjoner → mer vekt automatisk. Ingen tuning av blandingsvekter.

**ABC-kalibrert simulator.** En lokal simulator med parameterestimering via rejection sampling. Tok ~27 sekunder, men ga en fysikkbasert prior som hjalp spesielt på celler langt fra observerte områder.

**Automatisert pipeline (night_watch).** En cron-drevet pipeline som automatisk observerte, løste, submittet og sendte resultater til Discord. Gikk fra rank ~37 til ~20 over natta bare ved å submitte konsistent på alle runder vi ellers ville sovet gjennom.

**60+ eksperimenter.** Systematisk testing av alt fra MLP-modeller til isotonisk kalibrering. De fleste ga null eller negativ effekt — men prosessen sikret at vi visste hva som IKKE fungerte.

## Hva som IKKE fungerte

**MLP/neural approaches.** Med bare 4-5 seeds x ~1600 dynamiske celler var datagrunnlaget altfor tynt. MLP ga +3.4 i isolert eksperiment, men dette VAR cross-seed-transfer — som vi allerede hadde via lookup.

**Isotonisk kalibrering.** +0.08 i snitt, men overfittet på individuelle runder. Biasene var reelle men for små til å rettferdiggjøre kompleksiteten.

**Sim blend med heuristiske parametere.** MC-simulator med manuelt justerte parametere ga bare +0.01. Simmen matchet ikke serveren godt nok.

**Temperature scaling.** T=1.0 var allerede optimalt. Ingen gevinst.

**Bayesian observation denoising.** Katastrofe: -33 til -59 poeng. One-hot observasjoner ER korrekte — å "glatte dem ut" fjerner informasjon.

## Overraskelser og learnings

**Brutal vs stabil varierer enormt.** R8 hadde 0% settlement survival (100% S→E), mens R11 hadde 95.2%. Samme modell måtte håndtere begge — og cross-seed-modellen tilpasset seg automatisk basert på observasjoner. Ultra-brutale runder var faktisk ENKLERE å predikere (alt dør = lav entropi).

**Cross-seed var den største enkelt-forbedringen.** +25 poeng. Ingen annen endring kom i nærheten. Leksjonen: tenk på hva som er DELT mellom instansene.

**ABC param estimation var dead code lenge.** Vi implementerte den tidlig, men den var buggy og ga null effekt. Da vi endelig fikset den og koblet den til MC-sim som pseudo-counts, ga den +0.5-1.0 på vanskelige runder.

**Expansion cap fix (0.01→0.05) — liten endring, stor effekt.** Simulatoren hadde en altfor lav cap på bosettingsekspansjon. Å øke den fra 0.01 til 0.05 matchet observerte rater mye bedre.

**Experiment-to-solver gap.** Flere eksperimenter viste +0.5 i isolert test men REGREDIERTE i full solver. Rotårsak: eksperimenter brukte litt annen baseline. Lærdom: alltid verifiser i full pipeline.

**Automatisering sparte enormt med tid.** 20 plasser opp på leaderboard bare ved å submitte konsistent. Mange konkurrenter misset runder — vi var der hver gang.

## Scores

| Runde | Karakter | Score | Merknad |
|-------|----------|-------|---------|
| R1 | Stable | 16.5 | Tidlig versjon, ingen cross-seed |
| R2 | High expansion | 71.3 | Fortsatt tidlig |
| R3 | Brutal outlier | 22.7 | 1.8% survival, ekskludert fra historikk |
| R4 | Moderate death | 75.6 | |
| R5 | Brutal winter | 55.4 | |
| R6 | Stable | 48.8 | |
| R7 | Stable+expansion | 63.2 | Første Dirichlet-versjon |
| R8 | Ultra-brutal | 87.5 | Cross-seed introdusert |
| R9 | Ultra-brutal | 92.0 | |
| R10 | Ultra-brutal | 91.6 | 5-deep query strategy |
| R11 | Very stable | 89.6 | |
| R12 | Stable+death | 71.0 | Vanskeligste "stabile" runden |
| R13 | Ultra-brutal | 91.8 | |
| R14 | Stable+high F→S | 85.8 | |
| R15 | Ultra-brutal | 91.6 | |
| R16 | Brutal | ~91 | Automatisert pipeline |
| R17 | Very stable | ~91 | |
| R18 | Stable+expansion | ~91 | Høyeste F→S (43%) |
| R19 | Ultra-brutal | ~92 | |
| R20 | Ultra-brutal | ~92 | |
| R21 | Ultra-brutal | ~92 | |
| R22 | — | ~91 | Siste runde |

**Snitt R8–R22: ~91.** Tidlige runder (R1–R7) drar snittet ned fordi modellen var uferdig.

## Hva vi ville gjort annerledes

**Startet med cross-seed mye tidligere.** De første 7 rundene ble brukt på å finpusse historisk prior og feature engineering — cross-seed-innsikten kom først i R8 og ga +25 poeng umiddelbart. Hadde vi startet der, ville R1–R7 scoret 20-30 poeng høyere.

**Mer fokus på observation strategy fra dag 1.** Vi brukte 2-deep/3-shallow lenge før vi oppdaget at 5-deep (alle seeds med full grid) var bedre. Hvert seed gir unik cross-seed-data — bredde > dybde.

**Bygd simulator-matching tidligere.** ABC-estimering var implementert tidlig men fungerte ikke ordentlig før R15+. En korrekt simulator fra starten ville gitt bedre priors gjennom hele konkurransen.

**Ikke brukt så mye tid på parametersweeps.** 11 eksperimenter på å tune concentration/cross_scale/temperature ga tilsammen <0.5 poeng. Den tiden hadde vært bedre brukt på å forstå feature-key-barrieren.

---

*Totalt ~60+ eksperimenter, 21 runder med ground truth, og en pipeline som gikk fra 16.5 til 92+ poeng. Det var en god konkurranse.*
