# 5 Whys: Hvorfor blir det congestion på nightmare?
**Dato:** 2026-03-16
**Prosjekt:** NM i AI Grocery Bot

## Symptom
9 av 32 ordrer tar 27r snitt (vs 9.5r for fast). Ordre 13 tar 43 runder.
x=4,12,16 aisles har 8-10 bots. Wait-rate 31.8% i R200-300.

## Analyse

### Nivå 1: Hvorfor tar ordre 13 hele 43 runder?
**Funn:** remaining=1 (bread) i **18 av 43 runder** (R219-R237).
Ordren bruker 18 runder på å levere SISTE item.
**Bevis:** Runde-for-runde trace: remaining synker fra 6→1 på 24r (ok),
men 1→0 tar 18r (katastrofalt).
**Confidence:** HØY

### Nivå 2: Hvorfor tar siste bread 18 runder?
**Funn:** B18 er den ENESTE boten som har bread. B18 er ved (19,1) —
toppen av kartet. Drop-off er (15,16). Ingen annen bot plukker bread.
B18 beveger seg 1 steg per runde UTEN EN ENESTE WAIT fra R218-R237.
Det er IKKE congestion — det er REN AVSTAND.
**Bevis:** B18 trace: (19,1)→(18,1)→...→(16,16)→(15,16) = 19 steg, 0 waits
**Confidence:** HØY

### Nivå 3: Hvorfor er B18 så langt unna med bread?
**Funn:** B18 plukket bread fra shelf ved (21,5) — toppen av kartet.
Round-trip (15,16)→(20,5)→(15,16) = 51 runder. B18 ble tildelt bread
ved R186 etter å ha levert andre items. Den var den eneste boten med
ledig inventory OG uten bread.
**Bevis:** B18 history: deliver@R185, pick bread@R186, arrive shelf@R206,
deliver bread@R237. 51 runder total cycle.
**Confidence:** HØY

### Nivå 4: Hvorfor er B18 den eneste som kan plukke bread?
**Funn:** Alle andre 19 bots har FULL inventory (3/3) og kan ikke plukke
nye items. Inventory er fylt med "diverse" items fra fill_pickup som
IKKE matcher current order. Bread var det siste remaining item, og
ingen annen bot hadde det i inventory.
**Bevis:** R220 inventory scan: 0/19 andre bots har bread. Alle er IDLE
med full inventory av andre typer.
**Confidence:** HØY

### Nivå 5: ROTÅRSAK
**Funn:** Det er IKKE congestion som forårsaker slow orders. Det er
**DISTANCE × SINGLE-BOT DEPENDENCY**:

1. Ordrer har items fra hyller spredt over HELE kartet (y=2 til y=16)
2. "Far items" (bread y=2, flour y=2) krever 40-50r round-trip
3. Når et "far item" er det SISTE remaining, venter alt på ÉN bot
4. Alle andre bots er fulle og kan ikke hjelpe
5. Congestion i aisles (wait-rate) er en BIEFFEKT av IDLE bots som
   sitter i korridorer mens de venter — ikke ÅRSAKEN til slow orders

**Bevis:** B18 har 0 waits i 19-steg reisen. Congestion-waits (31.8% i R200-300)
er fra de 19 IDLE bots som sitter i korridorer og blokkerer HVERANDRE — men
det påvirker ikke B18 som reiser fritt gjennom y=1 korridoren.
**Confidence:** HØY

## Rotårsak
**Beskrivelse:** Slow orders skyldes at siste remaining item tildeles en ENKEL
bot som er langt fra shelf (40-50r round-trip). 19 andre bots kan ikke hjelpe
fordi de har full inventory. Det som ser ut som "congestion" er faktisk
IDLE-bots som sitter i korridorer og venter på denne ene boten.

**Kategori:** `wrong-assumption` — antatt at slow orders skyldes congestion,
men det er distance × single-bot dependency.

**Confidence:** HØY

## Anbefalt fix

### Fix 1: Prioriter "far items" FØRST
Når en ordre aktiveres, tildel items med LENGST avstand til shelves FØRST.
Nære items leveres mens fjerne items er underveis. Siste item bør alltid
være det NÆRMESTE, ikke det fjerneste.

### Fix 2: Redundant pickup for siste item
Når remaining <= 2: tildel FLERE bots til å plukke de siste items fra
ULIKE shelves. Først-ankomne leverer, andre kansellerer.
Kost: 1-2 bots bruker noen runder "bortkastet" — men spare 10-20r.

### Fix 3: Hold 1 inventory-slot åpen
Ikke fill til 3/3. Hold 1 slot åpen slik at bots alltid KAN plukke
det siste remaining item. Med 2/3 inventory har 20 bots alltid
20 åpne slots tilgjengelig for emergency pickup.

### Beregnet effekt
Ordre 13 med Fix 1: bread tildeles tidlig (R195 istedenfor R219).
B18 starter reisen 24 runder TIDLIGERE → ordre 13 fullføres R213
istedenfor R237 (24r spart). Tilsvarende for andre slow orders.
Estimat: +50-100r spart totalt → +5-10 ordrer.

## Preventivt tiltak
- [ ] Test: "siste remaining item skal alltid være nearest-shelf item"
- [ ] Instrumenter per-item avstand for å fange future far-item dependencies
- [ ] CLAUDE.md: dokumenter at slow orders = distance, ikke congestion

## Nøkkeldata

### Ordre 13 tidslinje
| Runde | Remaining | Hendelse |
|-------|-----------|----------|
| R194  | 6         | Ordre aktiveres |
| R218  | 2 (bread) | 4 items levert |
| R219  | 1 (bread) | B18 er ved (18,1), 19 steg fra drop-off |
| R237  | 0         | B18 leverer bread. 43 runder total |

### B18 full reise
| Fase | Fra | Til | Runder | Waits |
|------|-----|-----|--------|-------|
| Deliver prev | - | (15,16) | R185 | 0 |
| Travel to shelf | (15,16) | (20,5) | R186-206 | 0 |
| Pick 2x bread | (20,5) | (20,5) | R206-207 | 0 |
| Travel to drop | (20,5) | (15,16) | R207-237 | 0 |
| **Total** | | | **51r** | **0** |

### Bread shelf distances from drop-offs
| Shelf | Drop-off (1,16) | (15,16) | (27,16) |
|-------|----------------|---------|---------|
| (3,2) | ~16r | ~25r | ~37r |
| (9,5) | ~19r | ~16r | ~28r |
| (15,2)| ~27r | ~14r | ~24r |
| (21,5)| ~31r | ~16r | ~17r |
