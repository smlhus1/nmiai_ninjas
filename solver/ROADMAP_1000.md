# Roadmap to 1000+ Nightmare Score

## Current: 381 (converged)

Workflow: optimizer → capture sim → MAPF replay → 381 live consistently.
95% of available orders completed. Throughput: 0.76 score/round.

## Why 381 is the ceiling

- 39 orders in recon, max 401 score
- Complete 37/39 in 500 rounds (95%)
- Throughput bottleneck: 14 rounds/order average
- Top team (#1): 1316 = 2.63 score/round = 3.5x our speed

## What 1000+ requires

1300 score = ~130 orders in 500 rounds = 2.6 orders/round
Currently: 0.074 orders/round. Need 35x improvement? No.
Actually: 130 orders * 5.3 items/order = 689 items. 689 items / 500 rounds = 1.38 items/round.
With 20 bots: 0.069 items/bot/round. Each item takes ~24 steps round-trip.
Need: 500/24 = 20.8 items per bot in 500 rounds = feasible!

The math says 1000+ IS possible with 20 bots if:
- Zero wasted rounds (no idle, no spawn delay)
- Average trip ~24 steps (current ~24, OK)
- Inventory cap 3 → batch 3 items per trip → 8 trips per bot
- 20 bots * 8 trips * 3 items = 480 items in ideal case

So ceiling is ~480 items + ~96 orders * 5 = 960. Hmm, still under 1000.

With tighter trips (avg 15 steps via zone affinity):
- 500/15 = 33 trips per bot = 660 items
- 660 items / 5.3 = 124 orders → 660 + 124*5 = 1280

## Architecture required

### Option A: Coordinated MAPF (recommended)
- Pre-compute collision-free paths for all 20 bots
- Time-space A* with reservation table
- Windowed planning (50-step windows, re-plan periodically)
- Eliminates ALL collision delays
- Estimated: 600-800 score (2-3x current)

### Option B: Improved reactive with zone lock
- Hard zone assignment (bots NEVER leave their zone)
- 3 zones × 6-7 bots each
- Zone-local item assignment
- Reduces cross-traffic, shorter trips
- Estimated: 450-550 score

### Option C: Hybrid (best of both)
- Pre-computed plan for first 250 rounds (known orders)
- Reactive fallback for rounds 251-500 (unknown orders)
- Capture more orders each iteration
- Estimated: 500-700 score

## Implementation priority
1. Option C hybrid is most practical — extends current MAPF replay
2. Key: run sim for 2000 rounds to generate extended order sequence
3. Plan first 250 rounds with deterministic MAPF
4. Reactive fallback handles sim-generated order mismatches
