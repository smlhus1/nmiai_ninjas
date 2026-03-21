# Research: Drop-off Bottleneck & Traffic Flow Optimization for 20 Bots on Nightmare Map

> Researched: 2026-03-17 | Sources consulted: 18 | Confidence: High

## TL;DR

The single drop-off at (1,16) is mathematically the binding constraint: at max throughput of 1 delivery/round, 300 rounds caps you at ~300 item-deliveries (plus order bonuses). The biggest wins come from (1) **guidance graph edge-weight optimization** to reduce contraflow congestion by 18-40%, (2) **IDLE bot parking in dead-end shelf bays** far from delivery lanes, and (3) **pipelined delivery queuing** where the next deliverer is always 1-2 steps from drop-off when the current one finishes.

## Key Findings

### 1. Traffic Flow Theory for Grid-Based Multi-Agent Systems

#### Counter-Clockwise Flow: Is It Optimal?

Research on lifelong MAPF traffic flow optimization (AAAI 2024) shows that **manually designed circulation patterns (like counter-clockwise) are typically 8-18% worse than automatically optimized guidance graphs**. The paper by Zhang et al. (IJCAI 2024) demonstrated that CMA-ES optimization of edge weights outperformed human-designed "crisscross" patterns:

- CMA-ES throughput: 6.58 goals/timestep vs crisscross baseline: 5.59 goals/timestep (+18%)
- PIU (neural update model): scales to 3,000 agents on 93x91 maps

**Practical implication**: Your current counter-clockwise (y=16 LEFT, verticals alternate DOWN/UP) is a good starting point, but the optimal flow pattern likely differs. The key insight is that **contraflow (opposing traffic on same edge) is the primary throughput killer**, not the direction itself.

#### Multi-Lane vs Single-Lane

The nightmare map has single-lane vertical aisles (x=1,4,8,12,16,20,24,27,28) and the y=15/y=16 horizontal corridor pair. Research shows:

- **Single-lane aisles with one-way rules**: effective for preventing head-on deadlocks
- **Two-lane corridors (y=15/y=16)**: the keep-right pattern is standard but your current setup (y=16 LEFT only, y=15 free) creates an asymmetry that may cause contraflow on y=15

**Recommendation**: Make y=15 RIGHT-only. The Traffic Flow Optimization paper specifically found that **contraflow cost = f(A->B) x f(B->A)** — even small bidirectional flow on y=15 creates quadratic congestion penalty.

#### Deadlock Prevention in One-Way Systems

PIBT guarantees deadlock-freedom when the graph satisfies a **biconnectedness** condition (every pair of adjacent nodes belongs to a simple cycle). Your one-way system must preserve this — every cell must have at least one valid exit route. The current setup where y=15 is bidirectional satisfies this, but making y=15 RIGHT-only requires that cross-corridor intersections remain bidirectional (which they naturally are since one-way rules only apply to non-intersection cells).

### 2. Drop-off Bottleneck Solutions

#### Queuing Theory Analysis

The drop-off is an **M/D/1 queue** (Markov arrivals, Deterministic service time of 1 round, 1 server):

- **Service rate (mu)**: 1 delivery/round (1 bot drops off per round)
- **Arrival rate (lambda)**: Must be < 1 to avoid infinite queue growth
- **Optimal utilization**: In M/D/1, avg wait time = lambda / (2 * mu * (mu - lambda))
- At lambda = 0.8: avg wait = 2 rounds. At lambda = 0.95: avg wait = 9.5 rounds.

**Key insight**: With 20 bots but only 1 delivery slot per round, you want lambda ~= 0.7-0.8 (deliver every 1.2-1.4 rounds). Higher than that and queue wait times explode. Your `max_deliverers=3` config is the right idea — it controls the arrival rate.

#### Should Bots Queue In Order?

**Yes, strict distance-based FIFO queuing is optimal for single-server systems.** Research on multi-server queues (CMU, Grosof 2023) shows that for single-server deterministic-service queues, FIFO minimizes average wait time. Your current approach (sort by completes_order, matching_count, distance) is correct — the priority should be:

1. Bot that completes the order (triggers +5 bonus + auto-delivery cascade)
2. Bot with most matching items (higher throughput per delivery round)
3. Closest bot (minimizes service gap)

#### Delivery Pipeline Design

The ideal pattern is a **relay pipeline**:

```
Round N:   Bot A at (1,16) doing DROP_OFF
           Bot B at (1,15) waiting (next in line)
           Bot C at (1,14) approaching (2 rounds out)
           Bots D-F: picking items, not heading to drop-off yet

Round N+1: Bot A escapes to (2,16) or (1,15)->RIGHT
           Bot B moves to (1,16) for DROP_OFF
           Bot C moves to (1,15)
```

**Critical**: The gap between consecutive deliveries should be exactly 1 round (Bot B enters as Bot A exits). This requires:
- The ESCAPE urgency (-1) on Bot A works correctly to move it out
- Bot B is pre-positioned at distance 1 from drop-off
- At most 1-2 bots staging near drop-off, others stay in pickup zone

#### Practical Implementation: Delivery Lane

Create a dedicated **approach corridor** on x=1 going DOWN from y=14 to y=16:

```
(1,14) -> (1,15) -> (1,16) [DROP_OFF]
```

Rules:
- Only bots with DELIVER task and matching items may enter x=1, y<=15
- IDLE/PICK_UP bots must use x=2+ for north-south movement
- After delivery, bot escapes RIGHT along y=16 (already LEFT-only for others, but the delivering bot gets a one-time exemption or uses y=15 RIGHT)

### 3. Highway Systems for Grid-Based Warehouses

#### What Real Warehouse Systems Use

Amazon's warehouse robots use a **city-grid model**: robots travel north-south or east-west on virtual streets, with traffic rules similar to a road network. Key patterns:

1. **Dedicated pick aisles** (single-direction vertical)
2. **Cross-aisles** (bidirectional but with right-of-way rules)
3. **Express lanes** near delivery stations (restricted access)
4. **Buffer zones** for queuing near workstations

#### Applying to 30x18 Nightmare Map

```
Map structure (simplified):
x:  0  1  2  3  4  ...  24  27  28  29
y0: W  .  .  W  .       .   .   .   W
y1: W  .  .  W  .       .   .   .   W    <- cross-corridor
    ...shelves...
y9: W  .  .  W  .       .   .   .   W    <- cross-corridor
    ...shelves...
y15:W  .  .  W  .       .   .   .   W    <- cross-corridor (return lane)
y16:W  D  .  W  .       .   .   S   W    <- cross-corridor (delivery lane)
y17:W  W  W  W  W       W   W   W   W
```

Recommended highway system:
- **y=16**: LEFT only (delivery approach) -- ALREADY DONE
- **y=15**: RIGHT only (return after delivery) -- CURRENTLY FREE, SHOULD BE ONE-WAY
- **y=1**: LEFT (general circulation toward drop-off side)
- **y=9**: RIGHT (general circulation toward spawn side)
- **x=1**: DOWN only from y=1 to y=16 (delivery approach column)
- Other verticals: alternate DOWN/UP -- ALREADY DONE

#### Express Lane for Deliverers

The **hivemind_clear_path** function already does this reactively. Research suggests a **proactive** approach is better:

- Reserve x=1 column for DELIVER-task bots only
- IDLE and PICK_UP bots must not enter x<=2 unless their pickup target is there
- This prevents the 51% IDLE-bot-blocking problem at the bottleneck

### 4. IDLE Bot Management

#### The 51% Problem

51% of bottleneck waits are from IDLE bots blocking. This is the **single biggest improvement opportunity**. Research consistently shows:

> "When the agent density is high, it becomes necessary to optimize the paths not only for goal-assigned agents but also for those obstructing them." — MAPF-HD (2025)

#### Where Should IDLE Bots Park?

**Distributed parking in shelf bays, far from traffic lanes.** The ideal positions:

1. **Dead-end shelf adjacents** (cells with only 1-2 walkable neighbors)
2. **Upper map region** (y=2 to y=8) — far from delivery bottleneck
3. **NOT on cross-corridors** (y=1, y=9, y=15, y=16)
4. **NOT on vertical aisles** (x=1,4,8,12,16,20,24,27,28) unless at a dead-end

Your current `_find_spread_eviction` function does this partially, but only for bots near the drop-off. **All IDLE bots should be pre-assigned to parking positions**, not just evicted ones.

#### Parking Zone Algorithm

```python
def compute_parking_zones(grid, cross_rows, aisle_columns, drop_off):
    """Pre-compute ranked parking positions for IDLE bots."""
    parking = []
    for x in range(grid.width):
        for y in range(grid.height):
            if not grid.is_walkable((x, y)):
                continue
            if y in cross_rows:
                continue  # Never park on highways
            if x in aisle_columns and y not in cross_rows:
                continue  # Don't park in aisles (blocks traffic)

            # Score: prefer far from drop-off, in dead-ends
            walkable_neighbors = count_walkable_neighbors(grid, (x, y))
            d_to_dropoff = bfs_distance((x, y), drop_off)

            # Dead-ends (1 neighbor) > corners (2 neighbors) > open cells
            score = -walkable_neighbors * 100 + d_to_dropoff
            parking.append(((x, y), score))

    parking.sort(key=lambda p: -p[1])  # Best parking first
    return [p[0] for p in parking]
```

#### Should IDLE Bots Pre-Position for Future Tasks?

**Only if you know the next order's items** (which you do from recon). The Traffic Flow Optimization paper found that pre-positioning near expected pickup locations improves throughput, BUT:

- **For >=20 bots, future orders in demand scoring HURTS** (your CLAUDE.md notes this)
- Pre-positioning creates "gravitational pull" that causes clustering
- Better approach: distribute IDLE bots evenly across the upper map, they'll be within ~5-7 steps of any shelf

### 5. Congestion-Aware Routing

#### Dynamic vs Static Traffic Rules

Research strongly favors **hybrid approaches**:

- **Static rules**: one-way aisles, highway directions (prevent deadlocks, low overhead)
- **Dynamic guidance**: congestion-weighted edge costs (adapt to current traffic)

Your `GuidanceGraph` already implements the dynamic part with:
- Vertex congestion: `alpha * visits`
- Contraflow: `beta * forward * backward`
- Exponential decay: `0.7` per update cycle

The Traffic Flow Optimization paper uses the same structure but with a key difference:

**Their contraflow cost = f(v1,v2) x f(v2,v1)** — this is **multiplicative**, creating a strong quadratic penalty for bidirectional traffic. Your current implementation matches this.

#### Improvements to Current Guidance

1. **Update interval**: Your `guidance_update_interval=5` may be too slow. The LTM paper (2026) found that updating every round with decay works better for fast-changing scenarios.

2. **Edge weight normalization**: LTM normalizes weights to [0, 10] range, preventing runaway costs that make some routes appear unreachable.

3. **Hindrance metric** (PIBT preference paper, 2025): Add a one-step lookahead that checks "will my move block a neighboring agent?" This provides 10-20% cost reduction in dense scenarios with negligible runtime cost.

```python
# Hindrance: does moving to candidate block neighbor's best move?
def hindrance(candidate, neighbors_of_candidate, their_targets):
    blocked_count = 0
    for neighbor_id, neighbor_pos in neighbors_of_candidate:
        best_move_for_neighbor = min_distance_neighbor(neighbor_pos, their_targets[neighbor_id])
        if best_move_for_neighbor == candidate:
            blocked_count += 1
    return blocked_count
```

4. **Regret learning**: Track the gap between chosen and optimal actions over 3 PIBT iterations, then use weighted averaging (w=0.9) to improve future decisions. This gave **40%+ throughput improvement** in lifelong MAPF on similar-density maps.

#### Should Routes Change Based on Congestion?

**Yes, but only for PIBT candidate ordering, not for path planning.** The guidance graph should influence which neighbor PIBT prefers, but the actual path computation (A*/BFS distance) should remain static for cache validity. This is exactly what your current architecture does.

### 6. Spawn Stacking Solutions

#### Current Problem

20 bots at (28,16), one-way corridors. First 200 rounds = 5 orders, last 300 = 16 orders.

#### Optimal Dispersal Pattern

With y=16 LEFT-only and y=15 free, the fastest dispersal is:

```
Round 0: All 20 bots at (28,16)
Round 1: Bot 0 moves to (27,16) LEFT, Bots 1-19 stay (collision)
Round 2: Bot 0 -> (26,16), Bot 1 -> (27,16)
...
Round 19: Bot 0 at (9,16), Bot 19 at (27,16)  -- 19 rounds to spread on y=16
```

But this is SLOW — bots are linearly queued on y=16, not dispersing into the map.

**Better pattern**: Peel off into vertical aisles immediately:

```
Round 0: All at (28,16)
Round 1: Bot 0 -> (27,16)
Round 2: Bot 0 -> (27,15) UP via x=27, Bot 1 -> (27,16)
Round 3: Bot 0 continues UP, Bot 1 -> (27,15), Bot 2 -> (27,16)
...
```

Each aisle (x=27, x=24, x=20, x=16, x=12, x=8, x=4) absorbs 2-3 bots going UP. With 7 aisles and 2-3 bots each, full dispersal takes:

- **~8-10 rounds** for the first bot to reach y=1 from y=16 (15 steps)
- **~20 rounds** for all bots to reach their initial shelf areas
- **This matches your observed data**: "first 200 rounds = 5 orders" suggests dispersal takes ~40-60 rounds total

#### Optimization: Directed First Assignments

Instead of letting TaskPlanner assign randomly during dispersal:

1. **Round 0-3**: Assign bots to the **nearest active-order items** in reverse aisle order (rightmost aisles first, since bots start at x=28)
2. **Bots 0-2**: First items in x=24-28 aisles
3. **Bots 3-5**: Items in x=16-20 aisles
4. **Bots 6+**: Items further left

This creates a **wave dispersal** where bots naturally spread without fighting over the same items.

## Comparison: Improvement Techniques by Expected Impact

| Technique | Expected Impact | Implementation Effort | Risk |
|-----------|----------------|----------------------|------|
| IDLE bot parking zones (dead-ends) | +15-25 pts | 2-4 hours | Low |
| y=15 RIGHT-only (eliminate contraflow) | +5-15 pts | 30 min | Medium (test!) |
| Delivery pipeline relay (1-gap) | +10-20 pts | 4-6 hours | Medium |
| x=1 column reserved for deliverers | +5-10 pts | 1-2 hours | Low |
| Hindrance metric in PIBT | +10-20 pts | 2-3 hours | Low |
| Guidance update every round | +3-8 pts | 30 min | Low |
| Wave dispersal from spawn | +5-10 pts | 2-3 hours | Low |
| Full GGO (CMA-ES edge weights) | +15-30 pts | 1-2 days | High |

## Gotchas & Considerations

- **y=15 RIGHT-only risk**: Your experiment log shows "y=15 RIGHT: sim+30 live-10" — this FAILED in live. The issue was likely that bots couldn't reach certain shelves. Must ensure all shelf-adjacent cells remain reachable via the one-way graph. Test with BFS reachability check.

- **y=15 LEFT also failed**: "75->75 loop" — confirming that static y=15 rules cause routing loops. Consider making y=15 one-way RIGHT **only between x=1 and x=4** (delivery exit zone) and free elsewhere.

- **Parking zones can become traps**: If IDLE bots park in dead-ends and then need to pick up an item across the map, they waste 5-10 rounds traveling back. Compromise: park no deeper than 2 cells from an aisle.

- **Delivery pipeline race condition**: If Bot B arrives at (1,15) but Bot A hasn't finished DROP_OFF yet, Bot B blocks at (1,15) for a round. The ESCAPE urgency (-1) must fire BEFORE Bot B's move is resolved. Since game processes bots in ID order, low-ID deliverers should be prioritized.

- **Guidance graph cold start**: First 20-30 rounds have no congestion data. Consider seeding the guidance graph with expected traffic patterns from recon data.

- **Sim-live gap**: Your experiment log warns that "sim gives 118 but live gives 52" — any change must be validated in LIVE, not just sim. Sim-only improvements are unreliable for nightmare.

## Recommendations

### Priority 1: IDLE Bot Parking (Today)
1. Pre-compute parking zones at game start (dead-end shelf bays, y=2-8)
2. ALL IDLE bots get `navigation_override` to nearest unoccupied parking zone
3. Parking zones must not be on cross-corridors or vertical aisles
4. Keep parking 1-2 cells from an aisle for fast re-deployment

### Priority 2: Delivery Pipeline Relay (Today)
1. Limit active deliverers to 2 (not 3) — reduces queue at drop-off
2. Pre-stage next deliverer at (1,15) while current delivers at (1,16)
3. Ensure ESCAPE bot exits RIGHT (not UP) to avoid blocking the approach
4. After delivery, bot immediately gets new PICK_UP task (no IDLE gap)

### Priority 3: x=1 Delivery Column (Quick Win)
1. Only DELIVER bots may have targets on x=1, y=10-16
2. PIBT: if IDLE/PICK_UP bot is on x=1, give it maximum eviction urgency
3. This clears the delivery approach path proactively

### Priority 4: Hindrance Metric in PIBT (Tomorrow)
1. Add one-step lookahead: "will my move block a higher-priority neighbor?"
2. Use as secondary tiebreaker after BFS distance in PIBT candidate sort
3. O(4) per bot per round — negligible cost

### Priority 5: Guidance Graph Tuning (Tomorrow)
1. Reduce `guidance_update_interval` from 5 to 1-2
2. Normalize edge weights to [0, 10] range
3. Seed initial weights from recon traffic patterns if available
4. Consider increasing `guidance_alpha` to 3.0 (heavier vertex congestion penalty)

## Sources

1. [Traffic Flow Optimisation for Lifelong MAPF](https://arxiv.org/html/2308.11234v4) -- Core paper on guidance paths and congestion-avoiding routing for PIBT. AAAI 2024.
2. [Guidance Graph Optimization for Lifelong MAPF](https://arxiv.org/abs/2402.01446) -- CMA-ES and PIU for automatic edge weight optimization. IJCAI 2024.
3. [How Amazon Robots Navigate Congestion](https://www.amazon.science/latest-news/how-amazon-robots-navigate-congestion) -- City-grid model, predictive conflict avoidance.
4. [MAPF-HD: Multi-Agent Path Finding in High-Density Environments](https://arxiv.org/html/2509.06374) -- PHANS algorithm for evacuating obstructing agents.
5. [Lightweight Traffic Map for Efficient Anytime LaCAM*](https://arxiv.org/html/2603.07891) -- Real-time congestion tracking via committed/blocked action counting.
6. [Lightweight and Effective Preference Construction in PIBT](https://arxiv.org/html/2505.12623) -- Hindrance metric + regret learning. 40%+ throughput improvement.
7. [Congestion Mitigation Path Planning for Large-Scale Multi-Agent Navigation](https://arxiv.org/html/2508.05253v1) -- Multiplicative contraflow penalty C(v) = product of inflows - 1.
8. [Flow-Based Task Assignment for Multi-Agent Pickup and Delivery](https://arxiv.org/html/2508.05890) -- Flow-network approach to congestion-aware task assignment.
9. [PIBT: Priority Inheritance with Backtracking](https://www.sciencedirect.com/science/article/pii/S0004370222000923) -- Original PIBT paper, biconnectedness guarantees.
10. [Cooperative Hybrid Multi-Agent Pathfinding](https://arxiv.org/html/2503.22162v1) -- Switching between D* Lite and RL based on local density.
11. [Queueing Theory - Wikipedia](https://en.wikipedia.org/wiki/Queueing_theory) -- M/D/1 queue formulas for delivery throughput analysis.
12. [GitHub: ggo_public](https://github.com/lunjohnzhang/ggo_public) -- Reference implementation of Guidance Graph Optimization.
