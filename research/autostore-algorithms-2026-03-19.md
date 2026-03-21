# Research: AutoStore Warehouse Robot Algorithms

> Researched: 2026-03-19 | Sources consulted: 15+ | Confidence: Medium-High (proprietary details are guarded)

## TL;DR

AutoStore's grid robots use centralized real-time pathfinding (recalculated every second), natural slotting for bin retrieval (high-demand bins float to top), and closest-task assignment. The academic MAPF community has extensively studied this problem class. The most relevant finding for NM i AI: **PIBT + LNS refinement (WPPL) is the state-of-the-art for large-scale warehouse MAPF**, and **guidance graphs** (precomputed directional preferences on the grid) provide up to 4x throughput improvement. AutoStore's proprietary "Router+" likely implements similar traffic flow optimization.

## Key Findings

### 1. Grid Navigation & Pathfinding

AutoStore robots move on an X/Y rail grid on top of the storage structure. The Router software is the centralized "brain" that:

- **Recalculates all robot paths every second** in real time
- Solves collision avoidance + path optimization as a single problem
- Adapts dynamically to cancellations, priority changes, new orders
- Can handle high robot density without throughput degradation

**What algorithm?** AutoStore doesn't disclose specifics. Based on the problem structure (centralized, real-time, grid, 100s of robots), they almost certainly use something in the **WHCA\* / PIBT / reservation table** family. Academic evidence strongly suggests:

| Algorithm | Agents | Speed | Quality | Best For |
|-----------|--------|-------|---------|----------|
| **CBS** (Conflict-Based Search) | <100 | Slow | Optimal | Small instances |
| **ECBS** (Enhanced CBS) | <500 | Medium | Bounded suboptimal | Medium instances |
| **PIBT** | 10,000+ | <250ms | Low-medium | Fast initial solution |
| **WPPL** (PIBT + LNS) | 10,000+ | Configurable | Near-optimal | **Production systems** |
| **LaCAM\*** | 10,000+ | Fast | Good | Recent competitor to WPPL |

**WPPL** (the winning approach in the 2023 MAPF competition) combines PIBT for fast initial paths with Large Neighborhood Search (LNS) for iterative improvement. It used window size w=10 with replanning every h=3 steps on warehouse maps.

**Key insight for our competition:** PIBT alone gives fast but low-quality solutions. The real win is **guidance graphs** — precomputed edge weights that encode preferred traffic flow directions. On the Sortation map (54,320 vertices, 10,000 agents), guidance graphs increased throughput from **10.19 to 39.34 goals/step** (nearly 4x).

### 2. Bin Retrieval Optimization (Digging)

AutoStore bins are stacked 16 deep. Retrieving a bottom bin requires removing all bins above it ("digging"). Their optimization:

**Natural Slotting Algorithm:**
- Every bin returned to the grid goes to a **random top-layer position**
- High-demand bins naturally float to the top (retrieved often = returned to top often)
- Low-demand bins sink to the bottom over time
- Result: 80/20 Pareto distribution emerges automatically

**Performance metrics (from 350 real systems):**
- 39% of bins need zero digging (already on top)
- Average depth: only 2.5 cells (in a 16-deep grid)
- Average retrieval: 27 seconds
- Worst case: 3 min 36 sec (bottom bin)
- Robots spend 80% serving, 20% digging

**Academic optimization — Layer Complete Policy (LCP):**
A paper (arXiv:2312.05338) formalized this as an optimization problem with cost function:
```
Cr(l, he) = l^2 + l - he^2 - he + T(he, l)
```
Where l = depth and he = empty cells above target. Their LCP policy achieves:
- 50%+ bins at grid surface
- 30% retrieval time reduction vs. random
- Nearly 50% fewer slow retrievals

**Relevance to NM i AI:** Limited directly (we don't have stacked bins), but the principle of **natural prioritization** is relevant — frequently needed items should be pre-positioned near optimal locations. This is essentially what our pre-pick strategy does.

### 3. Task Assignment

AutoStore uses simple **closest-task assignment**: the Controller assigns each robot to the nearest required bin. Key details:

- Robots are NOT zone-locked — entire fleet works across entire grid
- WMS sends order priorities; Controller assigns robots to bins
- "Bin preparation" proactively moves anticipated bins to top during idle time
- No single point of failure — any robot can do any task

**Academic alternatives studied:**
- **Hungarian algorithm** — optimal assignment for cost matrix (what we use)
- **Contract Net Protocol** — auction-based distributed assignment
- **Deep RL (EDRL-OBOS)** — learned heuristic rules for batch order scheduling
- **Genetic algorithms** — for large-scale task-to-robot optimization

**Key finding:** For warehouse MAPF, the task assigner matters less than the pathfinder. The 2023 competition showed that even naive task assignment + good MAPF (WPPL) beats sophisticated assignment + poor MAPF.

### 4. Traffic Management & Congestion

AutoStore's Router+ (released ~2020) claims 15-20% efficiency improvement through "advanced traffic management algorithms." Based on available information:

**What they do:**
- Recalculate all paths every second (global replanning)
- Anticipate movements to prevent jams before they form
- Handle 8-12 robots per picking port as optimal ratio
- Square grids more efficient than rectangular (shorter average distances)

**What the academic literature says works for grid congestion:**

1. **Guidance graphs / traffic flow maps** — Precomputed preferred directions on each edge. Think of it as encoding one-way streets. Provides the single largest throughput improvement in warehouse MAPF (up to 4x).

2. **Agent disabling** — Temporarily "parking" agents by setting their goal to current position with lowest priority. On 97.7% density grids, this improved throughput by **51.8%**.

3. **Rotation cost modeling** — Requiring agents to face a direction before moving (kinematic constraints) massively increases congestion. Four-way instantaneous movement is far more efficient.

4. **Window-based replanning** — Planning only w steps ahead (w=10 typical), replanning every h steps (h=3 typical). Balances computation vs. solution quality.

**Critical insight for our competition:** "Even though [the algorithm] improves its approximated sum-of-costs objective in earlier MAPF instances, movement might cause more severe congestion" — i.e., locally optimal paths can create globally worse congestion. This is exactly the problem we face at the nightmare drop-off bottleneck.

### 5. Patents & Proprietary Details

AutoStore Technology AS holds extensive patents (100+), primarily covering:
- Physical grid/rail structure and robot hardware
- Bin lifting mechanisms
- Battery swap systems (BattPack)
- Controller communication protocols

**No patents found** that disclose specific routing or pathfinding algorithms in detail. This is typical — software algorithms are usually protected as trade secrets, not patents.

## Actionable Insights for NM i AI

### What we should steal:

1. **Guidance graphs** — We already have one-way aisles in nightmare. The research shows that precomputed traffic flow maps are THE single biggest throughput multiplier. Our current one-way system is a simple version of this. Could we optimize the flow direction weights further?

2. **Agent disabling / parking** — When bots have nothing useful to do, park them out of the way with lowest priority. We partially do this with IDLE, but could be more aggressive about removing bots from congested corridors.

3. **Natural slotting principle** — Pre-position bots near where they'll be needed next. Our pre-pick strategy already does this implicitly.

4. **Global replanning frequency** — AutoStore replans every second. We replan every round (effectively every step). This is already good.

5. **WPPL architecture** — PIBT for collision avoidance (we have this) + LNS for route optimization (we have this in C++). The key we might be missing: **guidance graph integration into PIBT** to bias movement toward preferred traffic directions.

### What's different from our problem:

| AutoStore | NM i AI |
|-----------|---------|
| Robots retrieve bins from grid | Bots pick items from shelves |
| Multiple I/O ports around grid | Single drop-off point (bottleneck!) |
| No time pressure per task | 300 round limit, 2s response time |
| Hundreds of robots, large grids | 1-20 robots, small grids |
| Continuous operation | Fixed-length game |
| Bin digging (vertical access) | Direct item pickup (no digging) |

### The biggest gap we could close:

**Traffic flow optimization at the drop-off bottleneck.** AutoStore's Router+ focuses on preventing congestion at I/O ports (their equivalent of our drop-off). They use 8-12 robots per port as the optimal ratio. With 20 bots and 1 drop-off in nightmare, we're at 20:1 — far beyond optimal. The academic literature suggests **agent disabling** (parking idle bots) and **guidance graphs** (directing traffic flow) as the main solutions. We have one-way aisles but could:
- More aggressively park bots away from the drop-off corridor
- Use time-windowed access to the drop-off (token/scheduling system — but we tested this and it regressed)
- Optimize the guidance graph weights for the specific nightmare map

## Sources

1. [AutoStore Controller Software](https://www.autostoresystem.com/system/controller) — Router architecture, 1-second replanning
2. [AutoStore Bin Preparation / Natural Slotting](https://www.autostoresystem.com/insights/how-does-bin-preparation-maximize-throughput) — Digging metrics, 80/20 distribution
3. [AutoStore Overview & Features (BestOpsChain)](https://bestopschainai.com/warehouse-inventory/autostore-overview-features) — Router+, AI features, fleet coordination
4. [arXiv:2312.05338 — RCS/RS Bin Retrieval Optimization](https://arxiv.org/html/2312.05338v1) — Layer Complete Policy, cost functions, 30% improvement
5. [arXiv:1706.09347 — Path Planning for RMFS](https://arxiv.org/abs/1706.09347) — WHCA*, CBS, FAR, OD&ID comparison for warehouse MAPF
6. [Scaling Lifelong MAPF (arXiv:2404.16162)](https://arxiv.org/html/2404.16162v1) — WPPL, PIBT+LNS, guidance graphs (4x throughput), agent disabling (51.8% improvement)
7. [AutoStore Router Introduction (GlobeNewsWire)](https://www.globenewswire.com/news-release/2020/9/29/2100400/0/en/AutoStore-Introduces-Router-Game-changing-Productivity-Software-to-Solve-order-fulfillment-challenges-for-eCommerce.html) — Router launch, real-time path recalculation
8. [Kardex: How AutoStore Works](https://www.kardex.com/en-us/blog/how-autostore-works) — System architecture overview
9. [Lifelong MAPF in Large-Scale Warehouses (AAAI 2021)](https://cdn.aaai.org/ojs/17344/17344-13-20838-1-2-20210518.pdf) — WPPL competition winner
10. [MAPF-HD: High-Density Environments](https://www.researchgate.net/publication/395355365_MAPF-HD_Multi-Agent_Path_Finding_in_High-Density_Environments) — Dense grid MAPF techniques
