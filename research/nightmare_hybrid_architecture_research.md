# Research: Hybrid Pre-computed + Reactive Architecture for Multi-Agent Coordination

> Researched: 2026-03-17 | Sources consulted: 18 | Confidence: High

## TL;DR

The optimal architecture for 20 agents with known task sequence is a **Rolling-Horizon Hybrid**: pre-compute the full game offline using LNS-based optimization, replay the plan live, and fall back to PIBT-based reactive planning on divergence. The 2023 MAPF Competition winner (WPPL) and 2020 Flatland Challenge winner both used exactly this pattern: fast initial solution (PIBT/Prioritized Planning) + iterative LNS refinement + windowed replanning. For your 2-second constraint, all computation should happen offline -- the live bot simply replays pre-computed actions with divergence detection.

## Key Findings

### 1. The Three Paradigms Compared

| Approach | Strengths | Weaknesses | Best for |
|----------|-----------|------------|----------|
| **Fully Pre-computed** | Optimal coordination, zero runtime cost | Brittle to any divergence, no recovery | Deterministic envs with perfect sim |
| **Fully Reactive (PIBT)** | Robust, adapts instantly, simple | Myopic (1-step lookahead), poor coordination, congestion | Unknown/dynamic environments |
| **Hybrid (Pre-compute + Reactive fallback)** | Best of both worlds, graceful degradation | Implementation complexity, needs good divergence detection | Known task sequences with sim-live gap |

**Verdict**: Hybrid is the clear winner for your scenario. You already have both pieces (MAPF replay + PIBT reactive) -- the gap is in the quality of the offline plan and the smoothness of the fallback transition.

### 2. What Competition Winners Actually Use

#### 2023 MAPF Competition Winner: WPPL (Windowed Parallel PIBT-LNS)
The winning solution combined three components:
- **PIBT** for fast initial solutions (~250ms for 10,000 agents)
- **MAPF-LNS** for iterative refinement within remaining time budget
- **RHCR** (Rolling-Horizon Collision Resolution) for windowed replanning

Key insight: They used **30-minute offline preprocessing per map** to compute guidance graphs (traffic flow patterns), then real-time planning was just PIBT following the pre-computed guidance. This is directly applicable -- your map is fixed per difficulty.

#### 2020 Flatland Challenge Winner: Team Old Driver
Used a two-phase approach:
- **Phase 1**: Initial solution generation with 200-second time limit (Prioritized Planning)
- **Phase 2**: Incremental LNS refinement during execution (50,000 iterations, 3s time limit)
- Five distinct neighbor selection strategies (random walk, intersection-based, adaptive)
- Outperformed all RL-based approaches among 700+ participants

#### Common Pattern Across Winners
1. Fast suboptimal initial plan (PIBT or Prioritized Planning)
2. Offline LNS refinement (maximize quality given time)
3. Online execution with monitoring
4. Selective replanning only when needed

### 3. Rolling-Horizon Collision Resolution (RHCR)

RHCR is the standard framework for combining offline planning with online execution:

```
Parameters:
  w = planning window (how far ahead to plan, in timesteps)
  h = replanning interval (how often to replan)
  Constraint: w >= h

Algorithm:
  1. Compute collision-free paths for next w timesteps
  2. Execute h timesteps
  3. Replan from current positions for next w timesteps
  4. Repeat
```

For your case with known order sequence and 2-second constraint:
- **w = 500** (plan the entire game offline)
- **h = 1** (check every round for divergence)
- Replanning = fall back to reactive PIBT when plan breaks

This is essentially what your `ReplayPlanner` already does, but with the plan quality being the bottleneck.

### 4. Offline Planning: Right Granularity

Given you know all orders in advance, the offline plan should be at **per-round granularity**:

| Granularity | Pros | Cons |
|-------------|------|------|
| Per-order | Simple, easy to reason about | Misses inter-order coordination |
| Per-item | Good balance | Still misses movement coordination |
| **Per-round (full schedule)** | Optimal coordination, handles bottlenecks | Requires good simulator, larger search space |
| Per-bot-action | Maximum detail | Fragile, hard to maintain |

**Recommendation**: Plan at per-round level. Each round specifies exact (x,y) position and action for all 20 bots. This is what your MAPF capture system already produces.

### 5. MAPF-LNS: The Key Missing Piece

Your current offline planner uses brute-force optimization. The state-of-the-art approach is **Large Neighborhood Search (LNS)**:

```
Algorithm: MAPF-LNS
1. Generate initial solution (e.g., prioritized planning or current reactive run)
2. Repeat until time limit:
   a. SELECT subset of agents (neighborhood)
   b. DESTROY their current paths
   c. REPAIR by replanning just those agents (A*, SIPP, etc.)
   d. ACCEPT if total cost improves (or with simulated annealing probability)
3. Return best solution found
```

Key LNS neighborhood selection strategies:
- **Random**: Select k random agents
- **Collision-based**: Select agents involved in most delays/conflicts
- **Agent-based**: Select agents with worst individual path costs
- **Intersection-based**: Select agents sharing congested vertices (e.g., x=1 corridor)
- **Adaptive**: Track which strategy has been most effective, weight accordingly

**Why this matters for you**: Your reactive BotAdapter run gives you an initial solution scoring 393. LNS can iteratively improve it by replanning subsets of bots while keeping others fixed. This is fundamentally different from trying to plan everything from scratch.

### 6. Traffic Flow Optimization & Guidance Graphs

The competition winner's secret weapon was **guidance graphs** -- pre-computed traffic flow patterns:

```
Standard PIBT heuristic: h(v) = manhattan_distance(v, goal)
Guidance heuristic:      h(v) = (distance_to_guide_path(v), remaining_on_guide_path(v))
```

How guidance graphs are computed:
1. For each agent, compute shortest path to goal (ignoring others)
2. Update edge costs: edges on computed paths become more expensive
3. Repeat for all agents -- this naturally creates traffic separation
4. The result is congestion-aware paths that avoid bidirectional flows

**For your nightmare map**: You already have one-way aisles. The guidance graph approach would formalize this by computing optimal traffic patterns for the x=1 approach corridor and the y=15/16 motorway.

### 7. Execution Monitoring & Divergence Detection

The state-of-the-art approach uses an **Action Dependency Graph (ADG)**:

```
ADG Components:
- Type 1 edges (intra-agent): sequential actions for one bot
- Type 2 edges (inter-agent): timing dependencies between bots

Divergence metric: slack
  slack(edge) = actual_completion_time - planned_completion_time
  If slack > threshold -> trigger replanning

Performance: 27% mitigation of delay impact vs 7% for random replanning
```

**Simplified version for your system**:
```python
def detect_divergence(planned_state, actual_state):
    """Compare planned vs actual bot positions each round."""
    diverged_bots = []
    for bot_id in range(20):
        planned_pos = plan[current_round][bot_id]["pos"]
        actual_pos = actual_state.bots[bot_id].position
        if planned_pos != actual_pos:
            diverged_bots.append(bot_id)

    # Strategy options:
    if len(diverged_bots) == 0:
        return "continue_plan"
    elif len(diverged_bots) <= 3:
        return "replan_diverged_only"  # Fix only broken bots
    else:
        return "full_reactive_fallback"
```

### 8. Partial Plan Execution (Hybrid Bots)

A powerful technique from the competition: **some bots follow the plan while others react**:

```
Classification per round:
- ON_PLAN: bot at expected position -> follow pre-computed action
- MINOR_DIVERGE: bot 1-2 cells off -> pathfind back to plan
- MAJOR_DIVERGE: bot far from plan -> switch to reactive PIBT

Benefits:
- Plan stays valid for most bots even when a few diverge
- Reactive bots use PIBT with plan-following bots as "moving obstacles"
- No need for all-or-nothing plan/reactive switching
```

### 9. Implementation Within 2-Second Constraint

| Phase | Timing | What Happens |
|-------|--------|--------------|
| **Offline** (pre-game) | Unlimited | Full LNS optimization of 500-round plan |
| **Round 1** | 2s | Load plan, execute first action |
| **Rounds 2-500** | 2s each | Compare positions, execute plan or fallback |

The 2-second constraint is irrelevant for the pre-computed approach -- all heavy computation happens offline. The live bot only needs to:
1. Read current game state (~1ms)
2. Compare 20 bot positions to plan (~1ms)
3. Look up pre-computed action OR run reactive PIBT (~50ms worst case)

**If replanning is needed online**: PIBT for 20 agents on a 30x18 grid takes <10ms. Even with LNS refinement, you have 1900ms of budget remaining.

### 10. Concrete Architecture Recommendation

```
OFFLINE (before game):
  1. Run reactive BotAdapter against recon -> initial solution (score ~393)
  2. Run LNS optimization:
     a. Select 3-5 bots (neighborhood)
     b. Replan their paths using time-space A* with reservation table
     c. Re-simulate full game
     d. Accept if score improves
     e. Repeat 1000+ times
  3. Save best plan as MAPF JSON

ONLINE (during game):
  Round N:
    1. Parse game state
    2. For each bot:
       if bot.position == plan[N].position:
         action = plan[N].action          # Follow plan
       elif distance(bot, plan[N].position) <= 2:
         action = pathfind_to(plan[N+1].position)  # Recover to plan
       else:
         action = reactive_pibt(bot)      # Full fallback
    3. Return actions
```

## Comparison: Approaches for Your Specific Problem

| Criteria | Current Reactive | Full Pre-compute | Hybrid (Recommended) |
|----------|-----------------|------------------|---------------------|
| Score potential | ~393 | ~326 (current sim) | 500-600+ (with LNS) |
| Robustness | Excellent | Fragile | Good (graceful fallback) |
| Runtime cost | ~50ms/round | ~1ms/round | ~5ms/round |
| Coordination quality | Poor (myopic) | Excellent | Excellent |
| Drop-off bottleneck | Reactive scheduling | Pre-computed queuing | Pre-computed queuing |
| Implementation effort | Done | Need better planner | Need LNS + divergence monitor |

## Gotchas & Considerations

- **Sim-live gap**: Your simulator's collision resolution may differ slightly from the server. The ADG approach handles this by being robust to small timing differences. Per-bot divergence detection (not all-or-nothing) is critical.
- **Daily order changes**: Orders change at midnight UTC. You need fresh recon + re-optimization each day. The offline LNS step should be fast enough to run in minutes.
- **Plan fragility at bottleneck**: The x=1 corridor is the single biggest source of divergence. Consider planning extra slack (1-2 round buffer) in the drop-off queue.
- **Auto-delivery timing**: When order N completes, the delivering bot's preview items auto-deliver. The offline plan must account for this exactly, or divergence cascades.
- **Spawn stacking**: All 20 bots start at (28,16). The first 5-10 rounds of the plan are the most fragile -- if any bot moves differently, all downstream positions shift. Consider making the first 10 rounds reactive, then switching to plan.
- **LNS neighborhood size**: For 20 agents, neighborhoods of 3-5 agents work well. Larger neighborhoods give better improvements but take longer per iteration.
- **Reservation table size**: 20 agents x 500 rounds x 30x18 grid = manageable. Time-space A* with reservation table is feasible for your scale.

## Recommended Implementation Plan

### Phase 1: LNS Optimizer (High Impact)
1. Use BotAdapter reactive run as initial solution
2. Implement LNS: select k bots, replan their routes, re-simulate
3. Use your existing `Simulering/offline/simulator.py` for evaluation
4. Target: 1000+ LNS iterations, neighborhood size 3-5

### Phase 2: Robust Replay (Medium Impact)
1. Per-bot divergence detection (not all-or-nothing)
2. Three-tier response: on-plan / recover / reactive
3. Plan-following bots treated as reserved positions for reactive bots
4. Late-start plan: first 5-10 rounds reactive, plan starts after bots spread

### Phase 3: Guidance Graphs (Lower Priority)
1. Pre-compute congestion-aware heuristics for PIBT
2. Use traffic flow data from recon runs to weight edges
3. Formalize one-way aisle patterns as guidance graph

## Sources
1. [Scalable MAPF with Collision-Aware Dynamic Alert Mask and Hybrid Execution](https://arxiv.org/html/2510.09469v1) -- Hybrid centralized/decentralized execution strategy with 4-stage pipeline
2. [Holistic Architecture for Monitoring Robust MAPF Plan Execution](https://arxiv.org/html/2509.10284) -- ADG-based divergence detection, 27% delay mitigation, slack-based replanning triggers
3. [Scaling Lifelong MAPF to More Realistic Settings](https://arxiv.org/html/2404.16162v1) -- WPPL competition winner details, RHCR framework, guidance graphs, warehouse-scale benchmarks
4. [Traffic Flow Optimization for Lifelong MAPF](https://arxiv.org/html/2308.11234v4) -- Congestion-aware path computation, guidance heuristics, one-way traffic pattern optimization
5. [Real-Time LaCAM](https://arxiv.org/html/2504.06091v1) -- Millisecond-budget planning, incremental DFS, tree rerooting for continuous planning
6. [RHCR: Rolling-Horizon Collision Resolution](https://github.com/Jiaoyang-Li/RHCR) -- Windowed MAPF framework, parameters w/h, compatible with WHCA/ECBS/PBS
7. [MAPF-LNS: Anytime MAPF via Large Neighborhood Search](https://github.com/Jiaoyang-Li/MAPF-LNS) -- LNS optimization framework, neighborhood selection strategies
8. [MAPF-LNS2: Fast Repairing](https://github.com/Jiaoyang-Li/MAPF-LNS2) -- Starting from collision-containing paths, iteratively repair
9. [Flatland Challenge Winner: Team Old Driver](https://github.com/Jiaoyang-Li/Flatland) -- Two-phase approach, PP + LNS, 5 neighbor strategies, outperformed all RL approaches
10. [winPIBT: Extended Prioritized Algorithm](https://arxiv.org/abs/1905.10149) -- Windowed PIBT, multi-step lookahead, livelock mitigation
11. [PIBT2: Priority Inheritance with Backtracking](https://kei18.github.io/pibt2/) -- Core of 2023 MAPF competition winning strategy
12. [LNS2+RL: Combining MARL with LNS](https://arxiv.org/abs/2405.17794) -- Hybrid RL+search, 50%+ success rate on complex maps where CBS/PRIMAL fail
13. [Engineering LaCAM*](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1501.pdf) -- Near-optimal MAPF at scale, engineering optimizations for real-time
14. [Benchmarking LNS for MAPF](https://arxiv.org/abs/2407.09451) -- Rule-based heuristics remain strong baselines; ML methods show no clear advantage
15. [MAPF Competition Code Archive](https://github.com/MAPF-Competition/Code-Archive) -- Open-source solutions from 2023 competition
16. [Guidance Graph Optimization for Lifelong MAPF](https://www.ijcai.org/proceedings/2024/0035.pdf) -- Edge direction + weight optimization for traffic management
17. [Budget Allocation Policies for Real-Time MAPF](https://arxiv.org/html/2507.16874v2) -- Intelligent budget distribution among agents for windowed LNS2
18. [Robust MAPF Plan Execution in Warehouses](https://ieeexplore.ieee.org/document/8620328/) -- Persistent execution with ADG-based safety guarantees
