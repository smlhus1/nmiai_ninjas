# Research: Collision-Free Multi-Agent Path Finding (MAPF) for Grocery Bot Competition

> Researched: 2026-03-17 | Sources consulted: 18 | Confidence: High

## TL;DR

For 20 bots on a 30x18 grid with a single drop-off bottleneck and 2-second time constraint, **LaCAM with PIBT + hindrance/regret improvements** is the most practical path to 2-3x improvement. For the theoretical maximum, **offline pre-computation with RHCR-style windowed replanning** is ideal since order sequence is known. The single drop-off at (1,16) is the binding constraint -- no algorithm can overcome the physical throughput limit of ~1 delivery per 2 rounds through a single cell.

## Key Findings

### 1. Algorithm Landscape for 20 Agents on Small Grids

| Algorithm | Optimal? | Speed (20 agents) | Python? | Practical for 2s? |
|-----------|----------|-------------------|---------|-------------------|
| CBS | Yes | Too slow (seconds-minutes) | Yes (pip) | No |
| ECBS | Bounded suboptimal | Marginal | No | Risky |
| Priority-Based (PrP) | No | Fast (<100ms) | Easy to implement | Yes |
| PIBT | No | Very fast (<10ms) | Yes (pypibt) | Yes |
| LaCAM | Complete | Fast (<1s for 1000s) | Yes (pylacam) | Yes |
| MAPF-LNS2 | Anytime | Good (improves over time) | No (C++) | Yes with budget |
| RHCR | Depends on solver | Good | No (C++) | Yes |

**Verdict**: CBS is out -- exponential worst case with 20 agents in tight corridors. PIBT/LaCAM are the practical choices. Your existing PIBT implementation is actually the right foundation.

### 2. LaCAM: The Most Promising Upgrade Path

LaCAM (Lazy Constraints Addition search for MAPF) by Keisuke Okumura (AAAI 2023) is the breakthrough algorithm:

- **How it works**: Searches in *configuration space* (all agents' positions as one state). Uses PIBT as a sub-routine to generate successor configurations. Lazy constraint addition means it only adds constraints when conflicts are found.
- **Performance**: Solves instances with 10,000+ agents in under a second
- **Completeness**: Guaranteed to find a solution if one exists
- **LaCAM***: Anytime version that improves solution quality over time
- **Real-Time LaCAM**: Incremental version that maintains search tree across timesteps, works with millisecond per-iteration budgets

**Python implementations**:
- `pylacam` (https://github.com/Kei18/pylacam) -- uses `uv`, minimal implementation
- `py-lacam` (https://github.com/Kei18/py-lacam) -- uses Poetry

**Key insight**: LaCAM essentially wraps PIBT in a higher-level search that guarantees completeness. Since you already have PIBT, adding the LaCAM wrapper is the natural next step.

### 3. PIBT Improvements: Hindrance + Regret (40%+ Throughput Gain)

The most impactful near-term improvement (paper: May 2025, "Lightweight and Effective Preference Construction in PIBT"):

**Current PIBT preference**: `<dist(v, goal), random_tiebreak>`

**Enhanced preference**: `<dist(v, goal), hindrance, regret, random_tiebreak>`

- **Hindrance** (O(Delta) per action): Check if moving to vertex v blocks a neighboring agent's best move toward its goal. Avoid actions that hinder neighbors.
- **Regret** (learned over 3 PIBT runs): Run PIBT 3 times with exponential moving average. Regret captures how an action affects nearby agents' future options.

**Results reported**:
- 40%+ throughput improvement on 32x32 maps with 400 agents
- 10-20% solution cost reduction in dense scenarios
- 100% success rate even when agents = available vertices
- Sub-millisecond execution with thousands of agents
- **Orthogonal to guidance graphs** -- can combine both

**This is the lowest-hanging fruit.** Adding hindrance to your existing PIBT costs O(Delta) per action evaluation and could give 20-40% throughput improvement. The implementation is ~20 lines on top of existing PIBT.

### 4. Lifelong MAPF (LMAPF) -- This IS Your Problem

Your competition is a textbook LMAPF problem: agents continuously get new tasks (pick item, deliver to drop-off, repeat). Key approaches:

#### RHCR (Rolling-Horizon Collision Resolution)
- Plan only `w` timesteps ahead, replan every `h` timesteps
- Use any MAPF solver as the windowed planner (PIBT is simplest)
- Tested up to 1,000 agents in warehouse scenarios
- Parameters: `simulation_window=h`, `planning_window=w`
- **For your case**: h=5, w=15 would mean replan every 5 rounds, looking 15 ahead
- Implementation: C++ only (https://github.com/Jiaoyang-Li/RHCR)

#### League of Robot Runners 2023 Winner (Team Pikachu)
- Algorithm: **WPPL** (Windowed Parallel PIBT-LNS)
- Combines: PIBT (initial solution) + LNS (iterative improvement) + RHCR (windowed replanning)
- Key trick: **Guidance graph** with edge weights to steer agents away from congestion
- Key trick: **Disable congested agents** -- set their goal to current position and priority to lowest
- Won "Overall Best" and "Fast Mover" tracks
- Implementation: C++ (https://github.com/DiligentPanda/MAPF-LRR2023)

### 5. Traffic Flow Optimization / Guidance Graphs

Guidance graphs (AAAI 2024, IJCAI 2024) are highly relevant for your single-drop-off bottleneck:

- **Concept**: Assign directional weights to grid edges to create "traffic lanes"
- **Congestion cost**: `c_v = n*(n-1)/2` where n = agents using vertex
- **Contraflow cost**: `c_e = flow(A->B) * flow(B->A)` -- penalizes bidirectional traffic
- **Your one-way aisles are already a manual guidance graph** -- this validates your approach

**Key insight for (1,16) drop-off**: A guidance graph can create a "delivery queue" by making the approach corridor one-way with high costs for deviating. Essentially formalizing what your `_schedule_dropoff()` does.

Implementation: C++ (https://github.com/nobodyczcz/Guided-PIBT)

### 6. The Drop-Off Bottleneck: Physics Limit Analysis

With 1 drop-off at (1,16) and 1 delivery action per round:

- **Hard limit**: 1 delivery per round = 500 deliveries max in 500 rounds (if zero travel time)
- **Realistic limit**: With average 10-round travel time, ~33 deliveries per round window = ~165 total
- **Score ceiling**: Depends on items per delivery. If avg 3 items + 5 order bonus = ~8 pts per delivery trip
- **Key**: The bottleneck is NOT pathfinding -- it's the physical delivery throughput

**Strategies from literature**:
1. **Pipeline delivery**: Bot A delivers while Bot B approaches, Bot C picks -- maximize drop-off utilization
2. **Batch delivery**: Load as many matching items as possible before going to drop-off
3. **Disable idle bots**: Set congested/waiting bots to IDLE with lowest priority (Robot Runners winner trick)
4. **Staging area**: Queue bots near (but not at) drop-off

### 7. Pre-Computed Offline Plans (Your Deterministic Advantage)

Since order sequence is known, you have a massive advantage over standard LMAPF:

**Action Dependency Graphs (ADG)**:
- Pre-compute the complete plan offline
- ADG captures which actions must happen before others
- Execute online with robustness to timing variations
- Used in Flatland competition winner + warehouse systems

**Your current approach (capture + replay via MAPF plan) is already state-of-the-art for deterministic scenarios.** The question is how to compute better plans offline.

**Recommended offline planning approach**:
1. Use PIBT+hindrance to generate initial solution
2. Apply LNS to iteratively improve: select subset of agents, replan their paths
3. Run thousands of iterations, keep best solution
4. Export as timestep-by-timestep action plan
5. Replay live with fallback to reactive PIBT if divergence

### 8. Practical Python Libraries

| Library | Install | Agents | Notes |
|---------|---------|--------|-------|
| `cbs-mapf` | `pip install cbs-mapf` | <15 | Optimal but too slow for 20 agents. Last updated 2020. |
| `pypibt` | Clone + `uv sync` | 200+ | Minimal PIBT, good reference. ~100 lines core. |
| `pylacam` | Clone + `uv sync` | 10000+ | LaCAM* with simplified PIBT. Reference impl. |
| `MAPF-GPT` | Clone, needs PyTorch | 100+ | Transformer-based, trained on LaCAM trajectories. AAAI 2025. |

**Recommendation**: Don't use a library. Your existing PIBT implementation is already customized for your game rules (pickup, delivery, priorities). Instead, **add hindrance scoring** to your existing PIBT and **implement LNS on top** for offline improvement.

## Practical Recommendations (Priority Order for 2-Day Sprint)

### Day 1: Quick Wins (Expected: +50-100% throughput)

#### 1. Add Hindrance to PIBT (2-4 hours, expected +20-40%)
In your `PIBTResolver`, when evaluating candidate moves for an agent, add a hindrance check:

```python
def hindrance(agent_pos, candidate_pos, other_agents, grid):
    """Check if moving to candidate_pos blocks a neighbor's best move."""
    score = 0
    for neighbor in get_neighbors(candidate_pos, grid):
        if neighbor.agent and neighbor.agent != agent:
            # Does this neighbor want to move through candidate_pos?
            best_dist = bfs_distance(neighbor.pos, neighbor.goal)
            via_candidate = bfs_distance(candidate_pos, neighbor.goal)
            if via_candidate < best_dist:
                score += 1  # We're blocking this neighbor
    return score
```

Preference ordering becomes: `(distance_to_goal, hindrance, random_tiebreak)` -- lower hindrance is better.

#### 2. Disable Idle/Stuck Bots (1-2 hours, expected +10-20%)
From Robot Runners winner: bots with no useful task should have:
- Goal = current position (don't move)
- Priority = lowest (yield to everyone)
- This prevents "zombie shuffling" that creates unnecessary congestion

#### 3. Regret Learning for Offline Planning (2-3 hours, expected +10-15%)
Run PIBT 3 times per planning step. Track which actions led to better outcomes. Use exponential moving average to bias future tiebreaking. This is specifically powerful for your offline pre-computation where you can afford the 3x compute.

### Day 2: Structural Improvements (Expected: +30-50% on top)

#### 4. LNS Improvement Loop for Offline Plans (4-6 hours)
```python
def lns_improve(plan, agents, grid, iterations=1000):
    best_plan = plan
    best_score = evaluate(plan)
    for i in range(iterations):
        # Select 3-5 random agents
        subset = random.sample(agents, min(5, len(agents)))
        # Fix all other agents' paths, replan subset
        new_plan = replan_subset(best_plan, subset, grid)
        new_score = evaluate(new_plan)
        if new_score > best_score:
            best_plan = new_plan
            best_score = new_score
    return best_plan
```

Key: "destroy" paths for a random subset of agents, "repair" using PIBT with the other agents' paths as moving obstacles (reservation table, but only for the fixed agents).

#### 5. Delivery Pipeline Optimization (2-3 hours)
Ensure maximum drop-off utilization:
- At every timestep, exactly 1 bot should be at (1,16) performing DROP_OFF
- Next bot should be at (1,15) or (2,16) ready to move in
- Third bot approaching from further away
- Use PIBT priorities: DELIVER > approaching_delivery > PICK_UP

## Gotchas & Considerations

- **CBS is a trap**: Looks elegant, scales terribly. With 20 agents on a 30x18 grid with one-way corridors, CBS will timeout every time. Don't waste time on it.
- **LaCAM's Python impl is simplified**: The pylacam repo replaces PIBT with random action selection for clarity. Not production-ready. Use as reference only.
- **Reservation tables fill up**: You already discovered this. The fix is windowed planning (only reserve w timesteps ahead) or no reservation table at all (use PIBT's local collision avoidance).
- **LNS requires good initial solution**: Always start from a PIBT solution, then improve. Never start from scratch.
- **Hindrance is cheap but powerful**: O(Delta) per action = O(4) on a grid. Trivial compute cost, proven 40%+ improvement.
- **Don't chase optimal**: With 20 agents and 500 rounds, optimal MAPF is NP-hard. A good heuristic solution computed in 2 seconds beats an optimal solution that takes 2 hours.
- **The drop-off is the real bottleneck**: No MAPF algorithm can make 20 bots deliver through 1 cell faster. The win comes from maximizing drop-off utilization (always have the next bot ready) and minimizing wasted movement.

## Theoretical Score Ceiling Estimate

- Drop-off throughput: ~1 delivery every 2 rounds (approach + deliver action) = ~250 delivery trips
- Average items per trip: depends on order composition and inventory capacity
- With perfect pipeline: 250 trips * avg 3 items = 750 items + 250*5 order bonuses = 2000
- Realistic with congestion: 60-70% efficiency = 1200-1400
- Your current 393 = ~31% of theoretical max
- **Realistic target with PIBT improvements: 600-800 (1.5-2x current)**

## Sources

1. [LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding](https://ojs.aaai.org/index.php/AAAI/article/view/26377/26149) -- Core LaCAM paper (AAAI 2023)
2. [pylacam - Python LaCAM*](https://github.com/Kei18/pylacam) -- Minimal Python implementation
3. [pypibt - Python PIBT](https://github.com/Kei18/pypibt) -- Minimal Python PIBT reference
4. [Lightweight Preference Construction in PIBT](https://arxiv.org/html/2505.12623) -- Hindrance + regret improvements, 40%+ throughput gain
5. [Real-Time LaCAM](https://arxiv.org/html/2504.06091) -- Incremental LaCAM with millisecond budgets
6. [RHCR - Rolling Horizon Collision Resolution](https://github.com/Jiaoyang-Li/RHCR) -- Lifelong MAPF framework (C++)
7. [Traffic Flow Optimization for Lifelong MAPF](https://arxiv.org/html/2308.11234v4) -- Guidance graphs, congestion-aware paths
8. [MAPF-LRR2023 Winner (Team Pikachu)](https://github.com/DiligentPanda/MAPF-LRR2023) -- WPPL algorithm, guidance graphs
9. [Flatland Competition Winner](https://github.com/Jiaoyang-Li/Flatland) -- RHCR + ADG for railway MAPF
10. [MAPF-HD: High Density Environments](https://arxiv.org/html/2509.06374) -- Dense grid MAPF
11. [MAPF-LNS2: Large Neighborhood Search](https://github.com/Jiaoyang-Li/MAPF-LNS2) -- Anytime improvement via LNS
12. [CBS-MAPF Python Package](https://pypi.org/project/cbs-mapf/) -- Simple CBS implementation (limited scalability)
13. [Combined Online Task Assignment and LMAPF](https://arxiv.org/abs/2502.07332) -- Task assignment + path planning integration
14. [Guidance Graph Optimization for LMAPF](https://github.com/lunjohnzhang/ggo_public) -- Automatic guidance graph generation
15. [MAPF-GPT: Imitation Learning](https://github.com/CognitiveAISystems/MAPF-GPT) -- Transformer-based MAPF solver (AAAI 2025)
16. [Persistent Execution of MAPF Schedules](https://whoenig.github.io/publications/2019_RA-L_Hoenig.pdf) -- Action Dependency Graphs for robust replay
17. [Online Guidance Graph Optimization](https://arxiv.org/html/2411.16506v1) -- Dynamic guidance based on real-time traffic
18. [Jiaoyang Li's MAPF Research Overview](https://jiaoyangli.me/research/mapf/) -- Comprehensive MAPF foundations
