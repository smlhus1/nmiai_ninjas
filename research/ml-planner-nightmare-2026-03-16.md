# Research: GPU-Based ML Planner for Multi-Agent Grocery Warehouse Bot

> Researched: 2026-03-16 | Sources consulted: 25+ | Confidence: High

## TL;DR

For your specific problem (20 bots, 30x18 grid, known order sequence, 2s response time, 6GB VRAM, daily retraining), **pure RL is overkill and risky**. The highest-ROI approach is a **hybrid: learned scoring function + beam search over assignments**, with imitation learning from your existing heuristic as bootstrap. If you want the full RL path, MAPF-GPT's imitation learning approach is the closest proven architecture, but it solves pathfinding (which you already have via PIBT), not assignment. For assignment specifically, a **GNN on bot-item bipartite graph** trained with PPO is the most architecturally natural fit.

---

## Key Findings

### 1. Multi-Agent RL for Warehouse/MAPF — State of the Art

#### MAPF-GPT (AAAI 2025) — Most Relevant Paper
- **What**: Transformer-based imitation learning for MAPF, trained on LaCAM expert solutions
- **Architecture**: Decoder-only transformer, three sizes: 2M, 6M, 85M parameters
- **Input**: 11x11 FOV per agent, tokenized as 256 tokens (cost-to-go + agent info)
- **Training data**: 1 billion (observation, action) pairs from 3.75M problem instances
- **Training time**:
  - 2M model: **12 hours** on 1x H100 80GB (40M pairs)
  - 6M model: **50 hours** on 2x A100 80GB (150M pairs)
  - 85M model: **243 hours** on 4x H100 80GB (1B pairs)
- **Performance**: 13x faster inference than SCRIMP, 8x faster than DCC at 192 agents
- **Key insight**: Solves pathfinding/movement, NOT task assignment. Your PIBT already does this.
- GitHub: [CognitiveAISystems/MAPF-GPT](https://github.com/CognitiveAISystems/MAPF-GPT)
- Paper: [arxiv.org/abs/2409.00134](https://arxiv.org/abs/2409.00134)

#### Other MAPF-RL Methods
| Method | Type | Scales to 20+ agents? | Training cost | Notes |
|--------|------|----------------------|---------------|-------|
| PRIMAL | RL + IL hybrid | Yes (tested 16-64) | Hours on GPU | Decentralized, CNN-based |
| SCRIMP | RL + transformer comm | Poor (heavy compute) | Days | Single-step collision avoidance is bottleneck |
| DCC | RL + selective comm | Moderate | Days | Decision Causal Communication |
| EPH | Ensemble + prioritized | Yes | Hours | Ensembling Prioritized Hybrid policies |
| LaCAM | Classical (no ML) | Excellent (1000+) | N/A | Used to generate MAPF-GPT training data |

**Critical observation**: All these solve PATHFINDING. Your bottleneck is ASSIGNMENT (which bot gets which item). The tactical layer (PIBT) already handles collision avoidance well.

#### CTDE (Centralized Training, Decentralized Execution)
- Standard paradigm for multi-agent RL
- Train with global state access (all bot positions, all items, full order info)
- Execute with local observations per bot
- Techniques: QMIX (value decomposition), MAPPO (multi-agent PPO with centralized critic)
- **Relevant for your case**: You have full observability at decision time (coordinator sees everything), so CTDE's decentralized execution constraint is unnecessary. You can use a fully centralized policy.

### 2. Architecture Options for 6GB VRAM

#### Option A: Learned Scoring + Beam Search (RECOMMENDED)
- **What**: Train a small neural network to score (bot, item) assignment pairs, then use beam search to find the best global assignment
- **Architecture**: MLP or small GNN, ~100K-2M parameters
- **VRAM**: < 100MB. Trivially fits in 6GB.
- **Training**: Supervised learning on (state, assignment, score) triples from sim rollouts. Hours on your GPU.
- **Inference**: Score all candidate assignments, beam search over top-k. < 10ms.
- **Why this works**: Your problem is fundamentally a scoring problem. Given a state, how good is assigning bot X to item Y? A learned scorer replaces hand-tuned heuristics.
- **Literature**: [Simulation-guided Beam Search for Neural Combinatorial Optimization](https://openreview.net/forum?id=tYAS1Rpys5) — proven approach for combinatorial optimization.

#### Option B: GNN on Bot-Item Bipartite Graph
- **What**: Represent bots and items as nodes in a bipartite graph. GNN produces assignment probabilities.
- **Architecture**: 2-3 layers of Graph Attention Network, ~500K-5M parameters
- **VRAM**: < 500MB for 20 bots + ~50 items
- **Training**: PPO with REINFORCE on assignment quality (throughput reward)
- **Inference**: Single forward pass, ~1-5ms
- **Literature**: [MAGNET (AAMAS 2025)](https://dl.acm.org/doi/10.5555/3709347.3743773) — GNN + PPO for bipartite task assignment, 92.5% conflict-free rate, 7.49% gap vs Hungarian optimal
- **Advantage**: Naturally handles variable numbers of items/bots
- **Risk**: Reward shaping is hard; may need curriculum learning

#### Option C: Pointer Network / Attention (Sequential Assignment)
- **What**: Encoder-decoder with attention that sequentially assigns bots to items
- **Architecture**: Small transformer (2-4 layers, 128-256 dim), ~1-5M parameters
- **VRAM**: < 500MB
- **Training**: REINFORCE algorithm, as in [Attention, Learn to Solve Routing Problems](https://openreview.net/pdf?id=ByxBFsRqYm)
- **Inference**: ~5-20ms for 20 bots
- **Advantage**: Proven on VRP (Vehicle Routing Problem), which is structurally similar
- **Risk**: Sequential decoding means order matters; autoregressive = slower than parallel

#### Option D: PPO with Shared Policy (All Bots Share One Network)
- **What**: Standard multi-agent PPO (IPPO or MAPPO). Each bot gets same policy network.
- **Architecture**: MLP with 2x256 hidden layers, ~200K parameters
- **VRAM**: < 200MB for model. Main cost is experience buffer.
- **Training**: 10K-100K episodes needed. With vectorized sim: hours. Without: days.
- **Inference**: Batch forward pass for 20 bots, < 1ms
- **Libraries**: TorchRL (official PyTorch, excellent multi-agent tutorial), CleanRL (simpler)
- **Risk**: Reward shaping is THE hard problem. Sparse order-completion reward = very slow learning.

#### Option E: GPU Monte Carlo Tree Search
- **What**: Parallel MCTS on GPU to search over assignment space
- **Implementation**: [MCTS-NC](https://github.com/pklesk/mcts_numba_cuda) — Numba CUDA, lock-free, 18.8M+ playouts/second
- **VRAM**: Configurable (default 2GB tree allocation)
- **Key advantage**: No training needed. Pure search with your existing sim as evaluator.
- **Key disadvantage**: Your sim is Python, not CUDA. Porting sim to GPU is the bottleneck.
- **Hybrid approach**: Use learned value function to guide MCTS rollouts instead of random playouts

#### Option F: Imitation Learning (Behavioral Cloning / DAgger)
- **What**: Train a network to mimic your current heuristic planner's decisions
- **Architecture**: Any of the above (MLP, GNN, Pointer Net)
- **Training data**: Run your current bot 100-1000 games, record (state, assignment) pairs
- **Training time**: Minutes to hours (supervised learning is fast)
- **Ceiling**: Can only match your heuristic, not exceed it (without DAgger-style iteration)
- **Use as bootstrap**: Train IL model first, then fine-tune with RL (PPO). Common and effective pattern.
- **Library**: [imitation](https://imitation.readthedocs.io/) — clean BC and DAgger implementation

### 3. Training Pipeline Technologies

#### Framework Comparison

| Framework | Multi-Agent | Custom Env | Ease of Use | GPU Efficiency | Recommendation |
|-----------|-------------|------------|-------------|----------------|----------------|
| **TorchRL** | Yes (native) | Good (TensorDict) | Medium | Good (vectorized) | Best for PyTorch users |
| **CleanRL** | Limited | Manual | Easy (single-file) | Moderate | Best for learning/prototyping |
| **RLlib (Ray)** | Yes (mature) | Good | Medium-Hard | Good (distributed) | Overkill for single GPU |
| **Stable-Baselines3** | No native | Easy | Easiest | Moderate | Not for multi-agent |
| **JaxMARL** | Yes (native) | Requires JAX rewrite | Hard | **Best** (12,500x speedup) | Best if you port sim to JAX |
| **WarpDrive** | Yes (CUDA) | CUDA C or Numba | Hard | Excellent | Best if you port sim to CUDA |
| **PettingZoo** | API standard | Wrapper only | Easy | N/A (env API) | Use as interface, not trainer |

#### Key Decision: Sim Acceleration

Your Python simulator is the bottleneck for any RL approach. Options:

1. **Keep Python sim, use SubprocVecEnv** (easiest): 8-16 parallel envs on CPU. Training in days.
2. **Port sim to JAX** (medium effort): JIT-compile grid logic. 100-1000x speedup. Training in hours.
3. **Port sim to CUDA via Numba** (hard): Maximum throughput. Training in minutes.
4. **Port sim to CUDA via WarpDrive** (hard): End-to-end GPU. Training in minutes.
5. **Don't use RL** (smartest?): Learned scorer + search doesn't need fast sim. Use existing sim for data generation.

**Realistic assessment for JAX port**: Your sim is ~500 lines of Python with dict/list operations, A* pathfinding, collision resolution. Porting to JAX requires rewriting ALL data structures as fixed-size JAX arrays (no dicts, no dynamic lists, no classes with mutable state). Estimate: **2-4 days of focused work** for someone experienced with JAX. Major gotchas: variable-length orders, dynamic item spawning, A* with variable-length paths.

**Realistic assessment for Numba port**: Similar effort to JAX but uses Python-like syntax. Numba CUDA kernels require fixed-size arrays and no Python objects. Estimate: **3-5 days**.

### 4. Reward Shaping for Warehouse

This is THE make-or-break factor for any RL approach.

#### What works (from literature + competition experience):
- **Per-item delivery bonus** (+1 per item delivered to drop-off)
- **Order completion bonus** (+5 per completed order, matching your scoring)
- **Distance-based shaping**: Small negative reward proportional to distance to target (dense signal)
- **Throughput metric**: Reward = items_delivered / rounds_elapsed (velocity, not total)
- **Potential-based shaping** (theoretically optimal): F(s,s') = gamma * phi(s') - phi(s) where phi = negative total distance of all bots to their assigned items

#### What fails:
- **Pure sparse reward** (only order completion): Too sparse for 20 agents, 300 rounds
- **Per-step penalty** (-0.01 per round): Makes agents rush, ignoring coordination
- **Collision penalty**: Already handled by PIBT, don't penalize what's solved

#### Curriculum Learning
- Start with 1-3 bots on simple orders, increase to 20 bots with full complexity
- Start with short episodes (50 rounds), extend to 300
- Start with orders near drop-off, extend to full map
- **VACL** (Variational Automatic Curriculum Learning): automated curriculum for cooperative MARL
- Literature: [NeurIPS 2021 paper](https://proceedings.neurips.cc/paper/2021/file/503e7dbbd6217b9a591f3322f39b5a6c-Paper.pdf)

#### Hindsight Experience Replay (HER)
- Retrofits sparse reward episodes by relabeling goals
- **Applicable**: When a bot fails to deliver an order, relabel the episode as "successfully navigated to position X"
- **Limitation**: Multi-agent HER is still research-stage. [PHER](https://link.springer.com/article/10.1007/s40747-023-00985-w) extends HER to multi-agent with parallel hindsight.

### 5. Inference Speed

For a 2-second response time budget, inference speed is not a concern for any reasonable model size.

| Model Type | Parameters | Batch Size | RTX A3000 Inference | Notes |
|-----------|-----------|------------|---------------------|-------|
| MLP (2x256) | 200K | 20 (one per bot) | **< 0.1ms** | Trivial |
| Small GNN (3 layers) | 500K | 1 (full graph) | **< 1ms** | Single forward pass |
| Pointer Net (4 layers, 256d) | 2M | 1 | **< 5ms** | Autoregressive over 20 bots |
| Transformer (6 layers, 256d) | 5M | 20 | **< 2ms** | Parallel attention |
| MAPF-GPT 2M | 2M | 20 | **< 5ms** | Based on paper benchmarks |

**ONNX Runtime** can cut latency by 2-5x over PyTorch eager mode for small models. **TensorRT** provides further optimization with FP16/INT8, achieving sub-millisecond for BERT-class models. For your model sizes, even raw PyTorch is fast enough.

**Bottom line**: Any model under 10M parameters will inference in < 10ms on your GPU. The 2-second budget is 200x more than you need.

### 6. Relevant Open Source

#### Directly Relevant
| Project | What | Stars | URL |
|---------|------|-------|-----|
| MAPF-GPT | Imitation learning for MAPF | ~200 | [GitHub](https://github.com/CognitiveAISystems/MAPF-GPT) |
| RWARE | Multi-robot warehouse RL env | ~400 | [GitHub](https://github.com/semitable/robotic-warehouse) |
| TA-RWARE | Task-assignment warehouse env | ~50 | [GitHub](https://github.com/uoe-agents/task-assignment-robotic-warehouse) |
| JaxMARL | GPU-accelerated MARL | ~800 | [GitHub](https://github.com/FLAIROx/JaxMARL) |
| MCTS-NC | GPU-parallel MCTS (Numba CUDA) | ~30 | [GitHub](https://github.com/pklesk/mcts_numba_cuda) |
| WarpDrive | End-to-end GPU RL | ~400 | [GitHub](https://github.com/salesforce/warp-drive) |
| PureJaxRL | End-to-end JAX RL (4000x speedup) | ~1K | [GitHub](https://github.com/luchris429/purejaxrl) |
| PRIMAL | RL+IL for MAPF | ~200 | [GitHub](https://github.com/gsartoretti/PRIMAL) |
| imitation | BC + DAgger library | ~1K | [Docs](https://imitation.readthedocs.io/) |

#### Competition References
| Competition | Similarity | Winning Approach | Key Lesson |
|------------|-----------|------------------|------------|
| Flatland (rail scheduling) | High | MAPF + LNS + simulated annealing (classical) beat RL | Classical search still wins competitions |
| RWARE benchmark | Medium | QMIX outperformed IPPO by 8.5x | Value decomposition > independent policies |
| POGEMA benchmark | High | MAPF-GPT (IL) beat all RL methods | Imitation learning is underrated |

**Critical pattern from competitions**: Classical/hybrid approaches consistently beat pure RL in constrained multi-agent coordination. The Flatland 2020 winner used zero ML. The best MAPF solvers are still classical (LaCAM, CBS, EECBS).

### 7. Alternatives to Pure ML

#### A. Learned Heuristic + Classical Search (HIGHEST CONFIDENCE)
- Train a small MLP/GNN to score candidate assignments
- Use scores in beam search / branch-and-bound over assignment space
- **Training**: Supervised on (state, assignment, sim_score) from offline rollouts
- **Search budget**: 100-1000 candidates evaluated per round in < 100ms
- **Advantage**: Interpretable, debuggable, incremental improvement over current heuristic
- **This is what Google DeepMind does for chip placement, scheduling, etc.**

#### B. Monte Carlo Rollouts with Learned Value Function
- Run N random assignment rollouts in your sim for K steps (not full game)
- Train a value network to predict rollout outcome from state
- Use value network to replace expensive sim rollouts
- **Training**: Supervised on (state, simulated_score) pairs
- **Inference**: Forward pass + argmax, < 5ms
- **Key reference**: [Simulation-guided Beam Search](https://openreview.net/forum?id=tYAS1Rpys5)

#### C. Genetic/Evolutionary Search with GPU Parallelization
- You ALREADY have a genome-based search (from `genome_search_plan.md`)
- GPU-parallelize the fitness evaluation: batch 100+ genomes simultaneously
- Use Numba CUDA to parallelize sim rollouts on GPU
- **Speedup**: 30-100x over sequential CPU (based on CUDA GA literature)
- **Key reference**: [GPU Parallel Genetic Algorithm](https://arxiv.org/pdf/1903.10741)
- **Practical note**: You'd need to port your sim to Numba CUDA first

#### D. Differentiable Assignment (Research-Stage)
- Replace Hungarian algorithm with differentiable Sinkhorn operator
- Backpropagate through assignment layer to learn scoring
- Libraries: [OptNet](https://arxiv.org/pdf/1703.00443), cvxpylayers
- **Risk**: High complexity, research-stage, unclear benefit over simpler approaches
- **Skip this** unless other approaches plateau

---

## Comparison / Options Analysis

Ranked by expected ROI (impact / effort) for your specific situation:

| Rank | Approach | Effort | Expected Gain | VRAM | Training Time | Risk |
|------|----------|--------|---------------|------|--------------|------|
| 1 | **Learned Scorer + Beam Search** | 2-3 days | +50-200 pts | <100MB | 2-4 hours | Low |
| 2 | **IL Bootstrap + PPO Fine-tune** | 3-5 days | +100-300 pts | <500MB | 4-12 hours | Medium |
| 3 | **GNN Bipartite Assignment** | 5-7 days | +100-400 pts | <500MB | 8-24 hours | Medium-High |
| 4 | **GPU-Parallel Genome Search** | 3-5 days (sim port) | +50-150 pts | <2GB | N/A (search) | Medium |
| 5 | **Pointer Net Assignment** | 4-6 days | +100-300 pts | <500MB | 6-18 hours | Medium |
| 6 | **MAPPO/IPPO Full RL** | 7-14 days | +0-500 pts | 2-4GB | 1-3 days | HIGH |
| 7 | **JAX Sim + PureJaxRL** | 5-10 days (sim port) | +100-500 pts | 2-4GB | Hours | HIGH |

---

## Gotchas & Considerations

### Your Problem is Special
- **Deterministic order sequence**: You KNOW what orders come next. This is a massive advantage that pure RL doesn't exploit. A learned scorer that takes future orders as input can plan ahead.
- **Fixed map**: Grid never changes. Precompute all distances. BFS cache is already done.
- **Known sim**: You have a validated simulator. Use it for data generation, not just RL training.
- **Daily retraining**: Order sequence changes daily. Model must retrain each day. This favors fast-training approaches (supervised learning, few-shot) over slow RL.

### Reward Shaping Will Be Painful
- 20 agents + sparse rewards + 300 round horizon = credit assignment nightmare
- Multi-agent credit assignment: "which bot caused the order completion?" is fundamentally hard
- QMIX-style value decomposition helps but adds complexity
- **Mitigation**: Use supervised learning (IL/scoring) to bootstrap, then RL to fine-tune

### Sim-to-Live Gap
- Your sim already has validated collision model matching the server
- BUT: Order generation beyond recon data uses Random(42), which differs from live
- Training on sim-generated orders will overfit to sim
- **Mitigation**: Only train on recon data, or use the model for scoring (not full policy)

### JAX/CUDA Port is High-Risk
- Your sim uses Python dicts, classes, dynamic lists, A* with priority queues
- JAX requires pure functions, fixed-size arrays, no dynamic allocation
- Numba CUDA requires no Python objects, fixed-size everything
- A buggy port could waste days and produce invalid training signal
- **Mitigation**: Start with CPU-based approaches (scoring, IL) that use existing sim

### The 393 vs 1361 Gap
- First place scores 1361. You score 393. That's 3.46x gap.
- The gap is likely NOT just assignment quality — it's probably a fundamentally different architecture (pre-computed plan for all 300 rounds? MAPF solver? Different tactical layer?)
- ML alone won't close a 3.46x gap if the tactical layer (PIBT) has inherent throughput limits
- **Investigation**: What does a theoretically optimal assignment look like for your recon data? If optimal assignment in your sim gives ~500, the ceiling is assignment. If it gives ~1300, the ceiling is pathfinding/coordination.

---

## Recommendations

### Phase 1: Quick Win (2-3 days)
**Learned Scorer + Beam Search**

1. Generate training data: Run your current bot 50-100 games with recon data. Log every (state, assignment, next_5_round_score) tuple.
2. Train a small MLP (3 layers, 256 hidden) to predict score given (bot_features, item_features, global_features).
3. At each round: generate top-100 candidate assignments, score with network, pick best.
4. Fine-tune by running the new bot, collecting more data, retraining.

This is low-risk, uses your existing sim, and gives interpretable results.

### Phase 2: Imitation + RL (3-5 more days)
**If Phase 1 plateaus:**

1. Use Phase 1's best bot as expert for imitation learning (behavioral cloning)
2. Wrap sim as PettingZoo/TorchRL environment
3. Fine-tune with PPO using dense reward shaping (distance reduction + per-item bonus)
4. Use MAPPO with centralized critic (all bots' states visible)
5. TorchRL is the recommended framework (native PyTorch, good multi-agent support)

### Phase 3: Architecture Upgrade (if needed)
**If Phase 2 plateaus:**

1. Replace MLP scorer with GNN on bipartite bot-item graph
2. Add future order information as context (your key advantage)
3. Consider Pointer Network for sequential assignment generation
4. Profile whether pathfinding/PIBT is the actual bottleneck (not assignment)

### What NOT To Do
- Don't port sim to JAX/CUDA until you've proven ML helps with CPU-based approaches
- Don't train MAPF-GPT — it solves pathfinding, not assignment
- Don't use RLlib — overkill for single-GPU, massive dependency bloat
- Don't use pure sparse reward RL without curriculum learning
- Don't spend more than 1 day on reward shaping before trying supervised approaches

---

## Sources

1. [MAPF-GPT: Imitation Learning for Multi-Agent Pathfinding at Scale (AAAI 2025)](https://arxiv.org/abs/2409.00134) — Architecture details, training times, performance benchmarks
2. [MAPF-GPT GitHub](https://github.com/CognitiveAISystems/MAPF-GPT) — Implementation, pretrained models
3. [MAGNET: Multi-Agent GNN for Bipartite Task Assignment (AAMAS 2025)](https://dl.acm.org/doi/10.5555/3709347.3743773) — GNN + PPO for assignment, 92.5% conflict-free
4. [RWARE: Multi-Robot Warehouse Environment](https://github.com/semitable/robotic-warehouse) — Reference environment for warehouse RL
5. [TA-RWARE: Task-Assignment Warehouse](https://github.com/uoe-agents/task-assignment-robotic-warehouse) — Task assignment variant
6. [TorchRL Multi-Agent PPO Tutorial](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html) — Training setup, architecture, vectorization
7. [JaxMARL: Multi-Agent RL in JAX](https://arxiv.org/abs/2311.10090) — 12,500x speedup, custom env creation
8. [PureJaxRL](https://github.com/luchris429/purejaxrl) — 4000x speedup for end-to-end JAX RL
9. [MCTS-NC: GPU-Parallel MCTS](https://github.com/pklesk/mcts_numba_cuda) — Numba CUDA MCTS, lock-free design
10. [WarpDrive: Multi-Agent RL on GPU](https://github.com/salesforce/warp-drive) — End-to-end GPU RL, 100x throughput
11. [Simulation-guided Beam Search for Neural Combinatorial Optimization](https://openreview.net/forum?id=tYAS1Rpys5) — Learned heuristic + beam search
12. [Attention, Learn to Solve Routing Problems](https://openreview.net/pdf?id=ByxBFsRqYm) — Pointer network for VRP
13. [PRIMAL: RL + IL for MAPF](https://github.com/gsartoretti/PRIMAL) — Distributed RL/IL for pathfinding
14. [Flatland Challenge](https://www.aicrowd.com/challenges/flatland) — Rail scheduling competition (classical > RL)
15. [Scalable Rail Planning: Winning Flatland 2020](https://www.researchgate.net/publication/365035961_Scalable_Rail_Planning_and_Replanning_Winning_the_2020_Flatland_Challenge) — Classical MAPF + LNS beat RL
16. [VACL: Variational Automatic Curriculum Learning](https://proceedings.neurips.cc/paper/2021/file/503e7dbbd6217b9a591f3322f39b5a6c-Paper.pdf) — Automated curriculum for sparse-reward MARL
17. [Cooperative multi-agent target searching with PHER](https://link.springer.com/article/10.1007/s40747-023-00985-w) — Parallel HER for multi-agent
18. [imitation library (BC + DAgger)](https://imitation.readthedocs.io/) — Clean imitation learning implementation
19. [PettingZoo: Multi-Agent RL API](https://pettingzoo.farama.org/) — Environment wrapper standard
20. [Learning Beam Search for Combinatorial Optimization](https://www.ac.tuwien.ac.at/files/pub/huber-21.pdf) — RL-trained beam search guidance
21. [GPU Parallel Genetic Algorithm](https://arxiv.org/pdf/1903.10741) — 30x speedup for TSP with CUDA GA
22. [RL for MAPF via Distributed Policy Evolution (IEEE)](http://ieeexplore.ieee.org/iel8/7083369/11045364/11034721.pdf) — Evolutionary RL for warehouse MAPF
23. [NVIDIA Warp](https://github.com/NVIDIA/warp) — GPU simulation framework, differentiable
24. [OptNet: Differentiable Optimization Layers](https://arxiv.org/pdf/1703.00443) — Embedding optimization in neural networks
25. [CleanRL: Single-file RL Implementations](https://github.com/vwxyzjn/cleanrl) — PPO reference implementation

---

## Follow-Up Questions Worth Investigating

1. **What does the #1 team actually use?** If their architecture is fundamentally different (e.g., full MAPF solver like LaCAM), no amount of ML on assignment will close the gap.
2. **What's the theoretical maximum score with perfect assignment but current PIBT?** Run your sim with oracle-optimal assignments to measure the ceiling.
3. **Does the order sequence have learnable patterns?** If tomorrow's orders correlate with today's, a model could pre-learn item distributions.
4. **Can you get more recon data?** ML approaches scale with data. Running your bot 100 games/day gives much better training signal than 1-2.
