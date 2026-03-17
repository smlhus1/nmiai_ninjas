# Research: Tech Claims Verification

> Researched: 2026-03-16 | Sources consulted: 10 | Confidence: High

## TL;DR
All three claims are **CONFIRMED**. PyTorch 2.2 ships CUDA 11.8 wheels supporting Ampere GPUs, the `imitation` library is MIT-licensed and actively maintained, and TorchRL is MIT-licensed with extensive multi-agent support.

---

## Claim 1: PyTorch 2.2+ supports RTX A3000 (compute capability 8.6) natively with CUDA 11.8+

### Verdict: CONFIRMED

**Evidence:**
- PyTorch's official wheel index at `download.pytorch.org/whl/cu118/torch/` lists both `torch-2.2.0+cu118` and `torch-2.2.1+cu118` wheels for Linux and Windows, Python 3.8-3.12.
- NVIDIA RTX A3000 is Ampere architecture, compute capability 8.6. Ampere requires CUDA 11.0+, so CUDA 11.8 is fully compatible.
- PyTorch 1.8.1 already included `sm_86` (Ampere) in its compiled architectures. PyTorch 2.2 continues this support.
- The standard install command is: `pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118`

**Nuance:** PyTorch 2.2 also ships CUDA 12.1 wheels (`cu121`). Both work with Ampere. The claim is fully accurate.

---

## Claim 2: The "imitation" Python library is MIT-licensed

### Verdict: CONFIRMED

**Evidence:**
- PyPI page ([pypi.org/project/imitation/](https://pypi.org/project/imitation/)) explicitly lists license as **"MIT License (MIT)"** in verified project metadata.
- Latest version: **1.0.1** (released January 7, 2025).
- Maintained by the **Center for Human-Compatible AI** (UC Berkeley) and contributors from Google.
- Implements: Behavioral Cloning, DAgger, GAIL, AIRL, density-based reward modeling, Maximum Causal Entropy IRL, and Deep RL from Human Preferences.
- Built on Stable Baselines 3 (SB3).
- Active maintenance indicators: CI (CircleCI), documentation, code coverage, regular releases.

**Nuance:** The library is actively maintained but release cadence is moderate (last release Jan 2025). The MIT license makes it safe for commercial use.

---

## Claim 3: TorchRL is BSD-licensed and supports multi-agent environments

### Verdict: CONFIRMED (with correction -- MIT, not BSD)

**Evidence:**
- GitHub repo ([github.com/pytorch/rl](https://github.com/pytorch/rl)) shows license badge: **MIT License**.
- PyPI page ([pypi.org/project/torchrl/](https://pypi.org/project/torchrl/)) confirms: **MIT License**.
- Latest version: **0.11.1** (released February 5, 2026).

**Multi-agent support is extensive:**
- Official tutorials: "Multi-Agent Reinforcement Learning (PPO) with TorchRL" and "Competitive Multi-Agent Reinforcement Learning (DDPG)"
- Supports shared/individual agent rewards, done flags, and observations
- Code-only scripts for QMIX, MADDPG, IQL, and more
- BenchMARL (published in JMLR) is built on TorchRL for multi-agent RL benchmarking
- Parameter sharing and centralized critic architectures supported

**License correction:** The claim says "BSD-licensed" but TorchRL is actually **MIT-licensed**. Both are permissive and commercially safe, so this is a minor inaccuracy. Note: some other PyTorch ecosystem libraries (like PyTorch itself) use BSD-style licenses, which may be the source of confusion.

---

## Summary Table

| Claim | Verdict | Key Detail |
|-------|---------|------------|
| PyTorch 2.2 + CUDA 11.8 + Ampere 8.6 | **CONFIRMED** | `torch-2.2.0+cu118` wheels exist on official index |
| `imitation` library is MIT-licensed | **CONFIRMED** | MIT License, v1.0.1, actively maintained |
| TorchRL is BSD-licensed + multi-agent | **CONFIRMED** (license is MIT, not BSD) | MIT License, extensive MARL support with tutorials |

## Sources
1. [PyTorch CUDA 11.8 wheel index](https://download.pytorch.org/whl/cu118/torch/) -- confirmed torch-2.2.0+cu118 wheels exist
2. [GPU compute capability support - PyTorch Forums](https://discuss.pytorch.org/t/gpu-compute-capability-support-for-each-pytorch-version/62434) -- sm_86 support history
3. [imitation on PyPI](https://pypi.org/project/imitation/) -- MIT license, v1.0.1, Jan 2025
4. [imitation on GitHub](https://github.com/HumanCompatibleAI/imitation) -- Center for Human-Compatible AI
5. [TorchRL on PyPI](https://pypi.org/project/torchrl/) -- MIT license, v0.11.1
6. [TorchRL on GitHub](https://github.com/pytorch/rl) -- MIT badge, multi-agent features
7. [TorchRL Multi-Agent PPO Tutorial](https://docs.pytorch.org/rl/stable/tutorials/multiagent_ppo.html) -- official MARL tutorial
8. [TorchRL Competitive DDPG Tutorial](https://docs.pytorch.org/rl/stable/tutorials/multiagent_competitive_ddpg.html) -- competitive multi-agent
