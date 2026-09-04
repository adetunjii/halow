# HALOW: Heterogeneous Agent Learning in the Open World

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-Parallel_Env-brightgreen.svg)](https://pettingzoo.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**HALOW** is a Multi-Agent Reinforcement Learning (MARL) framework designed for autonomous, collaborative exploration of unknown, GNSS-denied environments using a heterogeneous robotic team. It combines classical probabilistic occupancy mapping, frontier extraction, and heuristic guidance with learned decision-making via **Multi-Agent Proximal Policy Optimization (MAPPO)** under a Centralized Training with Decentralized Execution (CTDE) paradigm.

---

## Table of Contents
- [Overview](#overview)
- [System Architecture & Agent Heterogeneity](#system-architecture--agent-heterogeneity)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [1. Training MAPPO](#1-training-mappo)
  - [2. Training IPPO Baseline](#2-training-ippo-baseline)
  - [3. Evaluating Trained Policies](#3-evaluating-trained-policies)
  - [4. Environment Rendering](#4-environment-rendering)
  - [5. Visualizing Training Logs](#5-visualizing-training-logs)
- [Theoretical Formulation](#theoretical-formulation)
  - [Heterogeneous Next-Best-View (H-NBV) Scoring](#heterogeneous-next-best-view-h-nbv-scoring)
  - [Reward Structure](#reward-structure)
  - [CTDE State-Action Formulation](#ctde-state-action-formulation)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

---

## Overview

Autonomous exploration of unknown terrain typically requires balancing rapid spatial coverage against map accuracy. Homogeneous teams often duplicate sensory effort or suffer from uniform kinematic bottlenecks. HALOW addresses this by modeling a complementary two-agent team:
1. **Unmanned Aerial Vehicle (UAV / Drone):** Serves as an agile scout. Unconstrained by ground obstacles, it transits rapidly ($3\times$ speed) with a wider sensor radius, but produces noisy, confidence-bounded occupancy updates.
2. **Unmanned Ground Vehicle (UGV / Rover):** Serves as a high-fidelity mapper and verifier. Constrained by obstacle topology (navigating via $A^*$ path planning) and moving at single-step speed, it provides near-ground-truth sensor fidelity to definitively resolve ambiguous cells.

The agents coordinate implicitly without explicit inter-robot communication channels during execution.

---

## System Architecture & Agent Heterogeneity

| Property | UAV (Drone / Scout) | UGV (Rover / Mapper) |
| :--- | :--- | :--- |
| **Observation Radius ($R$)** | $R = 3$ cells | $R = 2$ cells |
| **Kinematic Step Rate** | $3$ sub-steps per macro env step | $1$ sub-step per macro env step |
| **Obstacle Constraint** | None (elevated linear interpolation) | $A^*$ grid search on shared belief |
| **Sensor Confidence** | $\text{Occupied}: 0.70, \; \text{Free}: 0.70$ | $\text{Occupied}: 0.99, \; \text{Free}: 0.99$ |
| **Observation Space** | Private belief map + poses + frontiers | Shared fused belief + poses + frontiers |
| **Primary Objective** | Discover novel frontiers, scout unknown cells | Verify drone-scanned cells, finalize map |

---

## Key Components

- **Bayesian Log-Odds Mapping:** Real-time occupancy grid fusion incorporating sensor models, communication noise ($\mathcal{N}(0, \sigma^2)$), and update limits.
- **Heterogeneous Next-Best-View (H-NBV):** Action-candidate extraction algorithm providing top-$K$ frontiers scored with role-differentiated utility, kinematic distance normalization, and mutual deconfliction penalties.
- **CTDE MAPPO Policy:** Decentralized actor networks operating with action masks; centralized value critic with privileged global ground-truth access during training.
- **Independent PPO (IPPO) Baseline:** Decentralized ablation baseline utilizing local critic evaluations for benchmark comparison.
- **PettingZoo Parallel API:** Fully vectorized multi-agent environment compliance.

---

## Installation

### Prerequisites
- Python $\ge$ 3.11
- PyTorch $\ge$ 2.0 (with CUDA, MPS, or CPU support)

### Setup
Clone the repository and install the package in editable mode:

```bash
# Clone repository
git clone https://github.com/adetunjii/halow.git
cd halow

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and package
pip install --upgrade pip
pip install -e .
```

---

## Usage Guide

### 1. Training MAPPO

To train the full MAPPO architecture with centralized critic:

```python
from halow.environment import CustomEnvironment
from halow.mappo.mappo import Mappo

# Initialize environment
env = CustomEnvironment()

# Initialize MAPPO trainer
trainer = Mappo(env)

# Run 20-30 training iterations (128 episodes per batch)
trainer.train(runs=30)
```

Checkpoints will be saved automatically to `halow/mappo/weights/` and metrics to `halow/mappo/logs/training_log.csv`.

### 2. Training IPPO Baseline

To train the independent decentralized critic baseline:

```python
from halow.environment import CustomEnvironment
from halow.ippo.ippo import Ippo

env = CustomEnvironment()
trainer = Ippo(env)
trainer.train(runs=30)
```

### 3. Evaluating Trained Policies

Evaluate frozen policies across 50 unseen procedural test seeds (outside the training distribution):

```bash
python -m halow.eval_mappo
```

This outputs benchmark metrics:
- **Success Rate** ($\ge 90\%$ map coverage reached before battery/step timeout)
- **Mean Map Coverage** ($\% \pm \sigma$)
- **Episode Duration** (Steps to termination)
- **Cumulative Team Returns** ($R_D, R_R$)

### 4. Environment Rendering

To run real-time visualization with matplotlib showing live coverage, step counter, animated agents, and per-agent battery status:

```python
from halow.environment import CustomEnvironment

env = CustomEnvironment()
obs, infos = env.reset(seed=42)

for _ in range(200):
    # Dummy random action or policy action
    actions = {
        agent: env.action_space(agent).sample(infos[agent]["action_mask"])
        for agent in env.agents
    }
    obs, rewards, term, trunc, infos = env.step(actions)
    env.render()
    if any(term.values()) or any(trunc.values()):
        break
```

### 5. Visualizing Training Logs

Inspect training curves (Returns, Critic MSE Loss, Entropy Decay, Episode Duration) using the provided Jupyter notebook:

```bash
jupyter notebook halow/viz.ipynb
```

---

## Theoretical Formulation

### Heterogeneous Next-Best-View (H-NBV) Scoring

For each candidate frontier $\mathbf{f} \in \mathcal{F}_a$ within sensor visibility $\mathcal{V}_a(\mathbf{f})$:

$$S_a(\mathbf{f}) = \left( \frac{\mathcal{I}_a(\mathbf{f}) + \epsilon}{d_a^{\text{eff}}(\mathbf{p}_a, \mathbf{f})} \right) \cdot \Psi(\mathbf{f}, \mathbf{f}_{-a})$$

1. **Role-Differentiated Information ($\mathcal{I}_a$):**
   $$\mathcal{I}_a(\mathbf{f}) = \sum_{\mathbf{x} \in \mathcal{V}_a(\mathbf{f})} \mathbb{I}(\mathbf{x} \in \Omega_{\text{unc}}) + \mathbb{I}(a = \text{UGV}) \cdot \beta \sum_{\mathbf{x} \in \mathcal{V}_a(\mathbf{f})} \mathbb{I}(\mathbf{x} \in \Omega_{\text{refine}})$$
   where $\Omega_{\text{unc}} = [\tau_{\text{free}}, \tau_{\text{occ}}]$ defines unresolved cells, and $\Omega_{\text{refine}} = [0.35, \tau_{\text{free}}) \cup (\tau_{\text{occ}}, 0.65]$ designates partially-observed drone scans awaiting ground verification ($\beta = 0.3$).

2. **Kinematic Transit Distance ($d_a^{\text{eff}}$):**
   $$d_a^{\text{eff}}(\mathbf{p}_a, \mathbf{f}) = \max\left( \frac{\|\mathbf{p}_a - \mathbf{f}\|_2}{v_a}, \; 1.0 \right)$$
   where $v_{\text{UAV}} = 3$ and $v_{\text{UGV}} = 1$, normalizing Euclidean space into time-to-arrival.

3. **Mutual Deconfliction Discount ($\Psi$):**
   $$\Psi(\mathbf{f}, \mathbf{f}_{-a}) = 1 - \frac{\lambda}{1 + \|\mathbf{f} - \mathbf{f}_{-a}\|_2}, \quad \lambda = 0.5$$
   Penalizes spatial redundancy with the peer agent's committed frontier $\mathbf{f}_{-a}$.

### Reward Structure

The step-wise reward for agent $a \in \{\text{drone}, \text{rover}\}$ is defined as:

$$r_t^a = \frac{\Delta \text{Resolved Cells}}{H \times W} - \alpha \cdot \text{Penalty}_a(a_t) + r_{\text{terminal}}$$

- **Normalized Info Gain:** Shared team reward proportional to newly resolved occupancy cells ($< 0.2$ or $> 0.8$).
- **Suboptimal Action Penalty:** $\alpha \cdot \frac{S^* - S(a_t)}{S^*}$ penalizes selecting lower-scoring frontiers relative to the greedy candidate ($\alpha = 0.1$).
- **Terminal Battery Bonus:** If coverage target ($90\%$) is attained:
  $$r_{\text{terminal}} = c_{\text{bat}} \cdot (B_{\text{drone}} + B_{\text{rover}})$$

### CTDE State-Action Formulation

- **Decentralized Actor Input:** Local belief map (flattened $H \times W$) + normalized agent poses ($\mathbf{p}_a, \mathbf{p}_{-a}$) + top-$K$ packed frontier features + remaining battery $B_a$.
- **Centralized Critic Input:** Concatenation of all local observations + privileged full ground-truth map $\mathcal{M}_{\text{GT}}$ + joint poses + joint batteries.
- **Action Space:** Discrete choice over top-$K$ ranked frontiers ($a \in \{0, \dots, K-1\}$) plus a fallback idle/wait action ($a = K$).

---

## Repository Structure

```
halow/
├── halow/
│   ├── assets/              # Sprites and animation gifs (drone.gif, rover.gif)
│   ├── environment.py       # PettingZoo ParallelEnv with Bayesian grid updates
│   ├── agent.py             # Agent definitions, sensor models, log-odds updates
│   ├── frontiers.py         # H-NBV scoring, frontier detection, candidate packing
│   ├── constants.py         # Environment constants, sensor parameters, thresholds
│   ├── helpers.py           # A* search, Bresenham line generation, coordinate utils
│   ├── train_mappo.py       # Training launch script
│   ├── eval_mappo.py        # Evaluation suite on 50 unseen test seeds
│   ├── viz.ipynb            # Analysis & visualization notebook for training logs
│   ├── mappo/               # Multi-Agent PPO package
│   │   ├── actor.py         # Decentralized MLP Actor with action masking
│   │   ├── critic.py        # Centralized Critic
│   │   ├── buffer.py        # On-policy rollout buffer with GAE(lambda)
│   │   ├── mappo.py         # MAPPO trainer implementation
│   │   ├── weights/         # Saved policy model checkpoints
│   │   └── logs/            # CSV training logs and convergence plots
│   └── ippo/                # Independent PPO baseline package
│       ├── buffer.py        # Decentralized rollout buffer
│       └── ippo.py          # IPPO trainer implementation
├── pyproject.toml           # Package configuration and build specification
└── README.md                # Framework documentation
```

---

## Citation

If you use this framework or codebase in your research, please cite:

```bibtex
@misc{halow2026,
  title={HALOW: Heterogeneous Agent Learning in the Open World for Collaborative Robotic Exploration},
  author={Adetunji, Hajiyavand},
  year={2026},
  howpublished={\url{https://github.com/adetunjii/halow}}
}
```