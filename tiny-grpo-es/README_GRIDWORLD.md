# ES for Non-Differentiable RL: GridWorld Experiments

This repository contains code for comparing **Evolution Strategies (ES)** with **PPO** on simple gridworld environments, as a proof-of-life for parameter-space optimization in sparse reward settings.

## Project Overview

**Goal**: Study parameter-space optimization (ES) vs action-space RL (PPO) in long-horizon, sparse-reward environments where credit assignment is challenging.

**Hypothesis**: ES may exhibit more stable learning than PPO in environments with:
- Sparse, outcome-only rewards
- Long horizons
- Jagged reward landscapes (non-smooth reward functions)

## Environments

### 1. Simple GridWorld
- **State**: (x, y) position on an N×N grid (one-hot encoded)
- **Actions**: {up, down, left, right}
- **Goal**: Navigate from bottom-left to top-right corner
- **Obstacles**: Random obstacles that block movement
- **Rewards**:
  - +1.0 for reaching goal
  - -0.1 for hitting obstacles
  - 0.0 otherwise
- **Sparse reward**: Only at goal (long-horizon credit assignment problem)

### 2. Harder GridWorld (Key-Door)
- All features of Simple GridWorld, plus:
- **Key**: Must collect key before reaching goal
- **Locked goal**: Reaching goal without key gives penalty
- **Even longer horizon**: Must collect key → navigate to goal

## Methods

### Random Baseline
- Uniform random action selection
- Provides lower bound on performance

### Evolution Strategies (ES)
- **Parameter-space optimization**: Directly perturb policy network weights
- **Zeroth-order gradient estimate**: No backpropagation needed
- **Algorithm**:
  1. Sample N perturbations of policy parameters: θ' = θ + σε
  2. Evaluate each perturbed policy on environment
  3. Estimate gradient: ∇J ≈ (1/Nσ) Σ R(θ + σε) · ε
  4. Update parameters: θ ← θ + α∇J

### PPO (Proximal Policy Optimization)
- **Action-space RL**: Standard policy gradient method
- **Uses value function** for variance reduction (GAE)
- **Clipped objective** for stable updates
- Baseline comparison to ES

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements_gridworld.txt

# Or just core dependencies
pip install torch numpy matplotlib
```

### Run Individual Methods

```bash
# Test environment
python gridworld_env.py

# Train ES
python train_es_gridworld.py

# Train PPO
python train_ppo_gridworld.py
```

### Run Full Comparison (Proof of Life)

```bash
# Compare all methods on simple gridworld (3 trials, 100 iterations each)
python compare_methods.py --env simple --trials 3 --iterations 100

# Run on harder environment
python compare_methods.py --env harder --trials 3 --iterations 200

# Customize grid size and obstacles
python compare_methods.py --env simple --size 10 --obstacles 12 --trials 5
```

## Expected Results

Based on the sparse reward setting:

1. **Random policy**: Very low success rate (~0-5%), negative average reward
2. **ES**: Should show stable improvement, moderate success rate (30-60% depending on difficulty)
3. **PPO**: May struggle initially with sparse rewards but should eventually learn (20-50% success)

The key comparison metric is **learning stability** (variance across trials) and **sample efficiency** (reward vs iterations).

## Project Structure

```
tiny-grpo-es/
├── gridworld_env.py           # GridWorld environments
├── policy_network.py          # Policy and value networks
├── train_es_gridworld.py      # ES training loop
├── train_ppo_gridworld.py     # PPO training loop
├── compare_methods.py         # Main experiment script
├── requirements_gridworld.txt # Dependencies
└── README_GRIDWORLD.md        # This file

# Original GRPO/ES code for LLMs
├── train_es.py                # ES for LLM training
├── train.py                   # GRPO for LLM training
├── loss.py                    # Loss functions
└── data/math_tasks.jsonl      # Math dataset
```

## Key Parameters

### ES Hyperparameters
- `N`: Population size (number of perturbations) - default: 20
- `sigma`: Noise scale for perturbations - default: 0.05
- `alpha`: Learning rate - default: 0.01
- `T`: Number of iterations - default: 100-200

### PPO Hyperparameters
- `n_steps`: Steps per rollout - default: 128
- `n_epochs`: Optimization epochs per iteration - default: 4
- `gamma`: Discount factor - default: 0.99
- `clip_epsilon`: PPO clip range - default: 0.2
- `lr_policy`: Policy learning rate - default: 3e-4

## Experiments & Analysis

### Week 1 Proof of Life

**Objective**: Validate that ES can learn in sparse reward gridworld

**Experiment**:
```bash
python compare_methods.py --env simple --trials 5 --iterations 100 --output ./results/week1
```

**Success criteria**:
1. ES achieves >30% success rate on 8×8 gridworld
2. ES shows lower variance than PPO across trials
3. Both methods significantly outperform random baseline

**Deliverable**: 
- Learning curves (reward vs iterations)
- Final performance comparison (box plots)
- Analysis of stability (std across trials)

## Future Directions

1. **More complex environments**:
   - Tower defense games
   - Symbolic reasoning tasks
   - Multi-agent coordination

2. **Hybrid approaches**:
   - Combine ES and PPO (ES for exploration, PPO for exploitation)
   - Adaptive sigma scheduling

3. **Scalability**:
   - Larger networks
   - Distributed ES
   - GPU acceleration

## References

1. [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) - Salimans et al., 2017
2. [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
3. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) - DeepSeek, 2025
4. Original repo: [tiny-grpo-es](https://github.com/damek/tiny-grpo-es)

## Authors

STAT 4830 Project - Parameter Space Optimization for Non-Differentiable RL

## License

Apache 2.0 (inherited from original tiny-grpo-es repository)
