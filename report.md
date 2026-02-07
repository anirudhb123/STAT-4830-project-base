# Evolution Strategies for Sparse Reward Reinforcement Learning

## Problem Statement

**What are we optimizing?** We are training a neural network policy to navigate an 8×8 GridWorld with obstacles using Evolution Strategies (ES), a gradient-free optimization method. The agent starts in the bottom-left corner and must reach a goal in the top-right corner while avoiding 8 randomly placed obstacles. The environment provides sparse rewards: +1 only upon reaching the goal, -0.1 for hitting an obstacle, and 0 otherwise. The policy network (a 2-layer MLP with 64 hidden units, totaling 8,580 parameters) maps one-hot encoded grid positions to a probability distribution over four actions (up, down, left, right).

**Why does this problem matter?** Sparse reward settings are notoriously difficult for standard policy gradient methods like REINFORCE or PPO, because the learning signal is nearly zero until the agent stumbles upon the goal by chance. ES sidesteps this by estimating gradients through random parameter perturbations evaluated on total episode fitness, requiring no backpropagation through the environment. This makes ES a compelling alternative in non-differentiable or sparse-signal settings, as demonstrated by Salimans et al. (2017).

**How will we measure success?** Our primary metric is *success rate*: the fraction of evaluation episodes in which the agent reaches the goal. Secondary metrics include average episode reward, average steps to goal, and gradient norm over training. We compare ES against a PPO baseline.

**Constraints and risks.** The one-hot state encoding scales as O(n²) and becomes impractical beyond roughly 20×20 grids. ES is known to require large populations for high-dimensional parameter spaces, so scaling to deeper networks may demand significant compute. Additionally, random obstacle placement introduces environment stochasticity, meaning a single evaluation episode per perturbation yields noisy gradient estimates.

## Technical Approach

**Mathematical formulation.** We maximize the expected cumulative reward J(θ) = E[Σ_t r_t] over policy parameters θ. ES approximates the gradient of a smoothed objective:

∇_θ J(θ) ≈ (1 / Nσ) Σ_{i=1}^{N} F(θ + σε_i) · ε_i

where ε_i ~ N(0, I) are Gaussian perturbations, σ is the noise scale, N is the population size, and F is the fitness (average episode reward) of the perturbed policy. We apply fitness standardization (F_normalized = (F - mean) / std) to reduce gradient variance when fitness scales vary.

**Algorithm and hyperparameters.** We chose vanilla ES as the simplest baseline before considering CMA-ES or Natural ES. Hyperparameters were tuned through manual experimentation:

| Parameter | Value | Rationale |
|---|---|---|
| Population size N | 50 | Larger population for stable gradient estimates |
| Noise scale σ | 0.1 | Balanced exploration-exploitation |
| Learning rate α | 0.05 | Aggressive updates work well with fitness standardization |
| Episodes per evaluation | 5 | Reduces variance from random obstacle placement |
| Max steps per episode | 50 | Enough for optimal ~14-step path on 8×8 grid |

For the PPO baseline, we implemented a standard PPO-Clip algorithm with Generalized Advantage Estimation (GAE). Key hyperparameters include clipping $\epsilon=0.2$, GAE $\lambda=0.95$, discount $\gamma=0.99$, and an entropy coefficient of 0.01 to encourage exploration. The implementation uses separate optimizers for the policy and value networks.

**Implementation strategy.** All code uses PyTorch for network definitions and NumPy for environment logic. The codebase is organized as:

- `src/model.py`: `GridWorld` and `HarderGridWorld` environments, `PolicyNetwork` (state → action probabilities), and `ValueNetwork` (value function approximation)
- `src/utils.py`: `es_gradient_estimate` (core ES loop), `train_es` (full training pipeline), `evaluate_policy`, `plot_training_curves`, and `compute_statistics`
- `src/ppo_training.py`: `train_ppo` (PPO training loop), `RolloutBuffer`, and GAE computation
- `notebooks/week4_implementation.ipynb`: Full working comparison

The `PolicyNetwork` uses orthogonal weight initialization and supports both stochastic and deterministic action selection. A `get_action_batch` method is included to support the PPO training loop. The `ValueNetwork` shares the same 2-layer MLP architecture and is used for PPO's advantage estimation.

**Validation methods.** We validate through unit tests (environment mechanics, network forward pass shapes, ES gradient shape), integration tests (full training loop execution, evaluation pipeline), and edge case tests (empty grid, dense obstacles, larger 12×12 grid).

## Initial Results

**Full training run (80 iterations, 8×8 grid, 8 obstacles):**

| Metric | Value |
|---|---|
| Success rate | 100% (with reward shaping) |
| Time per iteration | ~1-2 seconds |
| Peak memory | ~150 MB |
| Model parameters | 8,580 (0.033 MB) |
| Total training time | ~2-3 minutes (CPU) |

The policy successfully converged in 80 iterations when trained with reward shaping (+0.2 for moving closer to goal, -0.01 step penalty). Both ES and PPO achieved 100% success rate on the shaped reward environment. When evaluated on sparse rewards only (+1 at goal, 0 elsewhere), both methods maintained 100% success, demonstrating that the learned policies generalize to the sparse setting.

**Test case results.** Environment mechanics tests confirmed correct collision handling, reward assignment, and episode termination. The policy network forward pass produces valid probability distributions (correct shape, sums to 1, non-negative). The ES gradient estimator returns gradients of the correct shape. All 19 unit tests pass.

**Current limitations.** With reward shaping, the task may be too easy (both ES and PPO reach 100% success). Future work should test harder configurations or pure sparse rewards to better differentiate method performance.

**Resource usage.** Training is CPU-only and lightweight. A full 80-iteration run completes in 2-3 minutes, making hyperparameter sweeps feasible on a laptop. No GPU is required for the current network size.

## Next Steps

**Immediate priorities (Week 5):**

1. **Test harder environments.** The current setup (8×8 grid with reward shaping) proved too easy, as both ES and PPO achieve 100% success. Next steps: (a) train on pure sparse rewards without shaping, (b) increase grid size to 12×12 or 16×16, (c) add more obstacles or use `HarderGridWorld` (key-door variant).
2. **Multi-trial evaluation.** Run 5–10 independent trials (different seeds) and report mean ± std. Use the existing `compute_statistics` and `print_comparison_table` utilities. Compute effect sizes (Cohen's d) for ES vs. PPO.
3. **Sample efficiency analysis.** Both methods reached 100% success, but we have not yet analyzed convergence speed or total environment interactions required.

**Technical improvements:**

- Mirrored sampling (evaluate both +ε and -ε) for variance reduction in ES gradient estimates
- Adaptive sigma scheduling to balance exploration and exploitation over training
- Parallel perturbation evaluation via multiprocessing to speed up training

**Alternative approaches to explore:**

- Natural ES or CMA-ES for potentially faster convergence
- Test ES on `HarderGridWorld` (key-door task) to evaluate scalability to multi-stage objectives
- Larger grids (12×12, 16×16) with convolutional encodings to replace one-hot states

**What we have learned so far:**

1. Hyperparameter sensitivity is real; σ=0.1 vs. σ=0.05 made a significant difference in stability and convergence.
2. Environment stochasticity demands multiple evaluation episodes per perturbation; single-episode evaluations produced unusable gradient estimates.
3. Fitness standardization is critical for ES stability, even on simple problems.
4. Reward shaping provides strong learning signal. Both ES and PPO converge quickly, perhaps too quickly to differentiate their performance.
5. Starting with the simplest possible environment (8×8 grid) was the right decision because it lets us iterate quickly and validate implementations before scaling up.

## References

1. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. *arXiv:1703.03864*.
2. Mania, H., Guy, A., & Recht, B. (2018). Simple random search provides a competitive approach to reinforcement learning. *NeurIPS*.
3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
