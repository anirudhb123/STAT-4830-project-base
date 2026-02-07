# Evolution Strategies for Sparse Reward Reinforcement Learning

## Problem Statement

**What are we optimizing?** We are training a neural network policy to navigate an 8×8 GridWorld with obstacles using Evolution Strategies (ES), a gradient-free optimization method. The agent starts in the bottom-left corner and must reach a goal in the top-right corner while avoiding 8 randomly placed obstacles. The environment provides sparse rewards: +1 only upon reaching the goal, -0.1 for hitting an obstacle, and 0 otherwise. The policy network (a 2-layer MLP with 64 hidden units, totaling 8,580 parameters) maps one-hot encoded grid positions to a probability distribution over four actions (up, down, left, right).

**Why does this problem matter?** Sparse reward settings are notoriously difficult for standard policy gradient methods like REINFORCE or PPO, because the learning signal is nearly zero until the agent stumbles upon the goal by chance. ES sidesteps this by estimating gradients through random parameter perturbations evaluated on total episode fitness, requiring no backpropagation through the environment. This makes ES a compelling alternative in non-differentiable or sparse-signal settings, as demonstrated by Salimans et al. (2017).

**How will we measure success?** Our primary metric is *success rate*: the fraction of evaluation episodes in which the agent reaches the goal. Secondary metrics include average episode reward, average steps to goal, and gradient norm over training. We will compare ES against a random baseline and, in future work, against PPO.

**Constraints and risks.** The one-hot state encoding scales as O(n²) and becomes impractical beyond roughly 20×20 grids. ES is known to require large populations for high-dimensional parameter spaces, so scaling to deeper networks may demand significant compute. Additionally, random obstacle placement introduces environment stochasticity, meaning a single evaluation episode per perturbation yields noisy gradient estimates.

## Technical Approach

**Mathematical formulation.** We maximize the expected cumulative reward J(θ) = E[Σ_t r_t] over policy parameters θ. ES approximates the gradient of a smoothed objective:

∇_θ J(θ) ≈ (1 / Nσ) Σ_{i=1}^{N} F(θ + σε_i) · ε_i

where ε_i ~ N(0, I) are Gaussian perturbations, σ is the noise scale, N is the population size, and F is the fitness (average episode reward) of the perturbed policy. We apply fitness standardization — F_normalized = (F - mean) / std — to reduce gradient variance when fitness scales vary.

**Algorithm and hyperparameters.** We chose vanilla ES as the simplest baseline before considering CMA-ES or Natural ES. Hyperparameters were tuned through manual experimentation:

| Parameter | Value | Rationale |
|---|---|---|
| Population size N | 20 | Sufficient gradient quality for small parameter count |
| Noise scale σ | 0.05 | σ=0.1 caused divergence; σ=0.01 was too slow |
| Learning rate α | 0.01 | Conservative to avoid instability |
| Episodes per evaluation | 5 | Reduces variance from random obstacle placement |
| Max steps per episode | 50 | Enough for optimal ~14-step path on 8×8 grid |

For the PPO baseline, we implemented a standard PPO-Clip algorithm with Generalized Advantage Estimation (GAE). Key hyperparameters include clipping $\epsilon=0.2$, GAE $\lambda=0.95$, discount $\gamma=0.99$, and an entropy coefficient of 0.01 to encourage exploration. The implementation uses separate optimizers for the policy and value networks.

**Implementation strategy.** All code uses PyTorch for network definitions and NumPy for environment logic. The codebase is organized as:

- `src/model.py` — `GridWorld` and `HarderGridWorld` environments, `PolicyNetwork` (state → action probabilities), and `ValueNetwork` (value function approximation)
- `src/utils.py` — `es_gradient_estimate` (core ES loop), `train_es` (full training pipeline), `evaluate_policy`, `plot_training_curves`, and `compute_statistics`
- `src/ppo_training.py` — `train_ppo` (PPO training loop), `RolloutBuffer`, and GAE computation
- `src/__init__.py` — Clean package exports

The `PolicyNetwork` uses orthogonal weight initialization and supports both stochastic and deterministic action selection. A `get_action_batch` method is included to support the PPO training loop. The `ValueNetwork` shares the same 2-layer MLP architecture and is used for PPO's advantage estimation.

**Validation methods.** We validate through unit tests (environment mechanics, network forward pass shapes, ES gradient shape), integration tests (full training loop execution, evaluation pipeline), and edge case tests (empty grid, dense obstacles, larger 12×12 grid).

## Initial Results

**Quick validation run (20 ES iterations, 8×8 grid, 8 obstacles):**

| Metric | Value |
|---|---|
| Success rate | 0% |
| Gradient norm | ~400 (remained high throughout) |
| Time per iteration | ~0.65 seconds |
| Peak memory | ~127 MB |
| Model parameters | 8,580 (0.033 MB) |
| Estimated 100-iteration training | ~1.1 minutes (CPU) |

The policy did not converge in 20 iterations. Gradient norms remained high, indicating the policy is still in early exploration and has not found reward signal. This is not unexpected — 20 iterations with population size 20 means only 400 total perturbation evaluations, which may be insufficient to discover a successful trajectory in a sparse reward landscape.

**Test case results.** Environment mechanics tests confirmed correct collision handling, reward assignment, and episode termination. The policy network forward pass produces valid probability distributions (correct shape, sums to 1, non-negative). The ES gradient estimator returns gradients of the correct shape. One edge case test revealed a potential bug in goal detection logic for the action mapping test, which is under investigation.

**Current limitations.** The most significant limitation is that we have not yet demonstrated learning — the 20-iteration run was a quick validation, not a convergence experiment. Additionally, the `HarderGridWorld` (key-door variant) has been built but not trained on. While the PPO training infrastructure is now implemented, full comparative experiments against ES have not yet been executed.

**Resource usage.** Training is CPU-only and lightweight. A full 100-iteration run is estimated at ~1.1 minutes, making hyperparameter sweeps feasible on a laptop. No GPU is required for the current network size.

## Next Steps

**Immediate priorities (Week 5):**

1. **Run full training experiments.** Execute 100+ iteration ES training runs and verify whether the policy converges on the 8×8 grid. If not, perform a hyperparameter grid search over σ ∈ {0.01, 0.03, 0.05, 0.1}, α ∈ {0.005, 0.01, 0.05}, and N ∈ {20, 50, 100}.
2. **Debug environment test failure.** The action mapping unit test flagged a potential issue in goal detection logic within `GridWorld.step`. This needs to be resolved to ensure environment correctness.
3. **Run PPO baseline experiments.** Execute the newly implemented `train_ppo` loop to establish a gradient-based baseline. We will compare convergence speed and final success rates against ES.
4. **Multi-trial evaluation.** Once ES converges, run 5–10 independent trials (different seeds) and report mean ± std. Use the existing `compute_statistics` and `print_comparison_table` utilities. Compute effect sizes (Cohen's d) for ES vs. random baseline.

**Technical improvements:**

- Mirrored sampling (evaluate both +ε and -ε) for variance reduction in ES gradient estimates
- Adaptive sigma scheduling to balance exploration and exploitation over training
- Parallel perturbation evaluation via multiprocessing to speed up training

**Alternative approaches to explore:**

- Natural ES or CMA-ES for potentially faster convergence
- Test ES on `HarderGridWorld` (key-door task) to evaluate scalability to multi-stage objectives
- Larger grids (12×12, 16×16) to stress-test the one-hot encoding approach

**What we have learned so far:**

1. Hyperparameter sensitivity is real — σ=0.1 vs. σ=0.05 was the difference between divergence and stable training.
2. Environment stochasticity demands multiple evaluation episodes per perturbation; single-episode evaluations produced unusable gradient estimates.
3. Fitness standardization is critical for ES stability, even on simple problems.
4. Starting with the simplest possible environment (8×8 grid) was the right decision — it lets us iterate quickly and isolate algorithm-level issues before scaling up.
5. 20 iterations is not enough to draw conclusions about ES performance in sparse reward settings; longer experiments are essential before making claims about convergence.

## References

1. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. *arXiv:1703.03864*.
2. Mania, H., Guy, A., & Recht, B. (2018). Simple random search provides a competitive approach to reinforcement learning. *NeurIPS*.
3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.
