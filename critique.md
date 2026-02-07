# Week 4 Self-Critique

## ORIENT

### Strengths
- Clear motivation + correct framing: The writeup clearly motivates ES as parameter-space optimization that can work when action-space gradients are unreliable, and the report’s mathematical framing matches the implementation.

- Working end-to-end comparison (now empirical, not just theoretical): The notebook actually runs a fair ES vs PPO comparison: both train on shaped rewards and are evaluated on sparse rewards only (0/+1). In the shown run, both methods reach 100% success on the sparse evaluation environment.

- Modular code + extensibility: The separation into model.py, ppo_training.py, and utilities supports controlled experiments (swapping envs, reward functions, hyperparams) without refactoring.

- Reproducible setup: Fixed seeds and consistent environment settings (8×8, 8 obstacles, max 50 steps) make results easier to replicate and compare.

### Areas for Improvement
- **Unverified Baseline:** While the PPO training loop is described as implemented in the report, no comparative results are included, leaving the claim that "ES is a compelling alternative" theoretically supported but empirically untested.
- **Unresolved Bugs:** The report mentions a "potential bug in goal detection logic" flagged by unit tests; proceeding with long training runs before fixing this risks invalidating all future results.
- Training and evaluation use the same obstacle seed/config (seed=123 for both shaped training env and sparse eval env). That makes the current results less generalizable.
- Since both methods achieve 100% accuracy, the current experiment does not demonstrate an accuracy advantage of ES over PPO, only differences in convergence speed. We will need more challenging experiments to determine whether either method yields accuracy gains and to guide further improvements to ES.

### Critical Risks/Assumptions
We are assuming that the failure to converge is solely due to the low iteration count (20) rather than a fundamental issue with the sparse reward signal or the one-hot state encoding. There is a risk that vanilla ES without fitness shaping or curriculum learning may never find the goal on an 8x8 grid with obstacles given the sparsity of the signal.

## DECIDE

### Concrete Next Actions
1. **Debug Environment Logic:** Isolate and fix the goal detection failure in `GridWorld.step` flagged by the unit tests to ensure the reward signal is reliable before running long experiments.
2. **Execute Convergence Experiments:** Run 200-iteration training jobs with a hyperparameter sweep over noise scale ($\sigma \in \{0.03, 0.05, 0.1\}$) to find a configuration that achieves non-zero success.
3. **Establish PPO Baseline:** Run the newly implemented `train_ppo` function on the same seed to generate the first set of comparative learning curves and validate the PPO implementation.

## ACT

### Resource Needs
The current CPU-only setup is sufficient for these experiments (~1.1 mins per 100 iterations), so no new hardware is needed. I may need to consult standard PPO implementation references (like OpenAI Spinning Up) if the PPO baseline fails to learn, to ensure the hyperparameters (clipping, entropy coef) are set correctly for discrete gridworlds.
