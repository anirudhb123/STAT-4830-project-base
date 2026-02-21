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
We are assuming that the 100% success rate is meaningful rather than an artifact of an overly easy task. The current results suggest reward shaping may provide too much guidance, potentially masking important differences between ES and PPO that would emerge on harder problems. There is a risk that our positive results are not generalizable to truly sparse reward settings or more complex environments.

## DECIDE

### Concrete Next Actions
1. **Test on Harder Environments:** Remove reward shaping and train on pure sparse rewards, or increase difficulty (larger grids, more obstacles, `HarderGridWorld` key-door variant) to create a more discriminative comparison.
2. **Multi-Seed Evaluation:** Run 5-10 trials with different random seeds to compute mean ± std for success rates and rewards, enabling statistical significance testing.
3. **Generalization Testing:** Evaluate trained policies on environments with different obstacle configurations to test if learning generalizes beyond the training seed.
4. **Sample Efficiency Analysis:** Track cumulative environment interactions for both methods and plot learning curves to identify which converges faster in terms of total samples (not just iterations).

## ACT

### Resource Needs
The current CPU-only setup remains sufficient for immediate experiments (80 iterations completes in 2-3 minutes). For scaling to larger grids or multi-seed trials, parallel execution across multiple cores or a compute cluster would accelerate experimentation. No GPU is required for current network sizes, though convolutional architectures for larger grids might benefit from GPU acceleration.
