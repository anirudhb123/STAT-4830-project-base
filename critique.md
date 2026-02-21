# Week 4 Self-Critique

## ORIENT

### Strengths
- **Clear Motivation:** The report effectively articulates why ES is a relevant candidate for sparse reward settings where gradient-based methods struggle, supported by a clear mathematical formulation.
- **Working Implementation:** Both ES and PPO successfully converge to 100% success rate, demonstrating that the implementations are correct and functional.
- **Fair Comparison Framework:** The notebook implements a proper comparison where both methods train on shaped rewards and evaluate on sparse rewards, ensuring an apples-to-apples comparison.
- **Code Structure:** The codebase is well-organized into modular components (`model.py`, `utils.py`, `ppo_training.py`) with 19 passing unit tests covering all major components.
- **Reproducibility:** Fixed seeds, documented hyperparameters, and clear implementation make results reproducible.

### Areas for Improvement
- **Task Difficulty Too Low:** With reward shaping (+0.2 for moving closer to goal), both ES and PPO achieve 100% success, making it impossible to differentiate their performance or identify strengths/weaknesses of each approach.
- **Limited Generalization Testing:** Training and evaluation use the same obstacle configuration (seed=123), so we don't know if policies generalize to unseen obstacle placements.
- **No Variance Analysis:** Results use fixed seeds without multiple trials. We lack error bars and statistical significance tests (e.g., Cohen's d) to make rigorous claims about method comparison.
- **Insufficient Sample Efficiency Analysis:** While both methods converge, we haven't analyzed how many environment interactions each requires or which is more sample-efficient.
- **Scalability Untested:** The 8×8 grid with one-hot encoding (64-dim) is small. We haven't tested larger grids, deeper networks, or more challenging variants like `HarderGridWorld`.

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
