# Development Log

## Week 4 (Jan 27 - Feb 6, 2026)

### Overview
Implemented Evolution Strategies (ES) for parameter-space optimization in sparse reward reinforcement learning. Built a GridWorld environment and ES training pipeline from scratch in `src/`.

### Key Decisions

**1. Problem Selection (Jan 27-28)**
- **Decision:** Focus on sparse reward RL using ES as the optimization method
- **Rationale:** ES is theoretically interesting for non-differentiable settings, and sparse rewards provide a clear test case where gradient-based methods struggle
- **Alternatives considered:** Multi-armed bandits (too simple), continuous control (too complex for Week 4)

**2. Environment Design (Jan 29)**
- **Decision:** Use GridWorld with one-hot state encoding
- **Rationale:** Simple, interpretable, and allows precise control over difficulty
- **Implementation details:**
  - 8×8 grid with 8 obstacles (`GridWorld` class in `src/model.py`)
  - +1 reward at goal, -0.1 at obstacles, 0 elsewhere (sparse!)
  - One-hot encoding (64-dim state space)
  - Also built `HarderGridWorld` variant with key-door mechanics for future multi-stage experiments
- **Trade-offs:** One-hot doesn't scale beyond ~20×20, but sufficient for proof-of-concept

**3. ES Algorithm (Jan 30)**
- **Decision:** Vanilla ES with Gaussian perturbations
- **Hyperparameters chosen:**
  - Population size N=20 (balance between gradient quality and compute)
  - Noise scale σ=0.05 (found through trial - 0.1 was too large, 0.01 too slow)
  - Learning rate α=0.01 (conservative to avoid instability)
- **Why not CMA-ES or Natural ES?** Wanted simplest baseline first; can extend later

**4. Reward Standardization (Jan 31)**
- **Decision:** Standardize fitness values in ES gradient estimation
- **Rationale:** Improves stability when fitness scales vary
- **Code:** `fitness_normalized = (fitness - mean) / std` (in `es_gradient_estimate` in `src/utils.py`)
- **Impact:** Reduced gradient variance qualitatively compared to unnormalized version

**5. Network Architecture (Feb 1-2)**
- **Decision:** Build both `PolicyNetwork` and `ValueNetwork` in `src/model.py`
- **PolicyNetwork:** 2-layer MLP (64 hidden units), maps state → action probabilities
  - Orthogonal weight initialization for stable training
  - Supports both stochastic and deterministic action selection
  - Includes `get_action_batch` for potential future PPO integration
- **ValueNetwork:** Same architecture, maps state → scalar value estimate
  - Scaffolded for planned PPO comparison (not yet implemented)

### Failed Attempts

**1. Large Noise Scale (Jan 30)**
- **Attempt:** Started with σ=0.1 (from literature for continuous control)
- **Result:** Complete divergence after 20 iterations
- **Lesson:** GridWorld needs smaller perturbations due to discrete actions
- **Fix:** Reduced to σ=0.05

**2. Single Episode Evaluation (Jan 31)**
- **Attempt:** Evaluate each perturbation on 1 episode (for speed)
- **Result:** Very noisy gradient estimates, unstable learning
- **Lesson:** Environment stochasticity (random obstacle placement) requires multiple episodes
- **Fix:** Increased to 5 episodes per evaluation

**3. Matplotlib Backend Issues (Feb 3)**
- **Attempt:** Use default matplotlib backend for visualization
- **Result:** Crashes on headless server
- **Lesson:** Always set backend explicitly for server environments
- **Fix:** Added `matplotlib.use('Agg')` at the top of `src/model.py` and save to file

**4. Insufficient Training Iterations (Feb 4)**
- **Attempt:** Initial validation run with only 20 ES iterations in the notebook
- **Result:** 0% success rate — policy did not converge
- **Lesson:** 20 iterations is not enough for ES on an 8×8 grid with 8 obstacles; the default `train_es` uses 100 iterations, and even that may need tuning
- **Next step:** Run longer training and investigate whether hyperparameters need adjustment

### Testing Strategy

**Unit Tests (in notebook):**
1. Environment mechanics (collisions, rewards, termination)
2. Policy network forward pass (shape, softmax validity)
3. ES gradient shape and validity

**Integration Tests:**
1. Full ES training loop runs without errors
2. Evaluation pipeline produces metrics correctly

**Edge Case Tests (in notebook):**
1. Empty grid (no obstacles) — tested policy generalization
2. Dense obstacles (15 on 8×8) — tested robustness
3. Larger grid (12×12) — tested scalability of state representation

**Note:** Environment mechanics test revealed a bug in the action-to-goal navigation test (assertion error on goal detection), which needs to be investigated further.

### Initial Results

**Quick validation run (20 ES iterations on 8×8 grid, 8 obstacles):**
- Success rate: 0% — policy did not converge in 20 iterations
- Gradient norms remained high (~400) throughout, suggesting the policy is still in early exploration

**Resource measurements (single ES iteration):**
- Time: ~0.65 seconds per iteration
- Memory: ~127 MB
- Model size: 8,580 parameters (0.033 MB)
- Estimated full 100-iteration training: ~1.1 minutes on CPU

**Honest assessment:** The ES implementation runs end-to-end but has not yet demonstrated learning on the 8×8 obstacle grid within 20 iterations. Longer training runs and hyperparameter tuning are needed to validate convergence.

### Open Questions

1. **Why hasn't ES converged in 20 iterations?**
   - Hypothesis: 8×8 with 8 obstacles may need more iterations, or hyperparameters (σ, α, N) need further tuning
   - Next: Run full 100+ iteration experiments and perform hyperparameter grid search

2. **Does the environment test bug indicate a logic issue?**
   - The action mapping test in the notebook failed on goal detection
   - Need: Debug the GridWorld step logic for edge cases around goal position

3. **How will ES perform on harder tasks?**
   - Built `HarderGridWorld` with key-door mechanics but haven't tested training on it yet
   - Concern: ES may not scale to higher dimensions or multi-stage objectives

4. **PPO comparison still needed**
   - Scaffolded `ValueNetwork` and `get_action_batch` for PPO but have not yet implemented a PPO training loop
   - Need: Implement `train_ppo` in `src/utils.py` for a fair comparison

### Resource Usage

**Computational:**
- Single ES iteration: ~0.65 seconds (CPU only)
- Memory: ~127 MB peak
- No GPU required for small networks

**Development Time:**
- Environment (`GridWorld`, `HarderGridWorld`): 4 hours
- Network architecture (`PolicyNetwork`, `ValueNetwork`): 3 hours
- ES implementation (`es_gradient_estimate`, `train_es`): 6 hours
- Utilities (evaluation, plotting, statistics): 3 hours
- Notebook validation & testing: 4 hours
- Documentation: 3 hours
- **Total: ~23 hours**

### Code Organization Evolution

**Initial (Jan 27):**
- Single file exploring ideas

**Iteration 1 (Jan 30):**
- Split into separate environment and training files

**Final (Feb 5):**
- `src/model.py` — Environments (`GridWorld`, `HarderGridWorld`) + Networks (`PolicyNetwork`, `ValueNetwork`)
- `src/utils.py` — Training (`train_es`, `es_gradient_estimate`) + Evaluation (`evaluate_policy`, `plot_training_curves`, `compute_statistics`)
- `src/__init__.py` — Clean package exports
- `notebooks/week4_implementation.ipynb` — End-to-end validation

### LLM Usage Log

**ChatGPT:**
- Helped debug ES gradient estimation (Jan 30)
- Suggested reward standardization approach (Jan 31)

**Cursor:**
- Used for refactoring code into `src/` structure (Feb 5)
- Code completion for boilerplate (throughout)

**Claude:**
- Asked about theoretical justification for ES in sparse reward settings (Feb 3)
- Got references to relevant papers (Salimans et al., Mania et al.)

### Next Week Plan

**Immediate (Week 5):**
1. Debug environment mechanics test failure
2. Run full 100+ iteration ES training and verify convergence
3. Implement PPO training loop in `src/utils.py` for comparison
4. Tune hyperparameters (grid search over σ, α, N)

**Technical Improvements:**
1. Parallel ES evaluation (multiprocessing)
2. Mirrored sampling for variance reduction
3. Adaptive sigma scheduling

**Exploration:**
1. Test ES on `HarderGridWorld` (key-door task)
2. Test on larger grids (12×12, 16×16)
3. Try Natural ES or CMA-ES variants
4. Ablation study: with/without fitness standardization

### Lessons Learned

1. **Start simple:** 8×8 grid was the right scope for Week 4. Complex environments can come later once the base pipeline works.

2. **Test incrementally:** Running a quick 20-iteration validation caught issues early, even though it wasn't long enough for convergence.

3. **Hyperparameters matter:** Spent time finding workable σ and α values through trial and error. Should formalize this with a grid search.

4. **Documentation while coding:** Writing docstrings as we coded saved time when assembling the notebook and log.

5. **Be honest about results:** Initial results show no convergence yet — this is valuable information for guiding next steps rather than something to hide.

### References Used

1. [Salimans et al., 2017] - Evolution Strategies as a Scalable Alternative to Reinforcement Learning
2. [Mania et al., 2018] - Simple random search provides a competitive approach to reinforcement learning
3. [Schulman et al., 2017] - Proximal Policy Optimization (for future PPO implementation)

---

## Week 3 (Jan 20-26, 2026)

### Project Exploration
- Explored multiple project ideas: LLM fine-tuning, portfolio optimization, ES for RL
- Decided on ES for sparse reward RL after reading DeepSeek-R1 paper
- Set up development environment (Cursor, GitHub, virtual env)

---

## Week 2 (Jan 13-19, 2026)

### Initial Setup
- Forked repository template
- Set up GitHub collaboration with team
- Read through course materials and past projects
- Brainstormed project ideas

---

*Log updated: February 6, 2026*
