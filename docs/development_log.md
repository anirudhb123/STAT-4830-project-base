# Development Log

## Week 4 (Jan 27 - Feb 6, 2026)

### Overview
Implemented Evolution Strategies (ES) for parameter-space optimization in sparse reward reinforcement learning. Compared with PPO on GridWorld environments.

### Key Decisions

**1. Problem Selection (Jan 27-28)**
- **Decision:** Focus on sparse reward RL with ES vs PPO comparison
- **Rationale:** ES is theoretically interesting for non-differentiable settings, and sparse rewards provide a clear test case where gradient-based methods struggle
- **Alternatives considered:** Multi-armed bandits (too simple), continuous control (too complex for Week 4)

**2. Environment Design (Jan 29)**
- **Decision:** Use GridWorld with one-hot state encoding
- **Rationale:** Simple, interpretable, and allows precise control over difficulty
- **Implementation details:**
  - 8×8 grid with 8 obstacles
  - +1 reward at goal, -0.1 at obstacles, 0 elsewhere (sparse!)
  - One-hot encoding (64-dim state space)
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
- **Code:** `fitness_normalized = (fitness - mean) / std`
- **Impact:** Reduced gradient variance by ~30% (qualitative observation)

**5. PPO Baseline (Feb 1-2)**
- **Decision:** Implement PPO for comparison
- **Hyperparameters:**
  - Learning rate: 3e-4
  - Clip epsilon: 0.2
  - GAE lambda: 0.95
- **Challenge:** Value function struggled with sparse rewards initially
- **Solution:** Increased number of rollout steps from 64 to 128

### Failed Attempts

**1. Large Noise Scale (Jan 30)**
- **Attempt:** Started with σ=0.1 (from literature for continuous control)
- **Result:** Complete divergence after 20 iterations
- **Lesson:** GridWorld needs smaller perturbations due to discrete actions
- **Fix:** Reduced to σ=0.05

**2. Single Episode Evaluation (Jan 31)**
- **Attempt:** Evaluate each perturbation on 1 episode (for speed)
- **Result:** Very noisy gradient estimates, unstable learning
- **Lesson:** Environment stochasticity (random obstacles) requires multiple episodes
- **Fix:** Increased to 5 episodes per evaluation

**3. Matplotlib Backend Issues (Feb 3)**
- **Attempt:** Use default matplotlib backend for visualization
- **Result:** Crashes on headless server
- **Lesson:** Always set backend explicitly for server environments
- **Fix:** Added `matplotlib.use('Agg')` and save to file

### Testing Strategy

**Unit Tests:**
1. Environment mechanics (collisions, rewards, termination)
2. Policy network forward pass
3. ES gradient shape and validity

**Integration Tests:**
1. Full training loop runs without errors
2. Policy improves on empty grid (sanity check)
3. Comparison script completes

**Performance Tests:**
1. 3 trials × 3 methods on 8×8 grid
2. Statistical comparison (mean ± std)
3. Visualization of learning curves

### Results Summary

**Simple GridWorld (8×8, 8 obstacles):**
- Random: 2.3% ± 1.1% success
- ES: 42.7% ± 8.3% success
- PPO: 38.5% ± 12.1% success

**Key Findings:**
- ✅ ES slightly outperforms PPO on average
- ✅ ES shows lower variance across trials (more stable)
- ✅ Both methods significantly beat random baseline

### Open Questions

1. **Why is ES more stable than PPO?**
   - Hypothesis: Parameter-space exploration is more robust to sparse rewards
   - Need: Theoretical justification or ablation study

2. **Does this hold for harder tasks?**
   - Next: Test on Key-Door gridworld (multi-stage task)
   - Concern: ES may not scale to higher dimensions

3. **Sample efficiency?**
   - Current: ES uses 100 episodes/iteration (20 perturbations × 5 episodes)
   - PPO uses 128 steps/iteration (fewer full episodes)
   - Need: Fair comparison of sample complexity

### Resource Usage

**Computational:**
- Training time: ~8 min/method/trial (CPU only)
- Memory: ~200MB peak
- No GPU required for small networks

**Development Time:**
- Environment: 4 hours
- ES implementation: 6 hours
- PPO implementation: 8 hours
- Testing & debugging: 6 hours
- Documentation: 4 hours
- **Total: ~28 hours**

### Code Organization Evolution

**Initial (Jan 27):**
- Single file `train.py` with everything

**Iteration 1 (Jan 30):**
- Split into `gridworld_env.py`, `policy_network.py`, `train_es.py`

**Final (Feb 5):**
- Proper structure: `src/`, `tests/`, `notebooks/`
- Modular: `model.py` (env + networks), `utils.py` (training functions)

### LLM Usage Log

**ChatGPT:**
- Helped debug ES gradient estimation (Jan 30)
- Suggested reward standardization (Jan 31)
- Reviewed PPO GAE implementation (Feb 2)

**Cursor:**
- Used for refactoring code structure (Feb 5)
- Code completion for boilerplate (throughout)

**Claude:**
- Asked about theoretical justification for ES vs PPO (Feb 3)
- Got references to relevant papers

### Next Week Plan

**Immediate (Week 5):**
1. Run full comparison on HarderGridWorld
2. Implement hyperparameter sensitivity analysis
3. Add statistical significance tests
4. Create better visualizations

**Technical Improvements:**
1. Parallel ES evaluation (multiprocessing)
2. Mirrored sampling for variance reduction
3. Adaptive sigma scheduling

**Exploration:**
1. Test on larger grids (12×12, 16×16)
2. Try Natural ES
3. Ablation studies (with/without fitness standardization)

### Lessons Learned

1. **Start simple:** 8×8 grid was perfect for Week 4. Didn't need to jump to complex environments immediately.

2. **Test incrementally:** Testing on empty grid first caught many bugs early.

3. **Hyperparameters matter:** Spent 2 days finding good σ and α values. Should have done grid search from start.

4. **Documentation while coding:** Wrote docstrings as I coded - saved time later.

5. **Multiple trials essential:** Single-trial results were misleading due to variance.

### References Used

1. [Salimans et al., 2017] - Evolution Strategies as a Scalable Alternative to Reinforcement Learning
2. [Schulman et al., 2017] - Proximal Policy Optimization
3. [Mania et al., 2018] - Simple random search provides a competitive approach to reinforcement learning
4. OpenAI Spinning Up documentation for PPO implementation details

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
