# Week 7 LLM Conversation Log

**Date Range:** February 7 - February 25, 2026  
**Focus:** Rank-1 LoRA Evolution Strategies (LoRA-ES) on GridWorld + diagnosing “0 progress” in sparse reward training

---

## Session 1: Week 7 Problem Framing — Parameter-Efficient ES (Feb 7)

### Conversation with ChatGPT

**My Question:**
> I have ES working on GridWorld. For Week 7, I want to explore something “new”: can I reduce ES’s search dimensionality without changing the environment? What’s a clean way to do parameter-efficient ES?

**Response Summary:**
- Proposed adapter-style parameterizations (LoRA) as a way to constrain the ES search space
- Recommended starting with a very small rank (rank-1) to make the parameter-efficiency tradeoff obvious
- Suggested keeping a controlled comparison: same architecture and environment, only change what ES is allowed to update

**Key Takeaway:**
Rank-1 LoRA is a clean “knob” for ES: it shrinks the search space dramatically while keeping the same base network.

**What I Did:**
- Wrote Week 7 notebook scaffold to compare:
  - Standard ES (`param_mode='all'`)
  - LoRA-only ES (`param_mode='lora'`, base frozen)
- Planned to run both under the same three perturbation distributions: Gaussian / Cauchy / Laplace.

---

## Session 2: Implementing Rank-1 LoRA for the Policy Network (Feb 10)

### Conversation with Cursor AI

**My Request:**
> Help me add a rank-1 LoRA wrapper around `nn.Linear` layers and expose a clean way to select only the LoRA parameters for ES updates.

**Response Summary:**
- Suggested a wrapper module that computes:
  - \(W' x = Wx + \\alpha (x^\\top a) b\) where \(a\\in\\mathbb{R}^{d_{in}}, b\\in\\mathbb{R}^{d_{out}}\)
- Recommended freezing base weights when LoRA is enabled
- Recommended adding helper accessors like `lora_parameters()` and `count_lora_parameters()`

**Key Takeaway:**
If ES operates over flattened parameters, you need an explicit parameter-selection mechanism to swap between “all params” vs “LoRA params only”.

**What I Did:**
- Implemented rank-1 LoRA in the policy stack and added a LoRA-only parameter iterator.
- Verified parameter efficiency in `notebooks/week7_implementation.ipynb`:
  - Standard ES search parameters: **8580**
  - LoRA-only (rank-1) ES search parameters: **324**
  - Compression factor in ES search space: **27.48×**

### Caveat: “Frozen random base \(W\)” is atypical LoRA

- **What classic LoRA assumes**: \(W\) is a *pretrained* weight matrix (useful features) that we freeze, and we learn a small low-rank \(\Delta W\) to adapt it.
- **What we currently do in this project**: when `use_lora=True`, each `LoRALinearRankK` constructs an `nn.Linear` base layer whose weights are randomly initialized (orthogonal init via `PolicyNetwork.apply(self._init_weights)`), and then we freeze `base.weight` / `base.bias`. ES then perturbs/updates only `lora_a` and `lora_b` (when `param_mode='lora'`).
- **Why this can be a bad idea**: freezing a randomly initialized backbone often makes learning less sample-efficient and undermines the main motivation of LoRA (parameter-efficient *adaptation* of a strong base model). Functionally, this is closer to “fixed random features + a low-rank correction” than standard LoRA-on-pretrained.
- **Empirical note (important)**: despite the random frozen \(W\), LoRA-ES *did* work in our GridWorld runs (achieved high success once reward shaping was used), so this approach may still be promising for future attempts.
- **Better alternatives (if we want LoRA to be conceptually clean)**:
  - Pretrain the base policy normally (e.g., ES with `param_mode='all'` or gradient-based RL), then enable LoRA and freeze the pretrained base for adaptation.
  - Or, don’t freeze the base (train \(W\) alongside LoRA) if the goal is simply to reduce dimensionality *somewhat* without relying on a pretrained backbone.

---

## Session 3: Running Week 7 Notebook — “Nothing Happens” (Feb 14)

### Conversation with ChatGPT

**My Question:**
> I run the Week 7 notebook and it looks like it’s doing nothing (or shows 0 progress). The gradients/norms seem to die. What are the likely causes in ES?

**Response Summary:**
- Pointed out that ES can appear “stuck” if fitness values across the population are nearly identical
- Highlighted a common trap: sparse reward environments produce many trajectories with exactly 0 return, especially early
- Explained that if you standardize fitness and the population variance is ~0, the ES update becomes ~0

**Key Takeaway:**
On sparse reward tasks, ES can silently collapse if the reward distribution has near-zero variance across perturbations.

**What I Did:**
- Confirmed that Week 7 was training directly on the sparse `GridWorld` reward, making it easy for many perturbations to tie.
- Treated the issue as “lack of learning signal,” not just “slow runtime”.

---

## Session 4: Root Cause — Fitness Standardization + Sparse Reward = Zero Update (Feb 18)

### Conversation with ChatGPT

**My Question:**
> Why exactly would ES gradients be near zero even if I’m sampling perturbations correctly?

**Response Summary:**
- If returns \(F_i\) are constant (or nearly constant), then after normalization:
  - \(F_i - \\bar{F} \\approx 0\)
- Then the estimator:
  - \(\nabla_\\theta J \\approx (1/N\\sigma) \\sum_i 0 \\cdot s(\\epsilon_i) = 0\)
- Suggested using reward shaping for training (dense signal), while keeping sparse evaluation for fair reporting

**Key Takeaway:**
The “dying gradient” was consistent with the algorithm doing exactly what it should do when the population has no fitness diversity.

**What I Did:**
- Compared Week 7 to Week 4 and noticed Week 4 trained ES on a shaped-reward wrapper to avoid early plateaus.

---

## Session 5: Fix — Port Week 4 Reward Shaping into Week 7 (Feb 24)

### Conversation with Cursor AI

**My Request:**
> Update Week 7 to train ES using the same distance-based shaping strategy as Week 4, but keep evaluation on the original sparse reward env.

**Response Summary:**
- Proposed copying the Week 4 `ShapedRewardEnvComparison(GridWorld)` wrapper into Week 7
- Recommended making:
  - `train_env = ShapedRewardEnvComparison(...)`
  - `eval_env = GridWorld(...)`
- Suggested updating notebook text so the reward scheme is explicit (train vs eval)

**Key Takeaway:**
Shaped training + sparse evaluation preserves interpretability while restoring a usable learning signal.

**What I Did:**
- Added Week 4-style shaping in `notebooks/week7_implementation.ipynb`:
  - Distance shaping: `reward + 0.2 * (prev_dist - curr_dist) - 0.01`
- Kept evaluation on the original sparse `GridWorld` env for final metrics (reward/success/steps).

---

## Session 6: Aligning Sampling Budget with Week 4 + Clarifying Plots (Feb 25)

### Conversation with ChatGPT

**My Question:**
> If I want a fair comparison to Week 4, what budget should I match, and how do I avoid misleading plots when training uses shaped rewards?

**Response Summary:**
- Recommended matching ES budget elements that directly impact estimator variance:
  - Population size \(N\)
  - Episodes per perturbation
  - Iteration count
- Warned that if you plot “eval_reward” measured on the training env, it may reflect shaped reward, not sparse reward
- Suggested relabeling plots (or explicitly evaluating on a separate sparse env)

**Key Takeaway:**
Budget and metric definitions matter as much as code correctness; otherwise it’s easy to misinterpret “learning”.

**What I Did:**
- Updated Week 7 defaults to match Week 4 ES sampling budget:
  - Iterations: **80**
  - Population size \(N\): **50**
  - Episodes per perturbation: **5**
  - (Kept: `sigma=0.10`, `alpha=0.05`, `max_steps=50`)
- Relabeled the training curves in Week 7 to explicitly indicate they are **shaped reward on the training env**.

---

## Key Lessons from LLM Interactions

### What Worked Well:
1. **Asking for mechanisms, not guesses:** “What exact condition makes ES update go to zero?” produced a concrete diagnosis.
2. **Comparing to a known-good baseline (Week 4):** Framing Week 7 as a controlled delta (LoRA + sparse reward) made debugging faster.
3. **Using Cursor for implementation, ChatGPT for explanation:** Cursor accelerated edits in the notebook; ChatGPT helped validate the reasoning.

### What Didn’t Work:
1. **Interpreting silence as failure:** Long ES loops can look dead if training prints are sparse; the real issue was reward variance, not just runtime.

### Best Practices Learned:
1. **Separate training reward from evaluation reward** in sparse tasks (use shaping for training but keep sparse evaluation for reporting).
2. **Track fitness variance/gradient norms** early to detect collapse quickly.

---

## Tools Used

**ChatGPT:**
- Diagnosis of “0 progress” via sparse reward + fitness standardization reasoning
- Guidance on how to structure reward shaping without breaking evaluation fairness

**Cursor AI:**
- Fast notebook edits to port Week 4 reward shaping into Week 7
- Assisted refactors to align Week 7 ES budget and clarify plot labeling

---

## Impact Assessment

**Time Saved:** ~3–5 hours
- Faster debugging (pinpointed fitness-variance issue quickly)
- Faster implementation (porting shaping + notebook updates)

**Quality Improvement:**
- Week 7 experiments are now closer to a controlled comparison (match Week 4 budget; explicit train-vs-eval reward scheme)
- Reduced risk of misleading interpretation by clarifying what the plotted curves represent

---

*Log completed: February 25, 2026*

