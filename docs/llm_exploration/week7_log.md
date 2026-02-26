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

## Session 7: Adaptation Pipeline + Pretrain Gate + Curriculum Perturbations (Feb 26)

### Conversation with Cursor AI

**My Requests:**
> I want an actual transfer experiment: pretrain ES on source GridWorld, perturb the environment, then continue training both full ES and LoRA-only from the same source policy and compare adaptation speed.

> Add stronger run logging, CSV export, and make sure source pretraining is functional before adaptation (with max-iteration guard + failure handling).

> Transition-noise perturbation is not what I meant; I want the **grid layout itself** perturbed in a curriculum style (small cell moves).

**Response Summary:**
- Refactored notebook into a paired adaptation protocol:
  - common pretrained source policy per seed
  - same perturbed target layout per seed for both methods
  - direct comparison of `param_mode='all'` vs `param_mode='lora'`
- Added adaptation metrics aligned to “how quickly”:
  - time-to-threshold (`0.6`, `0.8`, `0.9`)
  - interactions-to-threshold
  - success AUC
  - final sparse-eval reward/success/steps
- Added ES logging support in `src/utils.py` for:
  - `eval_steps`, `fitness_std`, and cumulative interaction counts
- Added CSV outputs for downstream reporting and plotting:
  - run-level, summary, and LoRA-minus-full delta tables
- Added verbose progress printing for long runs:
  - perturbation level, seed, method, per-run elapsed, ETA, and key metrics
- Implemented source-model robustness controls:
  - optional dynamic pretraining with early stop chunks
  - max-iteration cap
  - source-quality gate before adaptation
  - explicit skip rows (`skipped_reason`) if source quality fails
- Replaced transition perturbation with layout perturbation:
  - move a controlled fraction of obstacle cells locally
  - optional slight goal relocation at higher perturbation levels
  - updated notebook wording/axes to “layout move fraction”

**Key Takeaways:**
1. For LoRA transfer, quality of pretrained base \(W\) is first-order; explicit gating avoids misleading comparisons.
2. Layout perturbation better matches curriculum-style environment shift than transition noise.
3. Adaptation-speed claims are more credible with threshold/interactions/AUC and multi-seed CI curves.

**What I Did:**
- Updated `notebooks/week7_implementation.ipynb` to run the full transfer experiment with robust logging and export.
- Updated `src/utils.py` to expose adaptation diagnostics needed for sample-efficiency analysis.
- Added pretrain failure handling so bad source policies do not silently contaminate adaptation results.

---

## Session 8: New Adaptation Results from Notebook Exports (Feb 26)

### Conversation with Cursor AI

**My Request:**
> Summarize the new Week 7 adaptation results from `notebooks/results` and discuss what they mean for the final log.

**Response Summary:**
- Reviewed the exported result tables from `notebooks/week7_implementation.ipynb`:
  - `es_lora_adaptation_runs.csv`
  - `es_lora_adaptation_summary.csv`
  - `es_lora_adaptation_deltas.csv`
- Compared full-parameter ES vs LoRA-only ES across perturbation levels.
- Focused on adaptation-speed metrics (`AUC`, time/interactions to thresholds), not just final success.

**Key Takeaway:**
The new transfer results are **not great** for LoRA under larger layout shifts: both methods end with high final sparse-eval success, but LoRA often adapts slower at higher perturbation (especially at `perturb_std=0.5`, and still somewhat at `0.75`).

**What I Observed in the Results:**
- Final metrics alone are misleadingly optimistic:
  - both methods reach near-perfect final eval success/reward in most settings.
- Adaptation-speed metrics reveal the weakness:
  - At `perturb_std=0.5`, LoRA has lower success AUC and much larger interactions-to-0.8 than full ES.
  - At `perturb_std=0.75`, LoRA remains slower, though the gap is smaller than at `0.5`.
- At easier shifts (`0.0`, `0.25`), LoRA remains competitive and can match/beat full ES on interactions.

**Important Positive Note (Matches Our Goal):**
- Even though the adaptation outcome is weaker than hoped, the new plotting/export pipeline produced **less noisy, more stable-looking curves** than earlier Week 7 attempts.
- With paired seeds, controlled layout perturbations, and run-level logging, variance in the graphs is visibly reduced and trends are easier to interpret.
- So, while the performance conclusion is mixed-to-negative for LoRA speed under harder shifts, the experiment quality and variance behavior improved in the direction we wanted.

**Important Limitation:**
- We still have **not** done systematic hyperparameter tuning for this Week 7 transfer setup, so these comparisons should be interpreted as baseline results rather than optimized performance ceilings.

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

**Time Saved:** ~6–9 hours total across Week 7
- Faster debugging (pinpointed fitness-variance issue quickly)
- Faster implementation (porting shaping + adaptation refactor + exports)

**Quality Improvement:**
- Week 7 now supports a controlled transfer-learning comparison from a shared source checkpoint.
- Reduced risk of invalid LoRA conclusions by enforcing source-quality checks before adaptation.
- Improved reproducibility/reporting via explicit CSV outputs and verbose run-level diagnostics.
- Perturbation design now better matches stated curriculum objective (layout moves, not action slips).

---

*Log completed: February 26, 2026*

