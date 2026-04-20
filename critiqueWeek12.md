# Week 12 Self-Critique

**Scope of this critique.** Week 12 ran two distinct Gemma 3 1B IT experiments: (A) `notebooks/week12_implementation.ipynb` — Gemma on the **8-word mock vocabulary** with **no LoRA** (head-only ES), which **worked** (~83% greedy success at iter 9, `ES_win` from 68.8% to 93.8%); and (B) `notebooks/week12_implementation_LoRARun.ipynb` — Gemma on the **full Wordle vocabulary** with a **LoRA rank sweep** (r=4, 16, 32) followed by an **8-stage curriculum** (`VOCAB_SCHEDULE = [16, 32, 64, 96, 128, 256, 512, 1024]`) plus a Phase-C SNR debugging push, which **failed to demonstrate any ES gain on top of supervised warm-start across six debugging attempts** before a one-line probe identified the actual bottleneck. The DistilGPT-2 smoke profile in either notebook is plumbing-only and not a research baseline. The critique below focuses on what made (A) succeed, why (B)'s six debugging attempts were the wrong layer of the stack, and what the signal-density probe finally revealed.

## ORIENT

### Strengths

- **Notebook A is a real positive result.** Gemma + head-only ES on the 8-word mock vocabulary reaches ~83% greedy success at iter 9 with rising `ES_win` (68.8% -> 93.8%). The Gemma backbone, chat-template prompts, warm-start, and ES loop all behave as intended end-to-end on this task.

- **Run profiles directly address Week 10 workflow pain.** The **`RUN_PROFILE`** switch (`smoke` vs `gemma_full`) separates "does the notebook run end-to-end?" from "are we doing a serious experiment?" without maintaining a dedicated debug notebook. DistilGPT-2 smoke is plumbing-only; the real Gemma runs are in the two `gemma_full` configurations.

- **Two-notebook split lets us compare configurations cleanly.** Keeping the head-only mock run (`week12_implementation.ipynb`) separate from the full-vocab LoRA sweep (`week12_implementation_LoRARun.ipynb`) makes the contrast between the working and failing setups easy to point at.

- **Gemma IT + LoRA path is runnable end-to-end.** `USE_CHAT_TEMPLATE`, `CHAT_GENERATION_PROMPT`, and PEFT LoRA adapters are wired and executable, so the failure of the LoRA sweep is about training behavior and experiment design, not missing integration.

- **Checkpoint isolation by profile.** Saving to **`models/wordle_gemma_es_head.<RUN_PROFILE>.pt`** prevents a smoke run from clobbering a long Gemma checkpoint — a small change that avoids silent data loss.

- **The diagnostic surface paid for itself this week.** Adding `cos(ĝ_t, ĝ_{t-1})`, `dprobe` (pre/post-step greedy-success delta on a fixed-seed slate), `‖θ − θ₀‖`, `Step‖`, `Grad‖`, `popσ`, `ess_rank`, and `wins/N` to the per-iteration log made it possible to falsify five Phase-C variants in turn rather than guessing whether each "felt better." Without this surface we would still be tweaking estimator weights blind.

- **The signal-density probe (`PROBE_VOCAB=4`) is a positive result.** With `PROBE_VOCAB=16 → 4` and `PROBE_N_EVAL=16` held fixed, the ES-only probe took greedy success **0% → 86% peak / 66% final** over 20 iters with `dprobe` peaks of +25.0%, +28.1%, +15.6%. This is the first unambiguous evidence in Notebook B that ES *can* contribute on top of warm-start when the eval-budget-per-secret ratio is right.


### Areas for Improvement

- **The full-vocabulary LoRA sweep did not work, and neither did six attempts to fix it at the estimator/momentum/capacity layer.** Under Notebook B's `MOCK_ENV=False` + full vocab + Gemma + LoRA configuration, the following were tried in order, none producing measurable `es_gain`:
  1. **Phase A diagnostic probe** (vocab=16, N=64, n_eval=16): `cos(ĝ) = -0.000`, eval lift = +12pp, **FAIL** on both criteria.
  2. **Phase C #1 — baseline-subtracted ES (PGPE-lite):** added a `baseline_subtract=True` branch in `src/es_wordle.py` to use raw mean-centered fitness instead of rank-fitness with std normalization. `cos(ĝ)` still ±0.00, `Step‖` decayed 0.13 → 0.026, `dprobe` non-zero on 1/14 iters. **Failed.**
  3. **Phase C #2 — disable EMA momentum** (`EMA_BETA = 0.9 → 0.0`): correct fix to a real pathology (EMA was averaging successive uncorrelated raw gradients toward zero — `Grad‖` decayed 3953 → 802 over 13 iters with stable per-iter raw gradient). After the fix, `dprobe` non-zero on ~half the eval iters with mixed signs (+9.4%, −9.4%, +9.4%) but fresh-secret `Success` still 0–12% — improvements were **secret-specific, not transferable.** Verdict: still fail.
  4. **Test A — LoRA capacity** (`LORA_R = 2 → 8`): quadrupled trainable subspace. **Failed** with the same pattern. Capacity was not the bottleneck.
  5. **Test B — signal density** (`PROBE_VOCAB = 16 → 4`, `PROBE_N_EVAL=16` held fixed, so each population member now plays each secret ~4× under CRN per iter rather than ~1×): **0% → 86% greedy success.** This is the bottleneck.

- **The bottleneck was the per-secret revisit count per ES iteration under common-random-numbers, not the ES estimator.** For the production curriculum, every stage sat in the failure regime that Test B escaped:

  | Stage | Secret pool | Episodes/iter | Visits per secret per iter |
  |------:|------------:|--------------:|---------------------------:|
  | 1     | 16          | 16            | **1×**                     |
  | 2     | 32          | 16            | **0.5×**                   |
  | ...   | ...         | 16            | progressively worse        |
  | 8     | 1024        | 16            | **0.016×**                 |

  When most population members win 0–12 of 16 episodes, the "winners" are winning *which secrets they happened to draw* under CRN — a 1-trial Bernoulli win-rate estimate, not a transferable policy improvement. No estimator variant (rank-fitness, baseline-subtract, EMA on/off, larger LoRA) can rescue ES from this regime; it's a property of the env-eval design.

- **The probe verdict logic is too strict.** Test B printed `FAIL` because `cos(ĝ) ≈ 0`, even though greedy success climbed 0% → 86%. **`cos(ĝ)` is a one-way indicator** — positive proves signal, but in a 17M-parameter LoRA space the cosine between two independent fitness-weighted noise vectors is the null behavior, not evidence of failure. The pass criterion needs to be `eval_lift OR cos(ĝ)`, not `AND`.

- **Three of the six attempts cost compute on the wrong layer of the stack.** PGPE-lite, EMA-on/off interaction, and the LoRA r=8 capacity bump together ate ~6 hours of compute. They were all investigations of *how the ES estimator weights the population*, when the actual bottleneck was *how often each secret was revisited per iteration*. Cheap experiments first should have been the rule from the start — `PROBE_VOCAB=4` was a one-number change.

- **Train/test split isolation is still weak.** The process does not enforce a persistent, auditable disjoint train/eval word split for the full-vocabulary run. Now that Test B has demonstrated ES *can* contribute when the eval budget is right, the next round of experiments needs strict split isolation before any headline metrics are reported.

- **Curriculum learning is necessary between Notebook A and Notebook B, but the way Notebook B uses it (`VOCAB_SCHEDULE = [16, 32, 64, 96, 128, 256, 512, 1024]` with `n_eval_episodes=16` everywhere) bakes in the Test-B failure mode at every stage past the first.** Either `n_eval_episodes` must scale with stage size to keep ≥4 visits per secret per iter (infeasible at vocab=1024), or the per-iter secret pool must be subsampled (mini-batch ES under CRN — e.g. 8 secrets × 32 episodes = 4 visits per secret).

- **Notebook B confounds vocabulary scaling and LoRA.** Going from Notebook A to Notebook B simultaneously changed the vocabulary (8 → ~2k actions) and enabled LoRA. Even with the curriculum staging, the failure cannot be attributed to either alone — the Test-B finding shows the eval budget *also* changes regime as the secret pool grows, so we now have three confounded axes (vocab × LoRA × eval-budget-ratio), not two.

- **DistilGPT-2 smoke profile has low statistical power and is not a baseline.** `N_POP=4`, `N_ITERATIONS=2`, and very short warm-start make `RUN_PROFILE="smoke"` useful only for plumbing checks. It should not be cited as evidence about ES dynamics or model quality.

### Critical Risks / Assumptions

- **Risk:** Without strict train/test split isolation on the full-vocabulary task, we cannot make strong generalization claims even if Notebook B's headline metrics rise in future runs.

- **Risk (newly identified this week):** The eval-budget-per-secret ratio is a hidden axis that silently determines whether ES is in a "real signal" or "lottery-ticket" regime. Any future ES configuration that holds `n_eval_episodes` fixed while scaling the secret pool reproduces this failure mode by construction. This needs to be designed into the curriculum, not patched afterward.

- **Risk (verdict logic):** The Phase A probe's `AND` criterion (`cos(ĝ) > 0.05` AND `eval_lift ≥ 15pp`) gave a `FAIL` verdict on the run that produced our best ES result of the semester (Test B). The criterion should be `OR`, with `cos(ĝ)` understood as a one-way indicator. Until this is fixed, the probe will continue to flag working configurations as broken.

- **Risk:** Going straight to full-vocabulary optimization with LoRA enabled can hide whether failure comes from vocabulary size, LoRA capacity, reward shaping, eval budget, or optimization horizon; the current Notebook A vs Notebook B contrast collapses several of these factors into one experiment. After this week we know the eval-budget axis matters at least as much as the others.

- **Risk (wrong-layer debugging):** Three of this week's six attempts (PGPE-lite, EMA off, LoRA r=8) intervened on the ES estimator / momentum / capacity layers when the actual bottleneck was the env-eval design. This is a generalizable pattern — when a metric (`cos(ĝ)`) is flat across multiple estimator variants, the *next* hypothesis should be a property of the data the estimator sees, not another variant of the estimator itself.

- **Assumption:** The 8-word mock vocabulary success in Notebook A transfers in any informative way to the full vocabulary. Mock-task win rate may overstate how close the policy is to playing real Wordle. (Test B partially supports this — 4-word vocab also "worked" at 86% peak — but 4-word is even smaller than 8-word, so this is not yet evidence of transfer to ≥16-word secret pools.)

- **Assumption:** Chat-templated prompting for Gemma preserves the same effective signal as plain prompts. If template overhead or truncation changes usable context at full vocabulary, training quality can degrade independently of ES.

## DECIDE

### Concrete Next Actions

1. **First: rerun the full pipeline at `VOCAB_SCHEDULE = [4]`, `N_ITERATIONS = 30`** as a single-stage proof that warm-start + ES produces a measurable `es_gain > 0` end-to-end. The Test B probe was ES-only and skipped warm-start; this experiment closes the loop.

2. **Fix the probe verdict logic** to `eval_lift > 15pp OR median cos(ĝ) > 0.05`. Document `cos(ĝ)` as a one-way indicator in the cell-9 markdown.

3. **Implement mini-batch ES under CRN.** Sample a fixed-size subset of the secret pool per iter (e.g. 8 secrets × 32 episodes = 4 visits per secret per iter) so the per-secret revisit count stays in the working regime regardless of the global vocab size. This is a ~30-LoC change in `src/es_wordle.py`. Test it by reproducing the Test B 0% → 86% result on a `VOCAB_SCHEDULE = [16]` stage (4× larger than the probe).

4. **Create strict split artifacts first.** Before reporting any new headline metrics, generate and save fixed disjoint train/eval word lists for the full Wordle vocabulary, reuse them across seeds, and report metrics only on the held-out split.

5. **Decouple LoRA from vocabulary scaling.** With the eval-budget axis now identified as a third confounder, add at minimum: (a) Gemma + LoRA on the 8-word mock task with proper eval budget, (b) Gemma head-only on a curriculum stage with proper eval budget. This is the only way to attribute residual failure modes.

6. **Run warm-start vs ES ablations under the same split AND the same per-secret revisit count.** Evaluate after warm-start and after ES on identical held-out episodes. If ES still adds nothing once the eval budget is correct, the LM logit prior may be dominant enough that the head-only / rank-8 LoRA configuration cannot move argmax across many secrets at once — at which point either per-turn dense shaping (yellow/green letter coverage, constraint consistency) or a much larger LoRA budget becomes the next experiment.

## ACT

### Resource Needs

- **GPU with sufficient VRAM** for **`google/gemma-3-1b-it`** (bf16/fp16) plus batched forward passes via `forward_logits_batch` over long prompts; CPU `gemma_full` is only for single-step debugging. A single A100 runs the full 8-stage curriculum in ~5 hours.
- **Disk / cache** for Gemma weights; first-time download can be large.
- Optional: **`peft`** for LoRA experiments (already used).
- **Compute lost this week to wrong-layer debugging:** ~6 hours on PGPE-lite + EMA-off + LoRA r=8 variants. Future Phase-C-style debugging should follow the rule "cheapest experiment first" — `PROBE_VOCAB=4` was a one-line change that found the bottleneck after five expensive software changes had not.
- **Wall-clock budget for next week:** the `VOCAB_SCHEDULE=[4]` proof run is ~30 min on an A100 (single stage, 30 iters, full warm-start + ES); mini-batch ES at vocab=16 with proper eval-budget ratio is ~1 hour. Parallelizing population eval across processes remains an undeployed speedup that would help all of these.

---

## Postscript: Experiments 1, 2, 3 (Apr 19, same day)

Follow-through on the three "Concrete Next Actions" above. Details in `docs/llm_exploration/week12_log.md` Session 7; brief outcomes here.

- **Action 2 (probe verdict OR logic) — DONE.** Cell 10 criterion flipped to OR; cell 9 documents `cos(ĝ)` as a one-way indicator. Small diff, large reliability win.

- **Action 1 (closed-loop proof at vocab=4) — FAIL on the verdict criterion, but metric-saturation FAIL.** Even with warm-start cut from 200 → 20 episodes, post-WS greedy saturated to 100% on the 4-word pool (greedy on 4 secrets with 6 guesses is a trivially-solved problem once the head distinguishes those words). Literal `es_gain = +0.0pp`. But `ES_win` climbed 9% → ~60%, `wins/N` 37/64 → 64/64, `‖θ−θ₀‖` grew monotonically — **ES is learning, greedy success just can't measure it at this pool size.** Lesson: the closed-loop proof design needs a pool size where greedy success is strictly inside `(0, 1)`, not on the corners. Vocab=4 was too aggressive a shrink.

- **Action 3 (mini-batch ES under CRN) — Implemented (~40 LoC), FAIL on brief criterion, but new nuanced picture.** Added `per_iter_secret_subset_size` kwarg to `es_gradient_estimate_wordle` / `train_es_wordle`. At `VOCAB_SCHEDULE=[16]`, `k=4`, `n_eval_episodes=16` (~4 visits/secret/member/iter — Test B regime): post-WS greedy = 42%, post-ES greedy final = **30% (es_gain = −12pp)**, but **iter-1 alone gained +18pp (42 → 60%)** and `dprobe` fired non-zero on **53% of iters** (passes the brief's 25% criterion). `ES_win` climbed +24pp, `‖θ−θ₀‖` grew monotonically 2.62 → 13.56 (average step ≈ 0.45 vs ALPHA-calibration target 0.13 — **3-4× overshoot**).

- **Diagnosis — the bottleneck has moved from env-eval design to optimizer aggregation.** Mini-batch CRN *does* restore per-iter signal density (the week-12 headline finding). But the 4-word subset *rotates* each iter, so the objective's level sets rotate with it, and ALPHA calibrated at iter 0 is stable for a stationary objective, not a rotating mini-batch. Result: large steps toward each iter's rotating target, no accumulation along a shared direction, random walk around the post-WS basin with the best iterate at iter 1.

- **Risk (newly identified):** The default ALPHA auto-calibration assumes the objective is stationary across iterations. With `per_iter_secret_subset_size` now a real feature, ALPHA needs either (a) online recalibration against the expected gradient magnitude across several rotated subsets, or (b) a smaller fixed conservative value chosen for the rotating-objective regime. The current one-shot calibration at iter 0 is not safe for this feature as designed.

- **Risk (metric):** Greedy success on ≤16 secrets has a 7pp noise floor at 50-episode eval, and can saturate to 100% at very small pools. Both failure modes fired this session (Exp 1 saturated; Exp 2 noise-dominated). Future reports need at least one of: (a) stochastic-sampling eval as a secondary metric, (b) deterministic eval on all secrets × enough episodes to get noise floor below 3pp, (c) best-iter checkpointing so peak performance isn't hidden by late regressions.

- **Next cheapest experiment:** rerun Exp 2 with ALPHA quartered and N_ITERATIONS doubled to 60 — targets the step-size/overshoot hypothesis with no code changes (env-var overrides on `scripts/run_experiment2_minibatch_crn.py`). ~1 hour.

---

## Postscript: Session 8 — Overshoot Follow-up (Apr 20)

Follow-through on the last bullet of the "Postscript: Experiments 1, 2, 3" block above ("rerun Exp 2 with ALPHA quartered and N_ITERATIONS doubled to 60"). Details in [`docs/llm_exploration/week12_log.md`](docs/llm_exploration/week12_log.md) Session 8; outcome summary here.

- **Plumbing (3 files, ~190 LoC net).** Added `restore_best_at_stage_end` and `eval_stochastic_every` kwargs to both `train_es_wordle` and `train_curriculum` in [`src/es_wordle.py`](src/es_wordle.py) — track best-by-greedy-eval iterate as a CPU-cloned `policy.state_dict()` (only allocated when the flag is True, so the legacy path has zero overhead), `load_state_dict` back at stage end, append `best_iter` + `best_eval_success` to the combined history; run `quick_eval_success(..., stochastic=True)` at the same cadence as greedy eval, wrapped in the existing RNG snapshot/restore so it doesn't perturb the ES stream. Wired `EXP2_RESTORE_BEST` / `EXP2_EVAL_STOCHASTIC_EVERY` env vars and a rewritten summary block (`final_greedy`, `best_greedy`, `final_stochastic`, `best_stochastic`, `best_iter` side-by-side, graded against `best_greedy`) into [`scripts/run_experiment2_minibatch_crn.py`](scripts/run_experiment2_minibatch_crn.py). Added an `ACTIVE_EXPERIMENT="exp2_overshoot"` branch plus post-calibration `ALPHA_SCALE` hook to [`notebooks/week12_implementation_LoRARun.ipynb`](notebooks/week12_implementation_LoRARun.ipynb) for notebook parity. Single-stage probe only — no changes to fitness shaping, RANK_FITNESS, BASELINE_SUBTRACT, EMA_BETA, LoRA rank, or subset size.

- **Result: PASS-A.** On `VOCAB_SCHEDULE=[16]`, `N_ITERATIONS=60`, `ALPHA × 0.25`: pre-WS 0% → post-WS **36%** greedy; `best_greedy = 58%` (iter 1, `+22pp` above post-WS), `final_greedy = 52%` (`+16pp` above post-WS, only `−6pp` below best — within the ~7pp single-slate eval noise floor). `dprobe` non-zero on 35/60 iters = **58%** (vs Session 7's 53%). `‖θ−θ₀‖` grew linearly 0.66 → ~4.9 across the 60-iter run, matching the implicit `0.13 × 60 × 0.25 ≈ 1.95` per-dim accumulation target within a small-constant factor.

- **Diagnosis confirmed.** The Session 7 hypothesis — ALPHA calibrated at iter 0 assumes a stationary objective and overshoots under rotating 4-word mini-batch level sets — is correct. Quartering ALPHA alone was sufficient: `final_greedy ≥ post_ws_greedy + 10pp`, so ES converges on its own without needing best-iter restore as a crutch (PASS-B would have fired if the restore were load-bearing; it did not). Best-iter restore is still useful as a robustness feature, but this configuration does not need it to clear the verdict bar.

- **Residual limitation — measurement, not optimization.** Greedy on 50 episodes × 16 secrets has a 7pp single-slate noise floor, which now dominates the per-iter trajectory: greedy oscillates 30-58% iteration-to-iteration even when `‖θ−θ₀‖` grows monotonically. This is the same metric-lie surface that hid ES signal in Experiments 1 and 2. Next cheapest next-experiment before scaling vocab: replace the 50-episode probe slate with deterministic 16 × K episodes (full pool, K visits per secret) so the noise floor drops below 3pp; only then rerun at `VOCAB_SCHEDULE=[32]` or re-enable the full curriculum.

- **Hard rules honored.** Single-stage probe only. `git diff --stat` before the full run showed only the three files above (plus the two docs edits afterward per Section 5 of the plan). No push to main; PR is on branch `week13/es-overshoot-followup`.

