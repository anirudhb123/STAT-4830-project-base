# Wordle with Qwen3 1.7B, Char-Level Generation, and ES on LoRA

**Status (this submission).** Week 14 retired the Gemma 3 1B IT pipeline that has been the running thread since Weeks 10-12 and switched the LM stack to **Qwen3 1.7B base** with **autoregressive 5-letter character generation under a vocabulary trie mask**. ES still trains LoRA adapters only. The motivation was the Week 12 result and its Session 8 follow-up: the Gemma + LoRA configuration could be made to lift `+22pp` at vocabulary 16 once ALPHA was quartered, but every attempt to scale the curriculum past that single stage on Gemma collapsed back to ~0% greedy. Switching to Qwen3 was both an architecture swap (stronger pretrained prior than Gemma 3 1B IT for English short-form generation) and a representation swap (drop the single-softmax over a fixed word vocabulary in favor of generating the five letters one token at a time, masked by a trie of legal Wordle words so every emission is in-vocabulary by construction). The headline result of Week 14 is that this configuration **also failed to learn**: even on small vocabulary stages where Week 12 had previously found signal, the trained adapter never produced a stable greedy `Success` lift over the post-warm-start baseline, and `dprobe` never fired non-zero on more than a handful of iterations within a stage. We treat Week 14 as the end of the "train-from-scratch on top of a generic LM" branch of the project and use the diagnostics to motivate the Week 16 pivot to Prime Intellect's Wordle-specific SFT/RL checkpoints.

## Problem Statement

**What changed since Week 12.** The optimizer (`train_es_wordle` + `train_curriculum` in `src/es_wordle.py`) is unchanged. Two things changed in the policy: (1) the **base LM** moved from `google/gemma-3-1b-it` to `Qwen/Qwen3-1.7B`, and (2) the **action head** moved from a single classifier over a closed word vocabulary (the Week 10/12 `WordleDiscretePolicy` and the head-only branch of `WordleGPT2Policy`) to **char-mode autoregressive generation**: the policy emits five tokens in sequence, after each emission the trie mask zeroes out logits for any letter that no longer leads to a legal Wordle word, and the resulting 5-letter string is wrapped in `<guess>...</guess>` for the env. The `ACTION_GRANULARITY = "char"` switch in `notebooks/week14_wordle_es_lora_run.ipynb` flips ES rollouts onto the new code path; the legacy `"word"` path is kept for smoke testing on `distilgpt2`.

**Why the swap.** Two motivations stacked. (a) **Architectural prior.** Gemma 3 1B IT's pretraining mixture is heavy on instruction-following dialogue and was, in qualitative reads of its rollouts on Week 12 prompts, slow to specialize to a fixed-format five-letter task; Qwen3 1.7B has a stronger raw English short-form prior in our spot-checks. (b) **Action-space representation.** The single-softmax word head bottlenecks ES gradient information through a logit-vector dimension equal to the vocabulary size; at the full vocabulary that is the layer the Week 12 LoRA-rank sweep was trying to "fix" with rank scheduling. Char-mode replaces that single classifier with the LM's own pretrained head distribution and an external trie-mask postprocessor, so the trainable surface of LoRA on attention modules controls behavior end-to-end without the head dimensionality discontinuity each curriculum stage was creating.

**How will we measure success?** Same metrics as Weeks 10/12 (`eval_reward`, `eval_success`, `eval_turns`; per-iteration `train_fitness`, `param_drift`, `pop_fitness_std`, `train_es_win`, `dprobe`, `cos(ĝ)`). Two additional diagnostics surface in the verbose log specifically for the char-mode path: **`fb%`** is the per-iteration trie-fallback rate (fraction of generations where the trie's legal-prefix mask had to redirect a sampled token to its top legal alternative — high `fb%` means the LM is repeatedly trying to emit characters that do not extend any legal Wordle word) and **`trie_steps`** is the per-iteration number of times the mask was applied. The pre-registered pass criterion for Week 14 mirrored Week 12 Session 8: at the smallest curriculum stage, `best_greedy − post_ws_greedy ≥ +10pp` AND `final_greedy ≥ post_ws_greedy` AND `dprobe` non-zero on at least 25% of iterations.

**Constraints and risks.** Qwen3 1.7B is ~70% larger than Gemma 3 1B IT in parameter count and noticeably slower per ES iteration; on the available A100 80GB nodes a single `qwen_full` curriculum stage at `N_POP=32`, `N_ITERATIONS=100`, `n_eval_episodes=32`, `eval_n_episodes=16` takes long enough that we ran Week 14 on a smaller schedule than the original Week 12 plan called for. `enable_thinking` is left on the default (off) for char-mode because the trie-masked autoregressive loop emits exactly five tokens; the model is given no budget to write a thinking block. Qwen3's chat templates require `transformers>=4.51.0` and `jinja2>=3.1.0`; both are pinned in `requirements.txt`.

## Technical Approach

**Environment and reward.** Unchanged: `src/wordle_env.py`, mock path via `load_wordle_environment(use_prime_intellect=False)`. The bundled NYT 2,315-answer pool at `data/wordle_answers.txt` is the source of truth for both legal solution words and the trie that masks generation.

**Policy.** Char-mode is implemented as an extension of `src/wordle_gpt2_policy.py` (`WordleGPT2Policy`) keyed by `action_granularity="char"`: at each guess turn the policy issues five forward passes (one per letter), each masked by `_WordleVocabTrie` against the prefix emitted so far, samples a token under the post-mask distribution, and concatenates. The trie is rebuilt at every curriculum stage to reflect the active vocabulary. Trainable parameters are LoRA adapters on the attention modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`); the LM base, embeddings, and lm_head are frozen.

**Supervised warm-start and ES.** Modules unchanged from Week 12 (`wordle_gpt2_warmstart.py`, `es_wordle.py`). Fitness shaping (`win_plus_return`, `RANK_FITNESS=True`, `BASELINE_SUBTRACT=True`, `WIN_FITNESS_SCALE=8.0`) and the Session-8 Week-12 plumbing (`restore_best_at_stage_end`, `eval_stochastic_every`) are still on by default. The ALPHA probe is the single-shot iter-0 calibration with `target initial step ≈ 0.13`.

**Experimental configuration.** `notebooks/week14_wordle_es_lora_run.ipynb` exposes a `RUN_PROFILE` switch for `smoke` (DistilGPT-2, ~2 ES iters, plumbing only) and `qwen_full` (the production configuration below).

| Setting | `RUN_PROFILE="qwen_full"` |
| --- | --- |
| `MODEL_NAME` | `Qwen/Qwen3-1.7B` |
| `USE_CHAT_TEMPLATE` | `True` |
| `ACTION_GRANULARITY` | `"char"` |
| `MAX_PROMPT_LENGTH` | 512 |
| LoRA | rank 8, α=16, target_modules=`["q_proj","k_proj","v_proj","o_proj"]`, dropout 0.05 |
| ES | `N_POP=32`, `N_ITERATIONS=100` per stage |
| `n_eval_episodes` | 32 |
| `eval_every` | 5; `eval_n_episodes`=16 |
| Warm-start | `WARM_START_STEPS=400` per stage |
| Vocabulary curriculum | `VOCAB_SCHEDULE=[16, 32, 64]` (ran in this order) |
| Global ES knobs | `SIGMA=0.02`, `ALPHA` from probe (target step ≈ 0.13), `RANK_FITNESS=True`, `BASELINE_SUBTRACT=True`, `EMA_BETA=0.0`, `RICHER_PROMPT=True`, `FITNESS_OBJECTIVE="win_plus_return"`, `WIN_FITNESS_SCALE=8.0` |

**Artifacts.** Per-stage adapter checkpoints save to `models/wordle_qwen_es_head.<RUN_PROFILE>.pt` (and the gitignored `models/wordle_qwen_es_head.*.pt` siblings for sweep variants). Training history pickles to the same prefix.

**Code paths.**

- `notebooks/week14_wordle_es_lora_run.ipynb` (Qwen3 + char-mode + LoRA + ES)
- `src/wordle_gpt2_policy.py` (now hosts the char-mode trie-masked generation path under `action_granularity="char"`)
- `src/wordle_gpt2_warmstart.py`, `src/es_wordle.py`, `src/wordle_env.py`, `src/wordle_hints.py` (unchanged in spirit; `es_wordle.py` already wires `_rollout_batched` through `sample_words_batch` when `action_granularity="char"`)

## Initial Results

**Stage 1 — `VOCAB_SCHEDULE=[16]`.** Post-warm-start greedy `Success` was 28% on the held-out 16-secret slate (comparable to Week 12 Session 8's 30-36% post-WS at the same vocab; the 8pp gap is partially attributable to the architecture swap and partially to per-stage warm-start variance). Across 100 ES iterations: median `dprobe` non-zero fraction = 9/100 (well below the 25% pass criterion); `cos(ĝ)` median ≈ 0.00 with a few isolated +0.05 spikes; `best_greedy = 34%` at iter 4 (`+6pp` vs post-WS, inside the single-slate noise band); `final_greedy = 22%` at iter 99 (`−6pp` vs post-WS, inside the same band). `‖θ − θ₀‖` grew approximately linearly to 5.1, so the optimizer was *moving* — it was just walking inside a level set rather than climbing one. **Verdict: FAIL on the pre-registered rubric** (no `+10pp` lift; `dprobe` non-zero on 9% of iterations vs the 25% floor).

**Stage 2 — `VOCAB_SCHEDULE=[32]`, carrying the stage-1 adapter.** Post-warm-start greedy `Success` dropped to 14% on the new 32-secret slate. ES contributed essentially nothing on top: 100 iterations, `best_greedy = 16%`, `final_greedy = 10%`, `dprobe` non-zero on 6/100 iterations. **Verdict: FAIL.** Because stage 1 had not produced a credible lift, we did not bother gating the stage-3 attempt; the run was killed after stage 2.

**`fb%` (trie fallback rate).** Stayed in the 60-90% range across both stages and across all population members, including the unperturbed θ₀. This is the most informative new diagnostic: it says the LM, even after warm-start, is consistently trying to emit characters that do not extend any legal 5-letter Wordle prefix, and the trie mask is constantly redirecting it. Two interpretations are consistent with this number: (a) the LM's distribution over the next-letter position is too spread out — the trie mask is doing all the structural work and the gradient from "did the masked sample win?" is dominated by the mask, not by the LM; (b) the prefix-space the trie defines is too restrictive relative to where Qwen3 wants to put probability mass, so the masked logit redistribution is essentially uniform-over-legal each time. Either reading predicts what we observed: ES gradient signal collapses because the per-step distribution the policy actually samples from is approximately mask-determined.

**Where the runs are saved.** The week-14 char-mode adapter and history pickle are at `models/wordle_qwen_es_head.qwen_full.pt`; the ES history (per-iteration diagnostics for both stages) is at the matching `models/wordle_qwen_es_history.qwen_full.pkl`. Plot panel layout matches Week 12 Session 8 with the addition of a `fb%` and `trie_steps` row.

**Interpretation.** The Week 14 result rules out two plausible Week-12-vintage stories. (1) "Gemma was the wrong base model" — replacing it with Qwen3 did not produce ES signal. (2) "The single-softmax word head was the bottleneck" — moving to char-mode autoregressive generation under a trie mask did not produce ES signal either. What it does *not* rule out is the broader Week 12 critique reading that the per-secret revisit count under CRN is the binding constraint; we did not vary that here. But within the design space the project has been exploring for five weeks (warm-start a generic LM and let ES specialize it via LoRA on Wordle), the marginal cost of the next experiment is now high (each curriculum stage on Qwen3 takes hours per seed) and the marginal information gain has been low for three consecutive weeks.

## Next Steps

1. **Stop fighting the warm-start curve.** Three consecutive weeks of "warm-start does most of the lifting; ES adds noise" is a signal about what we should be initializing from. The Week 16 plan is to *skip our own warm-start* and start from a Wordle-specific SFT checkpoint trained by someone with a real RLHF budget (`PrimeIntellect/Qwen3-1.7B-Wordle-SFT`), then ask only the narrower question: does ES add anything on top of an already-Wordle-specialized base?
2. **Drop char-mode for the Week 16 attempt.** The `fb%` story above suggests char-mode was hurting more than helping for our SNR regime. Week 16's `WordleQwenPolicy` will use `model.generate(...)` with a `<guess>WORD</guess>` parser instead of trie-masked per-letter sampling, so the LM is allowed to produce its native distribution end-to-end.
3. **Use Prime Intellect's RL checkpoint as a reference ceiling.** `PrimeIntellect/Qwen3-1.7B-Wordle-RL` is the next step in their pipeline (SFT → online RL on Wordle). Evaluating it gives us an "ES contribution" attribution number with a real numerator and denominator instead of the n/a reports we have been printing.
4. **Keep the new diagnostics.** The `fb%` / `dprobe` / `cos(ĝ)` / `‖θ − θ₀‖` quad is now load-bearing for any future ES claim; the Week 16 policy reuses the trie-stats hooks for a different metric (`<guess>` parse-failure rate) so the verbose-log column survives even when the trie itself does not.

## References

1. Yang, A., et al. (2025). *Qwen3 Technical Report.* arXiv:2505.09388.  
2. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. *arXiv:1703.03864*.  
3. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.
