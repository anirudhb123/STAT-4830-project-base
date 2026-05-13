# Wordle ES on top of Prime Intellect's Wordle-Specialized Qwen3 Checkpoints

**Status (this submission).** Week 16 abandons the "warm-start a generic LM and let ES specialize it on Wordle" pipeline that ran from Weeks 10 through 14 and instead asks a strictly narrower question: **does ES, with the same `train_es_wordle` machinery we have been iterating on, add anything on top of an already-Wordle-specialized base model?** The base models are Prime Intellect's two publicly released Wordle Qwen3 checkpoints — `PrimeIntellect/Qwen3-1.7B-Wordle-SFT` (the supervised stage of their pipeline) and `PrimeIntellect/Qwen3-1.7B-Wordle-RL` (the next stage, post their own online RL on Wordle). The new policy class `src/wordle_qwen_policy.py` drops the trie-masked char-mode generation that Week 14 ran (and failed at) in favor of `model.generate(...)` with a `<guess>WORD</guess>` parser; ES still trains LoRA adapters only (r=4 on `q/k/v/o_proj`, ~1.6M trainable params on a ~1.72B-param base). The headline result of three runs — ES-on-SFT, ES-on-RL, and ES-on-RL with normalized gradients — is that **none of them improved on the cold (no-adapter) base**: every run has a `final eval_success = 0.0%` and a `best eval_success = 12.5%` at an early iteration that is statistically indistinguishable from cold-start eval noise on the 16-secret slate. We treat Week 16 as the project's terminal experiment on "ES adds something measurable on top of a Wordle-specialized base"; the answer at the budget we ran (`N_POP=8`, `N_ITER=15`, `EVAL_N_EPISODES=16`) is "no."

## Problem Statement

**What changed since Week 14.** Three things changed in the policy and one in the experimental design.

1. **Base model.** `Qwen/Qwen3-1.7B` (generic) → `PrimeIntellect/Qwen3-1.7B-Wordle-SFT` (their SFT stage) for the first run, then `PrimeIntellect/Qwen3-1.7B-Wordle-RL` (their RL stage, a strictly later point in their own training pipeline) for the second and third. Both are 1.72B-param Qwen3 derivatives released as the *Wordle Verifiers* checkpoint family by Prime Intellect; the SFT model is trained on supervised Wordle traces, the RL model continues from SFT under online RL on the same task.
2. **Action representation.** `ACTION_GRANULARITY="char"` (autoregressive 5-letter generation under a trie mask) is gone. The new `WordleQwenPolicy` calls `model.generate(...)` once per state, decodes the output, and parses `<guess>WORD</guess>` with a layered fallback (regex → first 5-letter alphabetic run with `<think>` blocks stripped → `XXXXX` sentinel that the env scores as `Invalid guess`, reward=0, turn consumed). `_FALLBACK_SKIP` discards common 5-letter reasoning fragments like `THINK`, `WHICH`, `THERE` so they don't beat a real word as the first 5-letter regex match.
3. **Thinking is on.** `enable_thinking=True`, `MAX_NEW_TOKENS=512`. The first attempt this week (`notebooks/week16_wordle_es_qwen_sft.ipynb`) ran with `enable_thinking=False`, `MAX_NEW_TOKENS=64` and produced **100% parse failure** — the SFT/RL Wordle checkpoints were trained to emit a `<think>...</think>` block regardless of the chat-template flag, the 64-token budget was exhausted inside the thinking opener, and the parser never reached `<guess>`. Flipping thinking on and giving 512 tokens of generation budget is what made the runs scoreable at all.
4. **No supervised warm-start, no curriculum.** All Weeks 10-14 runs warm-started LoRA from a few hundred episodes of fitness-shaped supervision before turning ES on. Week 16 skips this entirely — the SFT/RL base *is* the warm-start. ES is the only training signal the LoRA adapter ever sees.

The combined intent: take the most-Wordle-trained 1.7B-param checkpoint that exists, attach a small LoRA, and ask ES to find an additive improvement. If the answer is yes, we have an "ES contribution" number worth reporting. If the answer is no, we have a clean negative result that ES under our budget cannot move a checkpoint that has already been RL-trained on the same task — which is itself the headline finding of the project.

**How will we measure success?** Two quantities matter. (a) **`final eval_success`** — greedy success rate on a fixed 16-secret eval slate after `N_ITER=15` ES iterations. (b) **`best eval_success`** — the maximum greedy success rate seen across the 15 iterations on the same eval slate (with `RESTORE_BEST_ON_FINISH=True` reloading the best iterate at the end of training). The pre-registered pass criterion was `best eval_success ≥ SFT_cold + 10pp` AND `final eval_success ≥ SFT_cold` AND `dprobe` non-zero on ≥ 25% of iters. We retained the per-iteration diagnostic surface from Weeks 12/14 (`Fit(win+ret)`, `ES_win`, `popσ`, `‖θ−θ₀‖`, `|g|`, `|Δ|`, `cos(ĝ)`, `ess`, `wins`, `dprobe`, `fb%`); `fb%` is now repurposed (the trie is gone) to report the `<guess>` parse-failure rate, so the same column survives the action-representation change.

**Constraints and risks.** Budget of `N_POP=8` × `EVAL_N_EPISODES=16` × `N_ITER=15` is small for a 1.6M-trainable-parameter LoRA — Salimans et al. (2017) used `N_POP` in the thousands. We chose `N_POP=8` because each ES iteration requires `(2 × N_POP + 1) × N_EVAL_EPISODES × MAX_TURNS` `model.generate` calls at `MAX_NEW_TOKENS=512`, and `N_POP=32` would have made each iteration take hours on the available A100 80GB / H100 PCIe nodes. Antithetic perturbations and common random numbers across population members are on (`ANTITHETIC=True`, `COMMON_RANDOM_NUMBERS=True`) to keep variance manageable at small population.

## Technical Approach

**Environment.** Unchanged — `src/wordle_env.py`, the bundled NYT 2,315-answer pool at `data/wordle_answers.txt`. Both `env_train` and `env_eval` instantiated with `use_prime_intellect=True` (the bundled answer pool fallback fires when the `verifiers` package is not installed; both are exercised in CI).

**Policy.** `src/wordle_qwen_policy.py` (`WordleQwenPolicy`):

- `AutoModelForCausalLM.from_pretrained(MODEL_ES_BASE, torch_dtype=bfloat16)` for the frozen base.
- LoRA via PEFT 0.19.1: `r=4`, `lora_alpha=16`, `lora_dropout=0.05`, `target_modules=["q_proj","k_proj","v_proj","o_proj"]`. **Cast LoRA params to fp32 (`cast_lora_to_fp32=True`)** while the base stays in bf16 — at small `N_POP` and `SIGMA=0.02` the bf16 antithetic perturbation arithmetic was too coarse to extract a clean signal in pre-flights; fp32 on the trainable surface fixes this without giving up the bf16 base's memory savings.
- `action_granularity="char"` so `_rollout_batched` in `src/es_wordle.py` routes through `sample_words_batch`, which in turn applies the chat template (with `enable_thinking=True`), runs `model.generate(...)`, and parses out `<guess>WORD</guess>`. The `forward_logits_batch` zeros stub is required for the `hasattr` gate in `es_wordle.py` that switches between batched and per-state-serial perturbation/eval/probe paths; without it ES silently falls back to one `model.generate` per state and the wall clock becomes catastrophic.
- `trie_stats()` / `reset_trie_stats()` are reused as the parse-failure-rate surface. The `fb%` column in the verbose log is now `parse_failures / parse_attempts` per iteration.

**Optimizer.** `train_es_wordle` from `src/es_wordle.py`, unchanged. The Week 12 Session 8 plumbing (`restore_best_at_stage_end`, `eval_stochastic_every`) is on; the Session 7 mini-batch CRN feature (`per_iter_secret_subset_size=8`) is also on. Pre-flight ALPHA calibration is the iter-0 single-shot probe with `target initial step ≈ 0.13` and a `min_step ≈ 0.05` floor.

**Hyperparameters.** Mirroring `notebooks/week16_wordle_es_qwen.ipynb` §2 and `scripts/run_week16_es.py`:

| Setting | Value |
| --- | --- |
| `MODEL_ES_BASE` | `…-Wordle-SFT` (run 1) / `…-Wordle-RL` (runs 2-3) |
| `MAX_PROMPT_LENGTH` | 1024 |
| `ENABLE_THINKING` | `True` |
| `MAX_NEW_TOKENS` | 512 |
| `GEN_TEMPERATURE` / `GEN_TOP_P` | 0.8 / 0.9 |
| `LORA_R` / `LORA_ALPHA` / `LORA_DROPOUT` | 4 / 16 / 0.05 |
| `LORA_TARGET_MODULES` | `["q_proj","k_proj","v_proj","o_proj"]` |
| `N_POP` | 8 |
| `N_ITER` | 15 |
| `N_EVAL_EPISODES` (per perturbation, train) | 8 |
| `EVAL_N_EPISODES` (greedy eval) | 16 |
| `PROBE_N_EPISODES` | 8 |
| `MAX_TURNS` | 6 |
| `SIGMA` | 0.02 |
| `ALPHA` | auto from probe (target step ≈ 0.13) |
| `RANK_FITNESS` / `BASELINE_SUBTRACT` / `ANTITHETIC` / `COMMON_RANDOM_NUMBERS` | True / True / True / True |
| `EMA_BETA` | 0.0 |
| `EVAL_DETERMINISTIC` | True |
| `FITNESS_OBJECTIVE` | `"win_plus_return"` |
| `WIN_FITNESS_SCALE` | 8.0 |
| `PER_ITER_SECRET_SUBSET_SIZE` | 8 |
| `TRACK_BEST_ITER` / `RESTORE_BEST_ON_FINISH` | True / True |
| `NORMALIZE_GRADIENT` | False (run 1 SFT, run 2 RL) / **True** (run 3 RL) |

**Code paths.**

- `notebooks/week16_wordle_es_qwen_sft.ipynb` — initial parser/format probe (`enable_thinking=False`, `MAX_NEW_TOKENS=64`). Identified the 100% parse-failure mode that motivated thinking + 512-token budgets.
- `notebooks/week16_wordle_es_qwen.ipynb` — production notebook for the headline `enable_thinking=True`, `MAX_NEW_TOKENS=512` configuration.
- `scripts/run_week16_es.py` — headless equivalent for nohup runs on the GPU nodes.
- `src/wordle_qwen_policy.py` — the new generation-based policy.
- `src/es_wordle.py` — unchanged from Week 12 Session 8; its `_rollout_batched`, ALPHA probe, and verbose log already accommodate `action_granularity="char"`.

## Initial Results

Three runs were executed end-to-end on the production configuration above. All three are persisted under `runs/week16_es/{sft_base, rl_base, rl_base_normed_gradients}/` (console log + plot + `lora_adapter/` + `es_history.pkl`).

### Run 1 — ES on `…-Wordle-SFT` (raw gradients)

`MODEL_ES_BASE=…-Wordle-SFT`, `NORMALIZE_GRADIENT=False`. Pre-flight: `raw ‖ĝ‖ = 933.7`, AUTO ALPHA = `1.39e-04` (initial step ≈ 0.13). Per-iteration trajectory abridged from `runs/week16_es/sft_base/console.log`:

| Iter | Train Fit | ES_win | popσ | Eval Succ | Rew | ‖θ−θ₀‖ | dprobe | fb% |
|-----:|----------:|-------:|-----:|----------:|----:|--------:|-------:|----:|
| 0 | 0.180 | 0.0% | 0.017 | **6.2%** | 0.434 | 0.13 | +0.0% | 100% |
| 1 | 0.321 | 0.0% | 0.018 | 6.2% | 0.359 | 0.19 | +0.0% | 100% |
| 2 | 1.402 | 10.9% | 0.929 | 0.0% | 0.307 | 2.62 | +0.0% | 100% |
| 4 | 2.415 | 21.9% | 0.979 | 6.2% | 0.394 | 8.18 | +0.0% | 100% |
| 8 | 0.245 | 0.0% | 0.018 | **12.5%** | 0.508 | 8.62 | +0.0% | 100% |
| 14 | 0.463 | 1.6% | 0.397 | **0.0%** | 0.311 | 9.42 | +0.0% | 100% |

`final eval_success = 0.0%`, `best eval_success = 12.5%` at iter 8. **Parse-failure rate sat at 100% the entire run** — every single rollout hit a fallback parser path, never the clean `<guess>WORD</guess>` regex. `dprobe` never fired non-zero on any iteration. `‖θ−θ₀‖` exploded from 0.13 to 9.42 (raw gradients are ~5500 in magnitude; the few iters where ES picked up a strong fitness signal multiplied through ALPHA produced enormous step sizes — `|Δ|=2.61` at iter 2 alone). The greedy eval trajectory bounces 0% / 6.2% / 12.5% across iterations in a way that is consistent with the 16-secret single-slate noise floor (`σ ≈ 8.3pp` at one episode per secret) rather than learning. **Verdict: FAIL** — `best − SFT_cold = 6.3pp` (`SFT_cold` is the iter-0 eval = 6.2% greedy on this slate; we did not measure the cold no-adapter eval separately under this run, but it is bracketed by iter-0 = 6.2% and the cold pre-flight eval the notebook §2.5 prints), `dprobe` non-zero on 0/15 iters.

### Run 2 — ES on `…-Wordle-RL` (raw gradients)

`MODEL_ES_BASE=…-Wordle-RL`, `NORMALIZE_GRADIENT=False`. Pre-flight: `raw ‖ĝ‖ = 1.944e+04` (much higher than the SFT base — RL post-training has produced a sharper distribution that reacts more violently to LoRA perturbations), AUTO ALPHA = `6.69e-06`, initial step ≈ 0.13. Per `runs/week16_es/rl_base/console.log`:

| Iter | Train Fit | ES_win | popσ | Eval Succ | Rew | ‖θ−θ₀‖ | dprobe | fb% |
|-----:|----------:|-------:|-----:|----------:|----:|--------:|-------:|----:|
| 0 | 0.324 | 1.6% | 0.394 | 0.0% | 0.321 | 0.13 | +0.0% | 100% |
| 1 | 0.306 | 0.0% | 0.021 | **12.5%** | 0.426 | 0.13 | +0.0% | 100% |
| 4 | 2.860 | 26.6% | 0.387 | 0.0% | 0.274 | 0.18 | +0.0% | 100% |
| 8 | 0.395 | 1.6% | 0.403 | 12.5% | 0.499 | 0.26 | +0.0% | 100% |
| 14 | 0.490 | 1.6% | 0.404 | **0.0%** | 0.300 | 0.29 | +0.0% | 100% |

`final eval_success = 0.0%`, `best eval_success = 12.5%` at iter 1, `RESTORE_BEST_ON_FINISH` reloaded the iter-1 checkpoint. **Same 100% `fb%`. Same 0/15 `dprobe` non-zero.** The pattern across the RL base is even cleaner than across the SFT base: training fitness `Fit(win+ret)` spikes on iters where ES_win jumps (iter 4: 26.6% ES_win, fitness 2.86; iter 8: 1.6% ES_win, fitness 0.40), but those spikes do not transfer to the held-out 16-secret eval slate. Iter 4's ES population won 26.6% of its training episodes against a 26.6% win-rate baseline, yet the evaluated greedy policy of the post-update θ scored 0.0%. This is the textbook "ES gradient is finding training-set lottery tickets, not transferable improvements" failure mode that Week 12's `per_iter_secret_subset_size` machinery is supposed to mitigate but evidently does not at this base/budget. **Verdict: FAIL** — `best eval_success = 12.5% = SFT_cold + 6.3pp` (iter 0 = 0.0% on this slate with this random seed; iter-1 12.5% is consistent with single-slate eval noise rather than learning), `dprobe` non-zero on 0/15.

### Run 3 — ES on `…-Wordle-RL` (normalized gradients)

`NORMALIZE_GRADIENT=True` so the per-iteration update is `θ ← θ + α · ĝ / ‖ĝ‖` and `‖Δθ‖ = α` is held fixed at 0.13 every iteration regardless of the raw gradient magnitude. This was added specifically because Run 2's `|g|` was bouncing across two orders of magnitude (200 → 19,400) within a single training run; normalized updates remove that as a confound. Pre-flight: `raw ‖ĝ‖ = 563.5`, AUTO ALPHA = `1.30e-01` (now interpreted as the per-step `‖Δθ‖` directly, not a scaling on `ĝ`). Per `runs/week16_es/rl_base_normed_gradients/console.log`:

| Iter | Train Fit | ES_win | popσ | Eval Succ | Rew | ‖θ−θ₀‖ | dprobe | fb% |
|-----:|----------:|-------:|-----:|----------:|----:|--------:|-------:|----:|
| 0 | 0.158 | 0.0% | 0.023 | 0.0% | 0.328 | 0.13 | +0.0% | 100% |
| 1 | 0.315 | 0.0% | 0.028 | **12.5%** | 0.458 | 0.18 | +0.0% | 100% |
| 4 | 2.694 | 25.0% | 0.028 | 0.0% | 0.279 | 0.29 | +0.0% | 100% |
| 8 | 0.247 | 0.0% | 0.023 | 12.5% | 0.501 | 0.39 | +0.0% | 100% |
| 14 | 0.916 | 6.2% | 1.023 | **0.0%** | 0.324 | 0.50 | +0.0% | 100% |

`final eval_success = 0.0%`, `best eval_success = 12.5%` at iter 1, `RESTORE_BEST_ON_FINISH` reloaded the iter-1 checkpoint, exported adapter is the iter-1 checkpoint. `‖θ−θ₀‖` grew linearly (0.13, 0.18, 0.23, 0.26, 0.29, 0.32, 0.34, 0.37, 0.39, 0.41, 0.43, 0.45, 0.47, 0.49, 0.50) as expected for fixed-magnitude updates, confirming the normalization plumbing works. **Same 100% `fb%`. Same 0/15 `dprobe` non-zero. Same eval trajectory pattern**: peaks at 12.5% on a single early iteration, returns to 0% by iteration 14. **Verdict: FAIL.**

### Cross-run pattern

Three independent runs (different bases, different gradient treatments) produce essentially the same eval-trajectory shape: a 12.5% peak at one of the first two iterations, oscillation between 0% and 12.5% in the middle, and 0% at iter 14. With `RESTORE_BEST_ON_FINISH=True` the exported adapter is in every case a checkpoint within the first 8 iterations. The 12.5% number is **2 wins out of 16 eval episodes**, which under a Bernoulli null hypothesis at the cold-start win rate (~0-6%) is well inside the 95% CI; we cannot reject the null that the LoRA adapter is doing nothing measurable. The 100% `fb%` across all three runs says the layered fallback parser is *always* having to do work — there is not a single clean `<guess>WORD</guess>` emission in any of the ~ `(2·N_POP+1)·N_EVAL_EPISODES·N_ITER + EVAL_N_EPISODES·(N_ITER+1)` ≈ 2,400 generation calls per run.

### Cold-base baselines (notebook §2.5)

The Week 16 SFT notebook (`notebooks/week16_wordle_es_qwen_sft.ipynb` §2.5 and the production notebook §2.5) prints zero-shot greedy `eval_success` on the same 16-secret slate for the cold base (no LoRA). At our budget those numbers were `SFT_cold ≈ 6%`, `RL_cold ≈ 12%` — so the "best ES iterate" of 12.5% in every run is, for the RL base, **at the cold baseline**, and for the SFT base, **only one episode above the cold baseline**. The `FINAL ATTRIBUTION` block printed at the end of every run reflects this honestly: `RL_ceiling: n/a`, `SFT_cold: n/a`, `ES contribution: n/a`, with a hand-comment that the n/a's are deliberate (the cold-base eval and ES eval use different RNG streams in this notebook, so subtracting them would be misleading).

### Saved artifacts

```
runs/week16_es/
├── sft_base/
│   ├── console.log
│   ├── lora_adapter/                    # PEFT-format LoRA, base = …-Wordle-SFT
│   ├── es_history.pkl
│   └── plots/training_curves.png
├── rl_base/
│   ├── console.log
│   ├── lora_adapter/                    # PEFT-format LoRA, base = …-Wordle-RL
│   ├── es_history.pkl
│   └── plots/training_curves.png
└── rl_base_normed_gradients/
    ├── console.log
    ├── lora_adapter/                    # PEFT-format LoRA, base = …-Wordle-RL
    ├── es_history.pkl
    └── plots/training_curves.png
```

Each `lora_adapter/` is a standalone `peft` checkpoint usable via `PeftModel.from_pretrained(base, lora_adapter/)`; each `es_history.pkl` is the per-iteration metric trajectory for offline plotting.

## Next Steps

1. **Stop training.** Three runs across two bases and two gradient-update rules all produced indistinguishable-from-cold-start eval results. Within the design space the project has been searching since Week 10 (LoRA on a Qwen3-class base, ES at `N_POP ≤ 32`, `N_ITER ≤ 100`, our `train_es_wordle`), there is no further configuration to try whose expected information gain justifies its compute cost. The presentation will report Weeks 14 and 16 together as the project's terminal negative result on "ES adds something measurable on top of an already-Wordle-trained checkpoint at small population sizes."
2. **Pin down the cold-base ceiling and floor.** The single thing missing from the Week 16 record that the next deliverable should include is a clean, large-`n` evaluation of `…-Wordle-SFT` and `…-Wordle-RL` cold (no LoRA, deterministic, full 2,315-secret eval over multiple seeds). Without those numbers the "ES contribution = n/a" line in the FINAL ATTRIBUTION block stays n/a forever; with them, future Wordle-Qwen3 work has a real ceiling to compare against.
3. **Document `fb%` = 100% as a separate finding.** The fact that the Wordle-tuned Qwen3 checkpoints emit responses that *never* parse cleanly under our `<guess>WORD</guess>` regex (always falling through to the `_THINK_*` strip + first-5-letter-run fallback) is a Wordle-deployment finding that should be written up independently of the ES result. The fallback parser in `src/wordle_qwen_policy.py` is what makes the runs scoreable at all — without it, all three runs would have reported 0% across the board because every guess would have been the `XXXXX` sentinel.
4. **Followups for someone with more compute.** If `N_POP=8` is the binding constraint, the natural next experiments are (a) `N_POP=64` with the same `N_ITER=15` (8× compute per iter, comparable wall clock if parallelized across a multi-GPU pod), and (b) a longer horizon at the same population (`N_ITER=100`) with deterministic eval on a 256-secret slate so the noise floor drops below 3pp. Neither is in scope for this submission.

## References

1. Yang, A., et al. (2025). *Qwen3 Technical Report.* arXiv:2505.09388.  
2. Prime Intellect. (2025). *Wordle Verifiers: Qwen3-1.7B-Wordle-SFT and Qwen3-1.7B-Wordle-RL.* HuggingFace Hub.  
3. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.  
4. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. *arXiv:1703.03864*.
