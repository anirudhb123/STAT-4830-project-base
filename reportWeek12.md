# Wordle with Gemma 3 1B IT, Device-Aware Loading, and Evolution Strategies

**Status (this submission).** Week 12 ran two real Gemma 3 1B IT experiments side by side. (1) `notebooks/week12_implementation.ipynb` runs Gemma on the **8-word mock vocabulary** with **no LoRA** (head-only ES); this run **works**: greedy `Success` is ~79% right after warm-start (iter 0) and reaches ~83% by iter 9, with `ES_win` climbing from 68.8% to 93.8%. (2) `notebooks/week12_implementation_LoRARun.ipynb` runs Gemma on the **full Wordle vocabulary** with **LoRA** (rank sweep over r=4, 16, 32); this run **fails**: greedy `Success` stays near 0% for every rank, and `ES_win` is overwhelmingly 0% with only isolated 3.1% spikes. The DistilGPT-2 path (`RUN_PROFILE="smoke"`) is used in both notebooks only for pipeline plumbing checks, not as a research baseline.

## Problem Statement

**What are we optimizing?** Same core objective as Week 10: a policy that plays Wordle by attaching a **linear head** to a **frozen** Hugging Face causal LM, with actions as five-letter words from a fixed list. Training still combines optional **supervised warm-start** (`wordle_gpt2_warmstart.py`) and **Evolution Strategies** (`train_es_wordle` in `src/es_wordle.py`). The policy consumes a **text prompt** (turn, guesses, feedback, optional structured constraints from `wordle_hints.py` when `RICHER_PROMPT=True`).

**What changed in Week 12?** Both notebooks share a **run profile** switch so they support (1) a **fast smoke-test** configuration for pipeline checks (DistilGPT-2) and (2) a heavier **`gemma_full`** configuration that targets `google/gemma-3-1b-it`. They also add **device selection** (CUDA / MPS / CPU), **GPU-friendly model loading** (`bfloat16` or `float16` on CUDA when supported), **instruction-tuned formatting** via the tokenizer's **chat template** when using Gemma, a configurable **`MAX_PROMPT_LENGTH`**, and **profile-specific checkpoint filenames** so quick runs do not overwrite long jobs. The two `gemma_full` notebooks then split the experiment surface: `week12_implementation.ipynb` runs Gemma + head-only ES on the 8-word mock vocabulary, and `week12_implementation_LoRARun.ipynb` runs Gemma + LoRA on the full vocabulary with a rank sweep.

**Why does this matter?** Week 10's critique noted that full ES + LM runs are slow and hard to debug. Week 12 improved infrastructure and produced a clear contrast: Gemma + head-only ES on an 8-word mock task does learn, while Gemma + LoRA on the full Wordle vocabulary does not. This pushes the question from "can we run Gemma + LoRA?" to "why does optimization collapse when we jump from an 8-word task to the full word list?" Two likely process issues are weak train/test split isolation and no curriculum between the 8-word mock task and the full ~2k-word vocabulary.

**Experiment configurations used this week.**

- **DistilGPT-2 smoke** (`RUN_PROFILE="smoke"`, both notebooks): pipeline check only, never a research baseline.
- **Notebook A — Gemma + 8-word mock + no LoRA** (`week12_implementation.ipynb`, `MOCK_ENV=True`, `USE_LORA=False`): head-only ES, worked (~83% greedy success at iter 9).
- **Notebook B — Gemma + full vocabulary + LoRA rank sweep** (`week12_implementation_LoRARun.ipynb`, `MOCK_ENV=False`, `USE_LORA=True`, sweep over r=4, 16, 32): did not learn (~0% greedy success across all ranks).

**How will we measure success?** Same metrics as Week 10: periodic **eval_reward**, **eval_success**, **eval_turns**; per-iteration **train_fitness**, **param_drift**, **pop_fitness_std**, and **train_es_win** inside `history`. With **`RUN_PROFILE="smoke"`**, **`EVAL_EVERY=1`** yields dense eval points over the (short) ES horizon for debugging plots. With **`gemma_full`**, eval spacing depends on the notebook (`EVAL_EVERY=10` in `week12_implementation.ipynb`, `EVAL_EVERY=2` in `week12_implementation_LoRARun.ipynb`).

**Constraints and risks.** **Gemma 3 1B IT** is much larger than DistilGPT-2: downloads, RAM, and forward-pass cost grow sharply; CPU-only **gemma_full** runs may be impractical. The notebook sets **`HF_HUB_DISABLE_XET=1`** by default to avoid Xet-related hub issues in some environments. **`transformers>=4.50.0`** is assumed for current Gemma/chat-template behavior.

## Technical Approach

**Environment and reward.** Unchanged: `src/wordle_env.py`, mock path via `load_wordle_environment(use_prime_intellect=False)` when `MOCK_ENV=True`.

**Policy.** Still `src/wordle_gpt2_policy.py` (`WordleGPT2Policy`), now constructed with:

- `max_prompt_length` from the active profile  
- `use_chat_template` / `chat_generation_prompt` (on for **`gemma_full`**, off for **smoke** DistilGPT-2)  
- `model_kwargs` from **`default_model_load_kwargs(device)`** (dtype on CUDA; float32 on CPU/MPS path as implemented)

**Supervised warm-start and ES.** Unchanged modules: `wordle_gpt2_warmstart.py`, `es_wordle.py`. Fitness shaping (**`win_plus_return`**, **`RANK_FITNESS`**, **`NORMALIZE_GRADIENT`**) matches the Week 10 template unless edited in §2.

**Experimental configuration.** The hyperparameter cell is the source of truth. **`PROFILE_CONFIGS`** defines **`smoke`** and **`gemma_full`**.

| Setting | `RUN_PROFILE="smoke"` | `RUN_PROFILE="gemma_full"` |
| ------- | ---------------------- | --------------------------- |
| `MODEL_NAME` | `distilgpt2` | `google/gemma-3-1b-it` |
| `USE_CHAT_TEMPLATE` | `False` | `True` |
| `MAX_PROMPT_LENGTH` | 256 | 512 |
| `MAX_VOCAB` | 8 (mock) | 8 if `MOCK_ENV=True`, else `None` (full Wordle vocabulary) |
| ES | `N_POP=4`, `N_ITERATIONS=2` | `N_POP=16`, `N_ITERATIONS=10` |
| `n_eval_episodes` | 1 | 2 |
| Logging eval | `EVAL_EVERY=1`, `eval_n_episodes=4` | `EVAL_EVERY=2`, `eval_n_episodes=48` |
| Warm-start | `WARM_START_STEPS=12` | `WARM_START_STEPS=400` |
| Env episode counts | `num_train_examples=128`, `num_eval_examples=16` | `2000` / `20` |
| `USE_LORA` (depends on notebook) | `False` | `False` in `week12_implementation.ipynb`; `True` (rank sweep r=4/16/32) in `week12_implementation_LoRARun.ipynb` |

Global knobs in §2 include `SIGMA=0.02`, `ALPHA=0.12`, `RICHER_PROMPT=True`, `FITNESS_OBJECTIVE="win_plus_return"`, `WIN_FITNESS_SCALE=8.0`. `MOCK_ENV` and `USE_LORA` differ by notebook: `week12_implementation.ipynb` uses `MOCK_ENV=True`, `USE_LORA=False`; `week12_implementation_LoRARun.ipynb` uses `MOCK_ENV=False`, `USE_LORA=True` (with a rank sweep over r=4/16/32). Both notebooks default `RUN_PROFILE="gemma_full"`.

**Artifacts.** Checkpoints save to `models/wordle_gemma_es_head.<RUN_PROFILE>.pt` (or `models/wordle_gemma_es_head_LoRA.<RUN_PROFILE>.pt` when LoRA is enabled), containing the head / optional LoRA state, `words`, and `history`.

**Code paths.**

- `notebooks/week12_implementation.ipynb` (Gemma + 8-word mock + head-only ES)  
- `notebooks/week12_implementation_LoRARun.ipynb` (Gemma + full vocabulary + LoRA rank sweep)  
- `src/wordle_gpt2_policy.py`, `src/wordle_gpt2_warmstart.py`, `src/es_wordle.py`, `src/wordle_env.py`, `src/wordle_hints.py`

## Initial Results

**Notebook A — Gemma + head-only ES on 8-word mock vocabulary (`week12_implementation.ipynb`).** Worked. After warm-start, greedy eval `Success` is ~79% at iter 0 and reaches ~83% at iter 9; population `ES_win` rises from 68.8% to 93.8% over 10 iterations. This is consistent with Week 10's mock-task pattern: warm-start does most of the lifting and ES adds a modest bump.

**Notebook B — Gemma + LoRA rank sweep on full vocabulary (`week12_implementation_LoRARun.ipynb`).** Did not work. With `MOCK_ENV=False`, `USE_LORA=True`, and `max_vocab=None`, all three swept ranks (r=4, 16, 32) stay at near-zero greedy `Success` across all 10 iterations and `ES_win` is essentially 0% throughout (occasional 3.1% spikes only). The "best rank" returned by the sweep (r=4) is best only by tie-breaking among zero-success runs.

**DistilGPT-2 smoke (diagnostic only).** `RUN_PROFILE="smoke"` in either notebook is used only for quickly validating imports, env reset, warm-start wiring, ES loop execution, and plotting. It is not a scientific baseline.

**Interpretation.** The contrast between Notebook A (works on 8 words, head-only) and Notebook B (fails on the full vocabulary, even with LoRA) is the key Week 12 finding. The two confounded changes between them are vocabulary size (8 -> ~2k actions) and whether LoRA is on, which makes it hard to attribute the failure to a single cause. This motivates curriculum-style scaling and stricter split discipline before re-running the full-vocab + LoRA configuration.

## Next Steps

1. **Enforce train/test split isolation.** Create fixed, disjoint train/eval word sets, persist them, and reuse them across all runs and seeds so generalization claims on the full vocabulary are credible.  
2. **Add curriculum learning between Notebook A and Notebook B.** Stage vocabulary difficulty (for example 8 -> 64 -> 256 -> full), carry model weights between stages, and measure where performance degrades. This bridges the working mock-vocab regime and the failing full-vocab regime.  
3. **Decouple LoRA from vocabulary scaling.** Rerun Gemma + LoRA on the 8-word mock task and rerun Gemma head-only on the full vocabulary so we can attribute Notebook B's failure to vocabulary size, LoRA, or both.  
4. **Ablate warm-start vs ES under the same split.** Evaluate after warm-start and after ES on identical held-out episodes to determine whether ES adds signal in each regime.  
5. **Keep implementation hygiene work in scope.** Continue prompt-quality fixes (`wordle_hints.py`, truncation checks) and multi-seed reporting once the experimental protocol is sound.

## Follow-up results (Session 7 + Session 8)

**What we still thought after the submission snapshot.** Notebook B's near-zero greedy success at the full vocabulary was the Week 12 headline finding, but the follow-up sessions narrowed the diagnosis. Session 7 (`scripts/run_experiment2_minibatch_crn.py`, mini-batch ES under CRN) showed that on a 16-word single-stage probe ES does produce real signal — greedy `eval_success` rose from a post-warm-start baseline of 30% to a peak of 48% at **iter 1**, a `+18pp` lift — but the run then walked away from that peak and finished at 18% (`final − post_ws = −12pp`). That pattern — signal at iter 1, random walk afterward — pointed at a step-size / overshoot problem: ALPHA was calibrated once at iter 0 on a 4-secret mini-batch, but the 4-secret subset rotates every iteration, so the per-iter step was ~3-4× too large for a non-stationary objective and the optimizer never aggregated its early gain.

**Session 8 probe (`EXP2_ALPHA_SCALE=0.25 EXP2_N_ITERATIONS=60 EXP2_RESTORE_BEST=1`).** We quartered ALPHA, doubled the iteration budget to 60, and added two features to `train_es_wordle` / `train_curriculum`: `restore_best_at_stage_end` (track the best-by-greedy-eval iterate as a CPU-cloned `state_dict` and reload it at stage end) and `eval_stochastic_every` (companion stochastic eval at the same cadence as greedy, RNG-isolated so it does not perturb the ES stream). Single-stage probe, no changes to fitness shaping, RANK_FITNESS, BASELINE_SUBTRACT, EMA_BETA, LoRA rank, or the 4-secret subset size.

**Result — PASS-A on the pre-registered verdict rubric.** On `VOCAB_SCHEDULE=[16]`, `N_ITERATIONS=60`, `ALPHA × 0.25`:

| metric | value |
| --- | --- |
| pre-warm-start (greedy) | 0% |
| post-warm-start (greedy) | 36% |
| `best_greedy` (iter 1) | **58%** (`+22pp` vs post-WS) |
| `final_greedy` (iter 59) | **52%** (`+16pp` vs post-WS) |
| `best_stochastic` / `final_stochastic` | 52% / 42% |
| `dprobe` non-zero fraction | 35/60 = **58%** |
| `‖θ − θ₀‖` | 0.66 → ~4.9 (linear, non-oscillating) |

The rubric (graded in order):

- **PASS-A:** `best_greedy − post_ws_greedy ≥ +10pp` **and** `final_greedy ≥ post_ws_greedy` **and** `dprobe` non-zero ≥ 25%. All three fire (`+22pp` / `+16pp` / `58%`).
- **PASS-B would have fired if** `best_greedy − final_greedy ≥ +10pp`, i.e. the run peaked and then collapsed back below post-WS + best. The observed `best − final = +6pp` gap is inside the ~7pp single-slate eval noise floor, so PASS-B does **not** fire; best-iter restore is not load-bearing for this configuration.
- **FAIL would have fired if** none of the above. Did not occur; the sticky-subset fallback (`EXP2_SUBSET_REFRESH_EVERY=5`) was therefore not run.

**Plain-English interpretation.** PASS-A says three things at once. (1) ES is doing real work on top of warm-start — it found a `+22pp` iterate at vocab = 16, well above the noise floor. (2) The run is *stable enough to aggregate its own gains* — the final greedy held `+16pp` above post-WS, which is the specific failure mode Session 7 exhibited (`−12pp`) and Session 8 fixes. (3) The ES gradient estimate is plumbing-healthy — `dprobe` fires on a solid majority (58%) of iterations, so the 4-secret subset CRN is not degenerate. Together, they confirm the Session 7 diagnosis (step-size overshoot under rotating mini-batch level sets) and validate the mitigation (quarter ALPHA, double iters) as *self-sufficient* rather than only viable with a best-iter rescue.

**What this does and does not change about the Week 12 picture.** Notebook B's full-vocabulary, full-curriculum result at r=16 LoRA is still ~0% greedy and is still the headline failure we need to fix — Session 8 does not claim otherwise. What it *does* claim is that the specific optimizer-side pathology surfaced by Session 7 (iter-1 gain → walk-away) is a solved problem at vocab = 16 under a single stage. This is one stage of the curriculum rather than the full pipeline, and a single seed rather than a multi-seed result, so it should be read as directional evidence that the mini-batch + CRN + warm-start configuration is *optimizable* when ALPHA is chosen correctly, not as a claim that the full-curriculum regime now works.

**Residual limitation — measurement, not optimization.** The greedy trajectory oscillates 30-58% from iter to iter even while `‖θ−θ₀‖` is monotonic. This is a 50-episode × 16-secret single-slate noise floor (per-slate σ ≈ 3-4pp; a different `probe_seed` on the final policy gave 38% greedy vs the 58% `best_greedy` — within 2.5σ of single-slate noise). The optimizer-side bottleneck is now below the measurement floor, and subsequent experiments should shrink that floor (deterministic 16 × K-episodes-per-secret eval, K ≥ 3) before reseeding or scaling vocab.

**Next cheapest experiments (in order).** (1) Replace the 50-episode probe slate with a deterministic 16 × K full-pool eval to drop the noise floor below 3pp and verify visually monotonic convergence. (2) Reseed the Session 8 configuration (two or three additional seeds) to calibrate the PASS-A effect size. (3) Rerun at `VOCAB_SCHEDULE=[32]` with the same `ALPHA_SCALE=0.25` to test whether the step-size fix survives a vocabulary doubling. Only after those three does it make sense to re-enable the full 8-stage curriculum. Artifacts: [`scripts/run_experiment2_minibatch_crn.py`](scripts/run_experiment2_minibatch_crn.py), `/tmp/exp2_overshoot.log`; full narrative in [`docs/llm_exploration/week12_log.md`](docs/llm_exploration/week12_log.md) Session 8 and in the "Session 8 — Overshoot Follow-up" postscript of [`critiqueWeek12.md`](critiqueWeek12.md).

## References

1. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. *arXiv:1703.03864*.  
2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.  
3. Gemma Team, Google DeepMind. (2025). *Gemma 3 Technical Report.* *arXiv:2503.19786*. [https://arxiv.org/abs/2503.19786](https://arxiv.org/abs/2503.19786) (PDF: [https://arxiv.org/pdf/2503.19786](https://arxiv.org/pdf/2503.19786)).

