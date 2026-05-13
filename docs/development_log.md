# Development Log

## Week 16 (May 8 - May 12, 2026)

### Overview
Pivoted from "warm-start a generic LM and let ES specialize it on Wordle" (Weeks 10-14) to "take the most-Wordle-trained 1.7B-param checkpoint that exists and ask whether ES adds anything on top." Built `src/wordle_qwen_policy.py` (`WordleQwenPolicy`) — a generation-based policy that wraps Prime Intellect's Qwen3 Wordle checkpoints with LoRA, calls `model.generate(...)`, and parses `<guess>WORD</guess>` with a layered fallback. Ran three production ES configurations end-to-end. Headline result: **all three converged to `final eval_success = 0.0%`, `best eval_success = 12.5%` at iter 1 or 8 — statistically indistinguishable from cold-base eval noise on the 16-secret slate.** This is the project's terminal experiment on small-population ES + LoRA on a Wordle-tuned Qwen3 base.

### Key Decisions

**1. Skip Our Own Warm-Start Entirely**
- **Decision:** Drop the per-stage supervised warm-start loop (Weeks 11-14) and start ES directly from `PrimeIntellect/Qwen3-1.7B-Wordle-SFT` (or `…-Wordle-RL`).
- **Rationale:** Three consecutive weeks of "warm-start does the lifting; ES adds noise" had been the consistent pattern. The Wordle-specific SFT/RL checkpoints from Prime Intellect's *Wordle Verifiers* release *are* the warm-start, and they were trained with a real RLHF-scale budget. ES is the only training signal the LoRA adapter ever sees in Week 16.
- **Consequence:** No more `train_curriculum`, no `wordle_gpt2_warmstart` calls, no `VOCAB_SCHEDULE`. Just `train_es_wordle` on the LoRA adapter against the full 2,315-secret pool with `per_iter_secret_subset_size=8` mini-batch CRN.

**2. Drop Char-Mode Generation; Use `model.generate(...)` with a `<guess>WORD</guess>` Parser**
- **Decision:** Retire `ACTION_GRANULARITY="char"` (autoregressive 5-letter generation under a vocabulary trie mask, the Week 14 design). Generate end-to-end with `model.generate(...)` and parse `<guess>WORD</guess>` from the output.
- **Rationale:** Week 14's char-mode result showed persistent 60-90% trie fallback rate across the entire run, meaning the trie mask was redirecting most token emissions and the LM's pre-mask distribution was approximately uninformative. The post-mask sampling distribution had almost no LoRA-controlled gradient surface.
- **Implementation:** New `src/wordle_qwen_policy.py` (`WordleQwenPolicy`) with three failure-mode handlers in `_parse_word_from_text`:
  1. Last `<guess>...</guess>` regex match (clean path; never fired in production runs).
  2. First 5-letter alphabetic run after stripping `<think>...</think>` blocks and residual XML-ish tags. `_FALLBACK_SKIP` denylist (`THINK`, `THERE`, `WHICH`, `SHOULD`, …) prevents common reasoning fragments from beating real words.
  3. `XXXXX` sentinel — env scores it as `Invalid guess`, reward=0, turn consumed; ES sees the perturbation as low-fitness and avoids it.

**3. Format/Parser Probe Notebook (`notebooks/week16_wordle_es_qwen_sft.ipynb`)**
- **Decision:** Run a fast probe at `enable_thinking=False`, `MAX_NEW_TOKENS=64` *before* the production runs to surface format-related failure modes.
- **Result:** **100% parse failure.** The Wordle SFT/RL checkpoints emit `<think>...</think>` blocks regardless of the chat-template flag; the 64-token budget was exhausted inside the thinking opener and the parser never saw `<guess>`.
- **Lesson learned:** This is exactly the cheap-experiment-first probe Week 14's critique said we needed. It cost ~30 min of compute and immediately ruled out the "compact emission" configuration without the cost of a full ES run. **Production configuration switched to `enable_thinking=True`, `MAX_NEW_TOKENS=512`.**

**4. Bf16 Base + Fp32 LoRA**
- **Decision:** `cast_lora_to_fp32=True` so the trainable surface is fp32 while the base stays bf16.
- **Rationale:** At `N_POP=8` and `SIGMA=0.02`, pre-flights showed bf16 antithetic perturbation arithmetic was too coarse to extract a clean signal from a ~1.6M-dim LoRA. The all-bf16 alternative is a roughly 50% memory saving on the trainable params (~3MB vs ~6MB) at the cost of much noisier ES gradients.
- **Implementation:** `src/wordle_qwen_policy.py` lines ~219-227 — iterate `self.lm.parameters()` post-PEFT-wrap and cast `requires_grad=True` params with `p.data = p.data.float()`. Frozen base parameters are untouched.

**5. Three Production Runs**
- Run 1 — ES on `…-Wordle-SFT`, raw gradients (`runs/week16_es/sft_base/`).
- Run 2 — ES on `…-Wordle-RL`, raw gradients (`runs/week16_es/rl_base/`).
- Run 3 — ES on `…-Wordle-RL`, **normalized gradients** (`NORMALIZE_GRADIENT=True` so `θ ← θ + α · ĝ / ‖ĝ‖`, `‖Δθ‖ = α = 0.13` per step) (`runs/week16_es/rl_base_normed_gradients/`).
- All three: `N_POP=8`, `N_ITER=15`, `EVAL_N_EPISODES=16`, `MAX_TURNS=6`, `SIGMA=0.02`, `LORA_R=4`, `RESTORE_BEST_ON_FINISH=True`, `RANK_FITNESS=True`, `BASELINE_SUBTRACT=True`, `ANTITHETIC=True`, `COMMON_RANDOM_NUMBERS=True`, `EMA_BETA=0.0`.
- Why three runs? Run 1 surfaced the SFT-base behavior; Run 2 swapped to the RL base to see if the additional Prime Intellect RL stage created a different local geometry; Run 3 added gradient normalization specifically to remove the `‖ĝ‖` bouncing two orders of magnitude as a confound (Run 2 had per-iter `|g|` ranging 200 → 19,400).

### Results

**All three runs: identical eval-trajectory shape.** Best `eval_success = 12.5%` at an early iteration, oscillation between 0% and 12.5% middle, `final eval_success = 0.0%` at iter 14 (with `RESTORE_BEST_ON_FINISH=True` reloading the iter-1 or iter-8 checkpoint as the exported adapter).

| Run | Base | Normalize Grad | Best Iter | Best Eval Succ | Final Eval Succ | dprobe non-zero | fb% |
|---|---|---|---|---|---|---|---|
| 1 | …-Wordle-SFT | False | 8 | 12.5% | 0.0% | 0/15 | 100% |
| 2 | …-Wordle-RL | False | 1 | 12.5% | 0.0% | 0/15 | 100% |
| 3 | …-Wordle-RL | True | 1 | 12.5% | 0.0% | 0/15 | 100% |

**12.5% = 2 wins / 16 eval episodes.** Under a Bernoulli null at the cold-base win rate (~6-12% on the same slate per the §2.5 cold-base eval), the 12.5% peak is well inside the 95% CI. We cannot reject the null that the LoRA adapter is doing nothing measurable.

**`fb%` = 100% across all three runs across every iteration.** This is the most informative finding: the layered fallback parser in `_parse_word_from_text` is *always* doing the work — there is not a single clean `<guess>WORD</guess>` emission in any of the ~2,400 generation calls per run. The Wordle-tuned Qwen3 checkpoints, even with `enable_thinking=True` and 512 tokens of generation budget, do not emit responses that match our `<guess>WORD</guess>` regex; the parser is choosing guesses out of the model's reasoning text.

**Run 2 vs Run 3 ablation:** Normalized gradients held `‖Δθ‖` at exactly 0.13 every iteration (vs Run 2's iter-2 spike to `|Δ| = 2.61`). The eval trajectories were essentially identical. The ALPHA / step-size axis is **not** the binding constraint at this base/budget — a real finding given that previous critiques had pointed at step size as the next thing to tune.

### Failed Attempts

**1. `enable_thinking=False`, `MAX_NEW_TOKENS=64` (Pre-production probe)**
- 100% parse failure. The model exhausted its 64-token budget inside `<think>` and never reached `<guess>`. Documented as the §2.5 probe in `notebooks/week16_wordle_es_qwen_sft.ipynb`.
- **Lesson:** The Wordle-tuned Qwen3 checkpoints have the `<think>` block baked into their training; flipping the flag does not turn it off. Generation budget must accommodate it.

**2. Run 1 with raw gradients on the SFT base**
- Run 1's `‖θ−θ₀‖` exploded from 0.13 to 9.42 over 15 iterations (raw gradient `|g|` averaged ~5500). Iter-2's single-step `|Δ| = 2.61` against the calibration target of 0.13 — a 20× overshoot. Best eval 12.5% at iter 8 was a single early lottery; final 0%.
- This is what motivated Run 3's `NORMALIZE_GRADIENT=True`.

**3. Expecting "best eval_success" to mean what it normally means**
- All three runs print "best eval_success = 12.5%" prominently in the FINAL ATTRIBUTION block. With `RESTORE_BEST_ON_FINISH=True` the exported adapter *is* the best-iter checkpoint. The temptation to read this as "ES found a +6pp lift" is strong and wrong: the cold-base eval on the same slate is 6-12% greedy, and 2 wins out of 16 episodes is inside the noise band.
- **Lesson:** The FINAL ATTRIBUTION block needs an "ES contribution" line that subtracts the cold-base eval under the same RNG stream. Today it prints `n/a` because the cold and ES evals use different RNG streams. Fix in next iteration.

### Code Changes

- `src/wordle_qwen_policy.py` (new, ~450 LoC): `WordleQwenPolicy` class. Wraps `AutoModelForCausalLM` + LoRA + chat template + `model.generate(...)` + layered `<guess>` parser. Implements the ES-required `forward_logits_batch` zeros stub (see `es_wordle.py` `hasattr` gate), `sample_words_batch`, `format_action_xml`, `get_action`, and the `trie_stats`/`reset_trie_stats` hooks repurposed as a parse-failure-rate counter for the verbose log.
- `scripts/run_week16_es.py` (new, ~410 LoC): headless equivalent of the Week 16 production notebook, `--artifacts-dir` for Slurm/nohup-friendly persistence, `--skip-alpha-probe` + `--alpha` for re-running with a known-good ALPHA, `--no-plots` for log-only runs.
- `notebooks/week16_wordle_es_qwen.ipynb` (new): the headline `enable_thinking=True`, `MAX_NEW_TOKENS=512` notebook.
- `notebooks/week16_wordle_es_qwen_sft.ipynb` (new): the format/parser probe notebook (`enable_thinking=False`, `MAX_NEW_TOKENS=64`).
- `runs/week16_es/{sft_base, rl_base, rl_base_normed_gradients}/` (new artifacts): per-run console.log + `lora_adapter/` (PEFT-format) + `es_history.pkl` + `plots/training_curves.png`. Three runs, ~30 MB on disk total.
- `src/es_wordle.py` (unchanged in behavior): the existing `_rollout_batched`'s `action_granularity == "char"` branch and the `hasattr(policy, "forward_logits_batch")` gate accommodate the new policy without modification.

### Open Questions

1. **Is `fb%` = 100% an artifact of the parser regex, or a real property of the Wordle-tuned Qwen3 outputs?** A diagnostic counter that disaggregates `fb%` into clean-`<guess>` matches vs fallback-real-word matches vs `XXXXX`-sentinel matches would settle this in one extra iteration.
2. **Would `N_POP=64`, `N_ITER=100` produce a measurable ES lift?** Salimans et al. used `N_POP` in the thousands. We ran 8 because each iter is hours on Qwen3. A wall-clock-budgeted `N_POP=32`, `N_ITER=30` sanity check is the cheapest experiment that could change the headline answer.
3. **What is the cold-base ceiling?** Notebook §2.5 prints `SFT_cold ≈ 6%`, `RL_cold ≈ 12%` on the 16-secret slate, but those are noisy. A clean large-`n` eval (full 2,315-secret pool, multiple seeds, deterministic) of `…-Wordle-SFT` and `…-Wordle-RL` cold (no LoRA) is the one missing number that would let `ES contribution` go from `n/a` to a real subtraction.

### Time Spent

- ES production runs (3× ~6-8 GPU-hours): ~24 GPU-hours
- Format/parser probe + cold-base eval: ~2 GPU-hours
- New code (`wordle_qwen_policy.py`, `run_week16_es.py`, two notebooks): ~12 hours
- Documentation (this entry, week 16 report, week 16 critique): ~4 hours
- **Total: ~30 hours of human time, ~26 GPU-hours of compute**

### LLM Usage Log

- **Claude (May 8-12):** Designed the layered `_parse_word_from_text` fallback (the `_THINK_*` strip + `_FALLBACK_SKIP` denylist for common reasoning 5-grams), wrote the `forward_logits_batch` zeros stub explanation, debugged the `enable_thinking` chat-template TypeError fallback. ~5 hours across the policy implementation and the Week 16 report/critique drafts.
- **Cursor AI (May 8-12):** Most edits in `src/wordle_qwen_policy.py` and `scripts/run_week16_es.py`; LoRA-cast-to-fp32 wiring, ALPHA probe / RNG snapshot/restore boilerplate, `--artifacts-dir` plumbing. ~6 hours.
- **ChatGPT (May 9):** Sanity check on the `‖θ − θ₀‖` arithmetic for the normalized-gradient ablation (Run 3); ~20 min.

---

## Week 14 (April 24 - April 30, 2026)

### Overview
Retired the Gemma 3 1B IT pipeline that ran from Weeks 10-12 and swapped the LM stack to **Qwen3 1.7B base** with **autoregressive 5-letter character generation under a vocabulary trie mask** (`ACTION_GRANULARITY="char"`). Motivation: Week 12 Session 8's PASS-A result (`+22pp` at vocab=16, mini-batch CRN, quartered ALPHA) had not generalized past a single curriculum stage, and the architecture+representation swap was an attempt to widen the operating window. Headline result: **the `qwen_full` configuration FAIL on the pre-registered rubric** at every stage tried (`VOCAB_SCHEDULE=[16, 32, 64]` planned, killed after stage 2). The most informative diagnostic was the new `fb%` (trie fallback rate) column staying at 60-90% across both stages — meaning the trie mask was doing all the structural work and the LM's pre-mask distribution was approximately uninformative for ES.

### Key Decisions

**1. Backbone Swap: Gemma 3 1B IT → Qwen3 1.7B**
- **Decision:** `MODEL_NAME = "Qwen/Qwen3-1.7B"` for `RUN_PROFILE="qwen_full"`. Smoke profile stays on `distilgpt2`.
- **Rationale:** Two motivations stacked. (a) Architectural prior — Qwen3 1.7B's English short-form prior reads stronger than Gemma 3 1B IT's in qualitative spot-checks of zero-shot Wordle prompts. (b) Newer base; better chance the Wordle-specific Qwen3 derivatives released by Prime Intellect (tracked since Week 13 as a candidate Week 16 pivot) become available before the project ends.
- **Compatibility:** Qwen3 needs `transformers>=4.51.0` and `jinja2>=3.1.0` for its chat template. Both were already in `requirements.txt` from Week 12's PEFT bump.

**2. Action Representation: Single-Softmax Word Head → Char-Mode + Trie Mask**
- **Decision:** Add `ACTION_GRANULARITY = "char"` knob to the policy and the ES rollout dispatcher. In char mode, the policy emits five tokens autoregressively, each masked by `_WordleVocabTrie` against the prefix emitted so far so every emission is in-vocabulary by construction.
- **Rationale:** The single-softmax word head bottlenecks ES gradient information through a logit-vector dimension equal to the active vocabulary size, which Week 12's LoRA-rank sweep was implicitly trying to fix. Char-mode replaces that classifier with the LM's own pretrained head distribution + an external trie-mask postprocessor; the LoRA-controlled gradient surface is the LM's attention modules, not a head whose dimensionality changes per curriculum stage.
- **Implementation:** `_rollout_batched` in `src/es_wordle.py` already had a `getattr(policy, "action_granularity", "word")` switch (added in Week 12 for the planned char-mode branch); extending it for production was a thin wrapper around `WordleGPT2Policy.sample_words_batch` plus the trie-mask machinery.

**3. New Diagnostics: `fb%` and `trie_steps` in the Verbose Log**
- **Decision:** Add `trie_stats() -> {trie_steps, trie_fallbacks, trie_fallback_rate}` to the policy class and surface `fb%` (= `trie_fallbacks / trie_steps` per ES iter) and `trie_steps` in `train_es_wordle`'s per-iteration verbose log.
- **Rationale:** Char-mode introduces a new failure mode the Week 12 logs cannot diagnose: the LM emitting characters that do not extend any legal Wordle prefix, with the trie mask redirecting them. `fb%` is the per-iteration count of those redirections divided by the total mask applications. We pre-registered "if `fb%` is consistently above 50%, the trie mask is dominating the LM and ES has no gradient surface to push against."

**4. Production Config (`RUN_PROFILE="qwen_full"`)**
- LoRA r=8, α=16, target_modules=`["q_proj","k_proj","v_proj","o_proj"]`, dropout=0.05.
- ES: `N_POP=32`, `N_ITERATIONS=100/stage`, `n_eval_episodes=32`, `EVAL_EVERY=5`, `eval_n_episodes=16`.
- Curriculum: `VOCAB_SCHEDULE=[16, 32, 64]` (planned to extend to `128, 256, 512` if stage 1 passed).
- Warm-start: 400 episodes per stage (proportional to vocab).
- Global ES: `SIGMA=0.02`, `RANK_FITNESS=True`, `BASELINE_SUBTRACT=True`, `EMA_BETA=0.0`, `WIN_FITNESS_SCALE=8.0`, `RICHER_PROMPT=True`, `FITNESS_OBJECTIVE="win_plus_return"`.

### Results

**Stage 1 — `VOCAB_SCHEDULE=[16]` over 100 ES iterations.**
- Pre-WS greedy `Success` = 4%; post-WS = 28%; `best_greedy = 34%` at iter 4 (`+6pp` vs post-WS, inside the ~7pp single-slate noise floor); `final_greedy = 22%` at iter 99 (`−6pp`, inside the same band).
- `dprobe` non-zero on 9/100 iters (= 9%, vs the 25% pre-registered pass floor).
- `cos(ĝ)` median ≈ 0.00 with a few isolated +0.05 spikes.
- `‖θ − θ₀‖` grew approximately linearly to 5.1 — the optimizer was *moving*, but not climbing a level set.
- **`fb%` ranged 60-90%** across the entire run across all population members, including the unperturbed θ₀.
- **Verdict: FAIL** on all three pass criteria.

**Stage 2 — `VOCAB_SCHEDULE=[32]`, carrying the stage-1 adapter.**
- Post-WS greedy `Success` dropped to 14% on the new 32-secret slate. ES added essentially nothing on top: 100 iters, `best_greedy = 16%`, `final_greedy = 10%`, `dprobe` non-zero on 6/100 iters.
- **Verdict: FAIL.** Killed the run after stage 2 instead of running stage 3.

**Mechanistic interpretation of `fb%` = 60-90%.** The LM's pre-mask distribution is putting most probability on letters that do not extend any legal Wordle prefix; the trie mask redirects almost every token to its top legal alternative. The post-mask distribution the policy actually samples from is therefore approximately uniform over each prefix's legal continuations — a uniform-over-legal sampling distribution has almost no LoRA-controlled gradient surface for ES to push against. The policy's behavior is mask-determined, not LM-determined; ES cannot improve a policy whose decisions are made by string-postprocessing.

### Failed Attempts

**1. Treating Char-Mode + Trie Mask as a Tunable**
- The 60-90% `fb%` is structural: mask vs LM is operating at cross-purposes for a generic LM that was not pretrained on legal-Wordle-prefix emission. No amount of LoRA, ALPHA scheduling, or `n_eval_episodes` is going to change that ratio meaningfully.
- **Lesson:** If we ever return to char-mode, the right hypothesis to test is not "trie mask vs no mask" but "trie mask vs additive logit *bias* (penalty proportional to how illegal the next character is)" — preserving the LM's pretrained gradient surface against the constraint instead of overwriting it.

**2. No `qwen_probe` Profile**
- The smoke profile (DistilGPT-2, two ES iters) validated end-to-end plumbing but not "is the experiment going to teach us anything." The Week 12 `PROBE_VOCAB=4` cell played that role for the LoRA sweep; Week 14 had no analogue.
- Failure mode (flat `dprobe`, persistent `fb%`) was visible inside the first ~5 ES iters but the run was not killed until stage 2 had also failed overnight.
- **Lesson:** Add a `qwen_probe` profile (`N_POP=8`, `N_ITERATIONS=10`, single stage) that exercises the new diagnostics in under an hour. Done in Week 16 as the §2.5 format/parser probe.

**3. Pre-Registered Pass Criterion Was a Week-12 Carryover**
- "`best_greedy − post_ws_greedy ≥ +10pp` AND `final_greedy ≥ post_ws_greedy` AND `dprobe` non-zero on ≥ 25% of iters" was tuned for the word-softmax head, where `dprobe` measures whether the perturbed policy's argmax over a 2,315-way classifier moves on the held-out probe. In char-mode, `dprobe` measures whether the masked autoregressive sampler produces a different *5-character word* — a strictly higher-bar event when `fb%` is 60-90%.
- **Lesson:** The rubric needs to be re-derived whenever the action representation changes, not just when the model changes.

**4. Confounded Two Changes At Once**
- Swapped the base LM (Gemma → Qwen3) AND the action representation (word-softmax → char-mask) in the same experiment. The FAIL is unattributable: we cannot say whether Qwen3 with the original word-softmax head would have worked, nor whether Gemma with the new char-mode would have worked.
- Practical reading given the Week 16 plan to skip our own training entirely: low-priority backfill. Cited in the Week 14 critique as a methodology note.

### Code Changes

- `notebooks/week14_wordle_es_lora_run.ipynb` (new): `RUN_PROFILE = {"smoke", "qwen_full"}`, char-mode wiring, `fb%`/`trie_steps` panel additions to the standard Week-12 plot grid.
- `src/wordle_gpt2_policy.py`: extended for `action_granularity="char"` — added `_WordleVocabTrie` (build per stage), `sample_words_batch` autoregressive char-loop with mask, `trie_stats()`/`reset_trie_stats()` hooks for the verbose log.
- `src/es_wordle.py`: added the `fb%`/`trie_steps` columns to the verbose-log row builder; the `getattr(policy, "action_granularity", "word")` switch was already present from Week 12 and required no change.
- Loaded checkpoints: `models/wordle_qwen_es_head.qwen_full.pt` + `models/wordle_qwen_es_history.qwen_full.pkl`.

### Open Questions

1. **Would Qwen3 + char-mode work at the Week-12 Session-8 operating point** (`per_iter_secret_subset_size=4`, quartered ALPHA, `N_ITER=60`)? We did not transfer those Session-8 settings into Week 14; the production config used a fresh ALPHA calibration at iter 0 against the full 16-secret stage-1 pool.
2. **Is the trie-mask-dominates-LM story specific to Qwen3, or would Gemma + char-mode show the same `fb%` = 60-90%?** Single-axis backfill that would disambiguate Week 14's confound.
3. **Does the `fb%` story flip if char-mode operates over a per-position legal-character mask at evaluation only** (not during sampling), so the LM samples freely and we read the legal argmax from its post-softmax distribution?

### LLM Usage Log

- **Claude (Apr 24-30):** Char-mode design discussions, the `_WordleVocabTrie` data-structure choice, mechanistic interpretation of `fb%` = 60-90%; ~4 hours.
- **Cursor AI (Apr 24-30):** Edits to `wordle_gpt2_policy.py` (trie + char-mode sample loop), notebook plumbing, `fb%`/`trie_steps` verbose-log column wiring; ~5 hours.
- **ChatGPT (Apr 27):** Sanity check that `transformers>=4.51.0` introduced the `Qwen3` model class; ~10 min.

---

## Week 13 (April 20 - April 23, 2026)

### Overview
Bridge week between Week 12's Session 8 PASS-A result and Week 14's Qwen3 char-mode pivot. No headline experiment — the week was mostly (a) cleanup and persistence of the Session-8 mini-batch-CRN + restore-best plumbing, (b) cataloging the failure modes from the previous five weeks of Gemma + LoRA + warm-start runs, and (c) deciding what to swap next.

### Key Decisions

**1. End-of-Week-12 Decision: Pivot, Don't Re-Tune**
- **Decision:** Stop adding more iters / shrinking ALPHA / sweeping LoRA rank on the Gemma + word-softmax pipeline. The Session-8 PASS-A at `VOCAB_SCHEDULE=[16]` is the ceiling of what the Week-12 architecture seems to give us, and three weeks of effort to scale past one stage have produced negative results.
- **Rationale (the falsification log to date):** Week 11 ran the full 8-stage curriculum and got `es_gain ≈ 0` per stage. Week 12 attempts 1-5 isolated estimator/momentum/capacity hypotheses (PGPE-lite, EMA off, LoRA r=8) — all FAIL. Week 12 attempt 6 found `per_iter_secret_subset_size` as a working knob at vocab=16. Week 12 Session 8 confirmed `+22pp` PASS-A under quartered ALPHA. Generalizing past the single 16-secret stage was the next experiment we owe; we did not run it because the marginal cost (full A100 day per stage) was high enough that a parallel "swap the base LM" experiment in Week 14 became more attractive.

**2. Catalog of Candidates for the Week-14 Pivot**
- `Qwen/Qwen3-1.7B` — newer-than-Gemma 1.7B-param dense LM with `transformers>=4.51.0` chat-template support. Selected.
- `Qwen/Qwen3-1.7B-Instruct` — instruction-tuned variant. Deferred (not clearly better for Wordle-specific prompts in spot-checks).
- `meta-llama/Llama-3.2-1B-Instruct` — comparable-size Llama. Deferred (no clear advantage over Qwen3, and the Wordle-specific Qwen3 derivatives at Prime Intellect made the Qwen family the more strategic bet).
- `microsoft/Phi-4-mini` — too new at the time, peft + chat-template integration not yet exercised in the repo. Deferred.

**3. Target Module Selection for LoRA on Qwen3**
- **Decision:** `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]` (attention only, no MLP).
- **Rationale:** Gemma's working LoRA target set in Week 12 was attention-only at r=2; quadrupling rank to r=8 in Test A did not change the bottleneck. Adding MLP modules would have multiplied trainable parameters by ~3× without a hypothesis that says MLP-rank is the missing capacity. Stayed conservative; revisit after the Qwen3 + char-mode result.

**4. Char-Mode Action-Representation Plan**
- **Decision:** Build the `ACTION_GRANULARITY="char"` branch of `WordleGPT2Policy` first, then wire it into Week 14's notebook.
- **Rationale:** The single-softmax word head is the layer the Week-12 LoRA-rank sweep was implicitly trying to fix; replacing it with autoregressive 5-letter generation under a trie mask sidesteps the dimensionality discontinuity per curriculum stage.

### Results

No new experiments — ~3 hours of bookkeeping and ~6 hours of `WordleGPT2Policy` char-mode + trie-mask scaffolding (carried into Week 14).

### Failed Attempts

None worth listing — the week was deliberately not running new experiments, only cleaning up Week 12's Session 8 output and prepping the Week 14 notebook.

### Code Changes

- `src/wordle_gpt2_policy.py`: scaffolded `_WordleVocabTrie` and `action_granularity="char"` branches (no production use until Week 14).
- `src/es_wordle.py`: small comment cleanup; no behavior change.
- `notebooks/week12_wordle_es_lora_run.ipynb`: pinned the Session-8 `EXP2_RESTORE_BEST=1`, `EXP2_EVAL_STOCHASTIC_EVERY=1`, `EXP2_ALPHA_SCALE=0.25`, `EXP2_N_ITERATIONS=60` configuration as the documented "known-good" probe configuration in the cell-4 header.

### Open Questions

1. Would the Week-12 Session-8 PASS-A reproduce on Qwen3 with the *same* word-softmax head and `per_iter_secret_subset_size=4`? (We did not check; Week 14 changed the action representation simultaneously.)
2. Are there published Wordle-specific Qwen3 checkpoints we could use as a Week-16 fallback if Week 14 fails? (Yes — Prime Intellect's *Wordle Verifiers* release: `…-Wordle-SFT` and `…-Wordle-RL`. Cataloged for Week 16.)

### LLM Usage Log

- **Claude (Apr 20-23):** Pivot decision discussion, Qwen3 vs Llama vs Phi candidate evaluation, char-mode action-representation design; ~3 hours.
- **Cursor AI (Apr 21-23):** `_WordleVocabTrie` scaffolding + Week 12 notebook header pinning; ~2 hours.

---

## Week 12 (April 10 - April 19, 2026)

### Overview
Diagnosed why ES on top of supervised warm-start was contributing zero gain across the Gemma-3-1b + LoRA + 8-stage curriculum pipeline built in Week 11. Six attempts identified the actual bottleneck as the **per-secret revisit count per ES iteration** under common-random-numbers — a property of the eval budget, not of the ES estimator, momentum, or LoRA capacity.

### Key Decisions

**1. Phase A Diagnostic Probe — Add `cos(ĝ)` and `dprobe` Instrumentation (Apr 11)**
- **Decision:** Treat the existing per-stage warm-start + ES loop as a black box and run an isolated 20-iter ES-only probe on a fixed small secret pool with full diagnostic logging.
- **Diagnostics added to `src/es_wordle.py`:**
  - `cos(ĝ_t, ĝ_{t-1})` — cosine between successive raw ES gradients
  - `dprobe` — pre/post-step greedy-success delta on a fixed-seed probe slate
  - `‖θ − θ₀‖`, `Step‖`, `Grad‖`, `popσ`, `ess_rank`, `wins/N`
- **Probe pass criteria:** `median cos(ĝ) > 0.05` AND `eval_success lift ≥ 15pp`
- **First run verdict:** **FAIL** on both criteria — `cos(ĝ) = -0.000`, eval lift = +12pp.

**2. Phase C Attempt #1 — Baseline-Subtracted ES (PGPE-Lite) (Apr 12-15)**
- **Decision:** Add a `baseline_subtract=True` branch to `es_gradient_estimate_wordle` that uses raw mean-centered fitness without std-normalization.
- **Rationale:** When most population members tie at "lost everything", rank-fitness compresses the bulk to identical ranks and `std(ranks)` renormalizes away the magnitude of the sparse win signal that `win_fitness_scale=8` injects.
- **Implementation:** `BASELINE_SUBTRACT` hyperparameter in cell 4, threaded through both probe (cell 10) and production run (cell 12).
- **Result:** **Failed.** `cos(ĝ)` still `±0.00`, `Step‖` decayed 0.13 → 0.026, `dprobe` non-zero on 1/14 iters.

**3. Phase C Attempt #2 — Disable EMA Momentum (Apr 16)**
- **Decision:** Set `EMA_BETA = 0.9 → 0.0`.
- **Diagnosis:** With successive raw gradients uncorrelated (`cos(ĝ) ≈ 0`), EMA was a low-pass filter averaging signal toward zero. Per-iter raw gradient norm was stable (~4000), but post-EMA `Grad‖` decayed monotonically 3953 → 802 over 13 iters.
- **Result:** Mechanism healthier — `Grad‖` no longer decays, `dprobe` non-zero on ~half the eval iters with mixed signs (+9.4%, −9.4%, +9.4%). But fresh-secret `Success` still bouncing 0–12% — improvements were secret-specific, not transferable. **Verdict: still fail.**

**4. Test A — LoRA Capacity (Apr 17)**
- **Decision:** Bump `LORA_R = 2 → 8` to quadruple the trainable subspace.
- **Hypothesis:** If frozen Gemma's logit prior dominates, rank-2 perturbations may not have enough directional capacity to consistently flip argmax across secrets.
- **Result:** **Failed.** Same pattern as r=2 — capacity was not the bottleneck.

**5. Test B — Signal Density / Secret-Pool Size (Apr 18-19)**
- **Decision:** Reduce `PROBE_VOCAB = 16 → 4` while keeping `PROBE_N_EVAL = 16`. Each population member now plays each secret ~4 times under CRN per iter (vs. ~1× before).
- **Hypothesis:** The bottleneck is variance from secret sampling — per-member fitness is a 1-trial Bernoulli win-rate estimate dominated by which secret got drawn.
- **Result:** **SUCCESS.** Greedy success climbed 0% → **86% peak / 66% final** over 20 iters (vs. +12pp before). `dprobe` peaks +25.0%, +28.1%, +15.6%. Mean turns 6.0 → 2.5. `popσ` 0.13 → 0.7.
- **Footnote:** The probe verdict still printed `FAIL` because `cos(ĝ) ≈ 0`. In a 17M-parameter LoRA space, `cos(ĝ_t, ĝ_{t-1})` between two independent fitness-weighted noise vectors is the *null behavior*, not evidence of failure. **`cos(ĝ)` is a one-way indicator — positive proves signal, zero does not prove no-signal.**

**6. The Diagnosis**
The bottleneck across all six attempts was the **per-secret revisit count per ES iteration under CRN**. For the production curriculum:

| Stage | Secret pool | Episodes/iter | Visits per secret per iter |
|------:|------------:|--------------:|---------------------------:|
| 1     | 16          | 16            | **1×**                     |
| 2     | 32          | 16            | **0.5×**                   |
| ...   | ...         | 16            | progressively worse        |
| 8     | 1024        | 16            | **0.016×**                 |

Every stage of the production curriculum was structurally in the "winners are lottery-ticket-secrets" regime. Warm-start was carrying all the gain, ES was a random walk on top.

### Failed Attempts

**1. Trusting `cos(ĝ)` as a two-way indicator (Apr 11-19)**
- Spent ~6 hours of compute on three estimator/momentum/capacity variants because the verdict logic treated `cos(ĝ) ≈ 0` as evidence of failure rather than as the high-dim ES null behavior.
- **Lesson:** The verdict logic will be revised to `eval_lift OR cos(ĝ)` before next week.

**2. PGPE-lite + EMA-on (Apr 12-15)**
- Phase C attempt #1 was implemented with the existing `EMA_BETA=0.9` still in place. The two interacted destructively: baseline-subtract gave the per-iter raw gradient stable magnitude, then EMA averaged it to zero anyway.
- Should have done EMA-off and baseline-subtract together as one experiment, not in series.

**3. LoRA r=8 capacity bump (Apr 17)**
- Quadrupled trainable parameters with no measurable effect because the actual bottleneck (eval budget per secret) was upstream of representational capacity. ~2 hours of compute, zero information gain on the bottleneck. Did rule out capacity as the bottleneck, which is small consolation.

### Code Changes
- `src/es_wordle.py`:
  - Added `baseline_subtract: bool = False` to both `es_gradient_estimate_wordle` and `train_es_wordle` signatures and docstrings
  - Implemented baseline-subtract branch with antithetic-pair difference (when `antithetic=True`) and raw mean-centered fitness (when `antithetic=False`)
- `notebooks/week12_wordle_es_lora_run.ipynb`:
  - Cell 4: added `BASELINE_SUBTRACT = True`, set `EMA_BETA = 0.0`, set `LORA_R = 8`, with paragraph-long comments explaining each diagnosis
  - Cell 9 markdown: rewrote probe description to document the signal-density test (Test B)
  - Cell 10 (probe): added `PROBE_VOCAB = 4`, threaded `baseline_subtract` into the probe ES call, added one-shot gradient-norm probe to auto-calibrate `PROBE_ALPHA` for whichever fitness shaping is active
  - Cell 11 markdown: documented the EMA-cancels-uncorrelated-gradients failure mode
  - Cell 12 (production run): threaded `baseline_subtract` into ALPHA-cal probe and `train_curriculum`
  - Cell 2: added `baseline_subtract` to `inspect.signature(train_es_wordle)` check so a stale `es_wordle.py` import is caught at notebook startup
- `scripts/run_es_signal_density_probe.py` (new): standalone CLI mirror of the probe cell so the signal-density test can be re-run without restarting the Jupyter kernel

### Open Questions
1. Does the full pipeline (warm-start + ES + diagnostics) at `VOCAB_SCHEDULE=[4]`, `N_ITERATIONS=30` produce measurable `es_gain > 0` on top of warm-start? (Next experiment.)
2. If yes: does mini-batch ES (e.g. 8 secrets × 32 episodes per iter, sample subset under CRN) preserve `es_gain` at the production vocab=1024 stage?
3. If no: is the LM logit prior so dominant that no head-only / rank-8 LoRA configuration can move argmax consistently across many secrets at once? Would per-turn dense shaping (yellow/green letter coverage, constraint consistency) help?

### LLM Usage Log
- **Claude (Apr 10-19):** All six debugging sessions — interpreting probe output, proposing variants, evaluating results, and the high-dim `cos(ĝ)` math; ~12 conversations, ~6 hours total.
- **Cursor AI (Apr 12-19):** Edits to `src/es_wordle.py` (`baseline_subtract` branch), notebook cells (4, 9, 10, 11, 12), and the standalone CLI probe script; ~3 hours of small high-precision edits.
- **ChatGPT (Apr 16):** Sanity check that EMA averages successive uncorrelated gradient vectors toward zero in expectation; ~20 min.

See `docs/llm_exploration/week12_log.md` for the conversation-level account.

---

## Week 11 (April 3 - April 9, 2026)

### Overview
Scaled the Wordle ES pipeline from distilGPT-2 + 16-word vocab to **Gemma-3-1b-it + LoRA + 1024-word vocab** with a multi-stage curriculum. End-to-end pipeline runs cleanly; central scientific question (does ES contribute on top of warm-start?) deferred to Week 12.

### Key Decisions

**1. Backbone Swap: distilGPT-2 → Gemma-3-1b-it**
- Switched to `google/gemma-3-1b-it` (instruction-tuned) so the policy could be prompted with the model's native chat template (`apply_chat_template`).
- Frozen base model; only LoRA + linear head are trainable.

**2. LoRA Wiring on a HuggingFace Causal LM**
- Used `peft.LoraConfig(r=2, target_modules=["q_proj","k_proj","v_proj","o_proj"])` on Gemma's attention projections.
- Verified `count_lora_parameters()` matches expectations (~17M total trainable LoRA params at r=8, ~4M at r=2) and that base weights stay frozen across an ES iteration.

**3. Curriculum Design (`VOCAB_SCHEDULE`)**
- 8 stages: `[16, 32, 64, 96, 128, 256, 512, 1024]`.
- Each stage runs supervised warm-start (200–1600 episodes, scaling with vocab size) followed by ES (10–30 iters depending on stage).
- Per-stage `quick_eval_success` before/after warm-start AND before/after ES, so per-stage `ws_gain` and `es_gain` are recorded separately.

**4. Variance Reduction Suite**
- `antithetic=True` — sample N/2 noise vectors and evaluate both `+ε` and `−ε`.
- `common_random_numbers=True` — same secret draws and same env-RNG seed for every population member within an iteration.
- `rank_fitness=True` — centered ranks instead of raw fitness.
- `EMA_BETA=0.9` — Adam-like momentum on the raw gradient.
- `win_fitness_scale=8.0` — bonus weight on win events in `win_plus_return` fitness so a single win in a population dominated by losses is visible.

### Results

**Pipeline mechanics:**
- 8-stage curriculum runs end-to-end on a single A100 in ~5 hours.
- Per-stage warm-start gains: 38–66pp greedy success on the training secret pool.
- Per-stage ES gains: **~0pp.**

**Honest assessment:**
The pipeline is correct, instrumented, and reproducible. The scientific question (does ES contribute beyond warm-start?) is unresolved — every stage's `es_gain ≈ 0` could mean (a) ES doesn't help, (b) ES would help but the eval budget is too small to detect it, or (c) the warm-start is saturating the head and there's no headroom for ES. Week 12 isolates which.

### Code Changes
- `src/wordle_gpt2_policy.py`: extended to load arbitrary HuggingFace causal LMs (`AutoModelForCausalLM`), apply chat templates, and batch-forward via `forward_logits_batch`
- `src/wordle_gpt2_warmstart.py`: added per-stage warm-start with `feedback_consistent_random=True` random pre-play and a post-WS success ceiling (skip WS if pool already at ≥0.85 greedy success)
- `src/es_wordle.py`: added `train_curriculum` driver with per-stage warm-start + ES + diagnostics
- `notebooks/week12_wordle_es_lora_run.ipynb`: full pipeline notebook (named "week12" by file path; structurally the Week-11 deliverable)

### Open Questions
1. Is `es_gain ≈ 0` an artifact of the eval-budget design or a real "ES adds nothing" result?
2. Does `cos(ĝ)` ever turn positive within an ES stage? (Need the Phase A probe to find out.)

### LLM Usage Log
- **Cursor AI:** LoRA wiring on Gemma, curriculum driver, per-stage diagnostics; ~10 hours
- **Claude:** Per-stage diagnostic design (`ws_gain` vs `es_gain` separation), warm-start headroom ceiling rationale; ~3 hours

---

## Week 10 (March 23 - April 2, 2026)

### Overview
Graduated from GridWorld to **Wordle** as the next testbed for ES. Built a Gym-style Wordle environment, a linear-head policy on top of a frozen language model, and ran the first end-to-end ES + LM training loop on a 16-word vocabulary with distilGPT-2.

### Key Decisions

**1. Domain Selection: Wordle**
- **Decision:** Pick Wordle as the bridge from GridWorld to LM-based RL.
- **Rationale:** Sparse reward (only know if you won at the end), discrete action space (a vocabulary of valid 5-letter words), reward shaping is natural (per-turn green/yellow letter coverage), and the action space is small enough to handle as a categorical head rather than open-ended generation.
- **Alternatives considered:** Math-token games (too dense), single-step QA (no multi-turn feedback structure), TextWorld (too unconstrained).

**2. Environment Design**
- `WordleEnvironmentWrapper` in `src/wordle_env.py`:
  - Action format: `<guess>WORD</guess>` with optional `<think>...</think>` block ignored at scoring time
  - Per-turn structured feedback (green/yellow/grey per letter)
  - `set_target_pool(words)` so the curriculum can swap secret distribution without rebuilding
  - Decoupled action vocabulary (what the policy can emit) from secret pool (what the env can pick) — this seam paid off heavily in Weeks 11/12
- Initial pool: `MOCK_WORDLE_TARGETS` (16 words) for smoke test; hook to load full Prime Intellect 2300-word answer list later

**3. Policy Architecture: Linear Head Over Frozen LM**
- `WordleGPT2Policy` in `src/wordle_gpt2_policy.py`:
  - Loads any HuggingFace causal LM (`AutoModel.from_pretrained`)
  - Linear `head: nn.Linear(hidden_dim, vocab_size)` over a fixed superset action vocabulary
  - **Previous-guess masking**: logits for already-guessed words set to `−∞` so the policy can't repeat itself
  - `richer_prompt=True`: prepends a per-turn structured constraint summary ("letters known to be in word: A, R; positions known: _R___") so the LM has the deductive state
  - `forward_logits_batch(states)` for batched forward across an ES population (critical for wall-clock when scaling to Gemma)
- Action space is a **fixed superset** of all curriculum stages — head dim never changes, secret pool is what gets restricted

**4. ES on Wordle: Carry the Week-7 Lessons Forward**
- Added to `src/es_wordle.py`:
  - `rank_fitness=True` (centered ranks robust to fitness ties)
  - `fitness_objective="win_plus_return"` (mix sparse win-rate with dense partial-credit return)
  - `win_fitness_scale=8.0` (bonus weight on win events)
  - `antithetic=True` and `common_random_numbers=True` (variance reduction)
- These four flags turned the initial "0% forever" run into a noisy "bouncing 0–12% on 16-word pool" run — clearly random, but no longer flat.

**5. Calibrate `Step‖`, Not `α`**
- Instrumented `param_drift`, `step_norm`, `grad_norm` per iter.
- Added an in-cell ALPHA-calibration probe: run one ES gradient estimate at init, measure `‖ĝ‖`, back-solve for `α` to hit a target initial step `~0.13` (LoRA-friendly Adam-equivalent magnitude).
- This abstraction made the `(σ, N, n_params)` knobs independently tunable without re-tuning LR.

**6. Supervised Warm-Start**
- `supervised_warm_start_wordle` in `src/wordle_gpt2_warmstart.py`:
  - Pick a secret, simulate 1–4 random opening guesses, record the resulting state
  - Cross-entropy loss between head logits and the secret's vocab index
  - **Critical constraint:** never put the secret in the prompt — must be inferred from feedback
  - `feedback_consistent_random=True`: random pre-play words constrained by accumulated feedback (so the supervised target is a sensible inference, not "predict random secret given garbage prefix")

### Failed Attempts

**1. Open-Vocabulary Generation**
- **Attempt:** Let the LM `generate(...)` a 5-letter completion as the action.
- **Result:** ~30% of guesses were not in the answer list, env rejected them, rollouts ended in 1–2 turns with no signal.
- **Fix:** Switched to a categorical head over a fixed vocabulary the same day.

**2. No Previous-Guess Masking**
- **Attempt:** First ES run let the policy pick the same word every turn.
- **Result:** Argmax stuck on first guess, every game lost in 6 identical turns.
- **Fix:** Add `−∞` masking on previously-guessed action indices. 5 lines of code.

**3. σ = 0.1 (carryover from GridWorld)**
- **Attempt:** Re-used Week-7 default noise scale.
- **Result:** Head exploded after 5 iters — `‖θ − θ₀‖ → 30+`, success collapsed to 0%.
- **Fix:** Reverted to `σ = 0.02`. Cost ~3 hours of compute.

**4. Standardization-Collapse on 16-Word Vocab**
- **Attempt:** First ES run with vanilla `(fitness − mean) / std` shaping.
- **Result:** Same Week-7 failure mode — sparse wins → near-zero population variance → near-zero ES update. 80 iterations, 0% success, every population member tied on fitness.
- **Fix:** Switched to `rank_fitness=True` + `win_plus_return` + `win_fitness_scale=8`. (Foundation for Week 11/12.)

### Initial Results

**Pipeline (16-word vocab, distilGPT-2, no LoRA, no curriculum):**
- Random init: 0% greedy success
- After 50 warm-start steps: ~60% greedy success on training pool
- After 30 ES iterations on top: ~62% greedy success
- **ES contribution: roughly noise.** This is the question that defined Weeks 11 and 12.

### Code Changes
- `src/wordle_env.py` (new): Gym-style Wordle env with structured per-turn feedback and `set_target_pool` API
- `src/wordle_gpt2_policy.py` (new): linear-head policy over a frozen HuggingFace LM with previous-guess masking and batched forward
- `src/wordle_gpt2_warmstart.py` (new): supervised warm-start with feedback-consistent random pre-play
- `src/es_wordle.py` (new): ES gradient estimator + training loop adapted for Wordle (rank-fitness, antithetic, CRN, win-plus-return shaping)
- `notebooks/week10_wordle_es_distilgpt2.ipynb` (new): smoke-test notebook for the 16-word vocab + distilGPT-2 pipeline

### Open Questions Heading Into Week 11
1. Does warm-start memorize the training secret pool, or generalize?
2. With Gemma-3-1b instead of distilGPT-2, does the LM prior absorb so much of the policy that LoRA + ES updates can't move argmax?
3. What does a multi-stage curriculum (vocab=16 → 32 → 64 → … → 1024) actually buy us, vs. one big stage at vocab=1024?

### LLM Usage Log
- **ChatGPT:** Domain selection (Wordle vs alternatives), diagnosis of "0% forever" failure mode, ~2 hours
- **Claude:** Policy class design (linear head over LM hidden state), warm-start no-leak design, ~1 hour
- **Cursor AI:** Wordle env, ES-on-Wordle module, hyperparameter sweep instrumentation, ~4 hours of pair programming

See `docs/llm_exploration/week10_log.md` for the conversation-level account.

---

## Week 7 (Feb 17 - Feb 24, 2026)

### Overview
Implemented rank-1 LoRA for GridWorld ES in LoRA-only mode (base weights frozen) and added a Week 7 notebook comparing standard ES vs LoRA-ES across Gaussian, Cauchy, and Laplace perturbation noise.

### Key Decisions

**1. LoRA Scope and Optimization Mode (Feb 24)**
- Scoped first implementation to **GridWorld only** to keep risk low and validate correctness before extending to Wordle
- Chose **LoRA-only updates** (`param_mode='lora'`) rather than updating LoRA + base weights
- Rationale: preserves PEFT behavior, reduces ES search dimensionality, and isolates whether rank-1 adapters are sufficient

**2. Rank-1 Adapter Design**
- Added `LoRALinearRank1` to `src/model.py` with:
  - Frozen base projection (`nn.Linear`)
  - Rank-1 delta: `alpha * (x @ a) * b`
- Added `PolicyNetwork` options: `use_lora`, `lora_alpha`, `lora_init_scale`
- Added helper methods: `lora_parameters()`, `base_parameters()`, `count_lora_parameters()`, and `freeze_base_parameters()`

**3. ES Parameter Subset Selection**
- Extended `src/utils.py` with `param_mode` support in:
  - `es_gradient_estimate(...)`
  - `train_es(...)`
- Added utilities to flatten/set selected parameter subsets (`all` vs `lora`) while keeping backward-compatible defaults
- Ensured perturbation apply/restore uses the same selected parameter set, preventing accidental base-weight mutation

### Results

**Validation and Testing**
- `pytest tests/test_basic.py -q`: **22 passed**
- Added LoRA-focused tests to `tests/test_basic.py`:
  - LoRA forward pass + freeze behavior
  - LoRA-only ES gradient shape
  - LoRA-only update changes adapters while base params remain unchanged

### Notebook Deliverable
- Added `notebooks/week07_gridworld_lora_perturbation.ipynb` following prior-week structure
- Focus: **GridWorld only**, comparing:
  - Standard ES (`param_mode='all'`)
  - Rank-1 LoRA ES (`param_mode='lora'`)
- Runs all perturbation distributions in `NOISE_TYPES`: Gaussian, Cauchy, Laplace
- Includes:
  - problem statement and math section
  - parameter-efficiency comparison
  - side-by-side metrics table (reward/success/grad norm/runtime)
  - plots for final metrics and training curves by noise type

### Code Changes
- `src/model.py`:
  - Added `LoRALinearRank1`
  - Extended `PolicyNetwork` with LoRA configuration and parameter helper methods
- `src/utils.py`:
  - Added `PARAM_MODES`
  - Added `param_mode` to ES APIs
  - Added selected-parameter flatten/set utilities
- `tests/test_basic.py`:
  - Added LoRA behavior and LoRA-only ES regression tests
- `notebooks/week07_gridworld_lora_perturbation.ipynb`:
  - New Week 7 experiment notebook

### Open Questions
1. Should LoRA ES use the same `(sigma, alpha)` as standard ES, or should each method/noise type be tuned separately?
2. Does rank-1 LoRA remain competitive on `HarderGridWorld` or larger state spaces?
3. For heavy-tailed noise (Cauchy), does LoRA-only search space improve stability enough to recover performance with smaller `sigma`?

### LLM Usage Log
**Cursor (Feb 24):**
- Implemented rank-1 LoRA wrappers and LoRA-only ES parameter selection
- Added regression tests and created Week 7 notebook comparison across all noise types

**ChatGPT (Feb 25):**
- Diagnosed the “0 progress” failure mode as **sparse rewards → low fitness variance → fitness standardization → near-zero ES update**
- Recommended training-time reward shaping while keeping sparse evaluation for fair reporting

**Cursor (Feb 25):**
- Updated Week 7 notebook to train on Week 4-style shaped rewards and match Week 4 ES sampling budget

### Updates (Feb 25)
- Added Week 4-style distance-based reward shaping to `notebooks/week07_gridworld_lora_perturbation.ipynb` for **training** (`ShapedRewardEnvComparison`), while keeping **sparse** `GridWorld` for evaluation metrics
- Matched Week 7 ES sampling budget to Week 4 defaults: **80 iterations, N=50, 5 episodes/perturbation** (kept `sigma=0.10`, `alpha=0.05`, `max_steps=50`)
- Added `docs/llm_exploration/week7_log.md` to document Week 7 AI-assisted development

### Updates (Feb 26)
- Refactored Week 7 notebook from single-run LoRA trials into a **paired adaptation protocol**:
  - Pretrain source policy on baseline GridWorld
  - Perturb target environment
  - Continue from the same source checkpoint with both `param_mode='all'` and `param_mode='lora'`
- Extended `src/utils.py` ES logging with adaptation diagnostics:
  - `eval_steps`
  - `fitness_std`
  - `env_interactions`
  - `cumulative_env_interactions`
- Added adaptation metrics and analysis outputs in `notebooks/week07_gridworld_lora_perturbation.ipynb`:
  - time-to-threshold (`0.6/0.8/0.9`)
  - interactions-to-threshold
  - success AUC
  - final sparse-eval reward/success/steps
  - CI-based adaptation curves and diagnostic panels
- Added **CSV exports** for report-ready tables:
  - `results/es_lora_adaptation_runs.csv`
  - `results/es_lora_adaptation_summary.csv`
  - `results/es_lora_adaptation_deltas.csv`
- Added verbose run-status printing (seed, method, perturbation level, ETA, final metrics) for long CPU experiments.
- Implemented dynamic pretraining safeguards in notebook utilities:
  - optional early-stop chunking with max-iteration cap
  - source-quality gate (`pretrain_success` threshold) before adaptation
  - explicit skip handling for failed source runs (with `skipped_reason`)
- Replaced transition-noise perturbation with **curriculum-style layout perturbation**:
  - local obstacle-cell moves controlled by perturbation level
  - optional slight goal relocation at higher perturbation levels
  - updated labels/docs to use layout move fraction instead of transition noise std.

---

## Week 5 (Feb 9 - Feb 16, 2026)

### Overview
Added Cauchy and Laplace perturbation noise to ES. Required implementing distribution-specific score functions for correct gradient estimation.

### Key Decisions

**1. Multi-Distribution Noise (Feb 16)**
- Added `sample_perturbation()` and `score_function()` in `src/utils.py`; threaded `noise_type` through `es_step`, `es_gradient_estimate`, and `train_es`
- Key insight: the ES gradient weight must use each distribution's score function (−∇ log p(ε)), not raw ε. Gaussian: ε, Cauchy: 2ε/(1+ε²), Laplace: sign(ε)
- Without this fix, Cauchy and Laplace both gave 0% success

### Results

**8×8 grid, 8 obstacles, 80 iterations, N=50, σ=0.1:**

| Noise Type | Success Rate | Time    |
|------------|-------------|---------|
| Gaussian   | 100%        | 318.4s  |
| Cauchy     | 0%          | 291.3s  |
| Laplace    | 100%        | 360.5s  |
| PPO        | 100%        | 14.7s   |

- Gaussian and Laplace both solve the task; Laplace converges slightly slower with more variance
- Cauchy never learns — train reward stays negative for all 80 iterations
- **Why Cauchy fails here:** This gridworld requires precise parameter tuning. Cauchy's extreme perturbations (regularly 10–1000x scale) randomize the policy, destroying any useful fitness signal. The environment is too "precise" for such heavy tails.
- **Future value:** Cauchy may still be useful in environments with many local optima or smoother reward landscapes where large jumps help escape basins. A σ sweep per distribution (Cauchy likely needs much smaller σ) and hybrid strategies (Cauchy early, Gaussian late) are natural next steps.

### Code Changes
- New in `src/utils.py`: `sample_perturbation()`, `score_function()`, `NOISE_TYPES`
- Modified: `es_gradient_estimate()`, `train_es()` accept `noise_type`
- Updated `src/__init__.py` exports; notebook loops over all noise types

### LLM Usage Log
**Cursor (Feb 16):** Implemented noise distributions, diagnosed score function mismatch, debugged `__pycache__` staleness

---

## Week 4 (Jan 27 - Feb 6, 2026)

### Overview
Implemented Evolution Strategies (ES) and Proximal Policy Optimization (PPO) for parameter-space optimization in sparse reward reinforcement learning. Built a GridWorld environment, ES training pipeline, and full PPO training pipeline from scratch in `src/`. Both methods run end-to-end; PPO validated on 5×5 grid, with early comparison results on 8×8.

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
  - Includes `get_action_batch` for PPO batch evaluation (log probs + entropy)
- **ValueNetwork:** Same architecture, maps state → scalar value estimate
  - Used by PPO for advantage estimation via GAE

**6. PPO Implementation (Feb 5-6)**
- **Decision:** Implement full PPO training pipeline in `src/ppo_training.py` as a gradient-based baseline for comparison with ES
- **Rationale:** PPO is the standard policy gradient baseline; comparing ES vs PPO on the same sparse reward gridworld gives insight into when gradient-free methods are competitive
- **Key design choices:**
  - **Separate optimizers:** Policy (Adam, lr=3e-4) and value network (Adam, lr=1e-3) use independent optimizers with different learning rates. Value network can learn faster since it has a stable regression target; policy updates need to be conservative.
  - **GAE (γ=0.99, λ=0.95):** Generalized Advantage Estimation balances bias-variance in advantage estimates. Implemented as standalone `compute_gae()` function with reverse TD-error accumulation.
  - **Clipped surrogate (ε=0.2):** Standard PPO clip to prevent destructive policy updates
  - **Entropy bonus (coef=0.01):** Encourages exploration, critical for sparse reward environments
  - **Gradient clipping (max_norm=0.5):** Prevents gradient explosions during minibatch updates
  - **Rollout buffer:** `RolloutBuffer` class collects (state, action, reward, log_prob, value, done) tuples per iteration, converts to tensors in batch
  - **Minibatch updates:** 4 epochs of shuffled minibatches (size 64) per iteration over 128-step rollouts
- **Alternatives considered:** A2C (simpler but less stable), SAC (overkill for discrete actions)

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

**5. PPO Combined Optimizer (Feb 5)**
- **Attempt:** Used a single Adam optimizer for both policy and value networks with a combined loss (`policy_loss + 0.5 * value_loss + entropy_bonus`)
- **Result:** Unstable training — value loss gradients were much larger than policy loss gradients, causing the policy to barely update while the value network overfit
- **Lesson:** Policy and value networks have very different loss scales and learning dynamics; a shared optimizer cannot balance both well
- **Fix:** Switched to separate optimizers with different learning rates (3e-4 for policy, 1e-3 for value)

**6. PPO Sparse Reward Stalling (Feb 6)**
- **Attempt:** Ran PPO on 8×8 grid with default `entropy_coef=0.01`
- **Result:** Success rate stayed at 0% for first ~50 iterations (~6,400 env steps). PPO needs to randomly stumble into the goal to get any gradient signal.
- **Lesson:** Entropy coefficient is a sensitive hyperparameter for sparse rewards — too low and the policy collapses to a deterministic (bad) action before finding the goal; too high and it never converges
- **Fix:** Verified correctness on 5×5 grid first (converged to ~80% success in 100 iterations), confirming the algorithm works. For 8×8, PPO eventually starts learning around iteration 100.

**7. PPO Old Log Prob Gradient Leak (Feb 5)**
- **Attempt:** Stored `log_prob` tensors directly in the rollout buffer (with gradient graph attached)
- **Result:** Backward pass tried to backpropagate through the old policy parameters, causing incorrect gradients and memory accumulation
- **Lesson:** Old log probabilities collected during rollout must be detached from the computation graph — they are constants in the PPO ratio calculation
- **Fix:** Store `log_prob` as a plain float in the buffer, convert to tensor only during the update phase

### Testing Strategy

**Unit Tests (in notebook):**
1. Environment mechanics (collisions, rewards, termination)
2. Policy network forward pass (shape, softmax validity)
3. ES gradient shape and validity
4. `RolloutBuffer` add/get/clear cycle — verify tensor shapes and types
5. `compute_gae` output shapes and value sanity (advantages sum ≈ 0 after normalization)

**Integration Tests:**
1. Full ES training loop runs without errors
2. Full PPO training loop runs end-to-end (`train_ppo` with small `n_iterations=5`)
3. Evaluation pipeline produces metrics correctly for both ES-trained and PPO-trained policies
4. `evaluate_policy` deterministic mode produces consistent results across calls with same seed

**Edge Case Tests (in notebook):**
1. Empty grid (no obstacles) — tested policy generalization
2. Dense obstacles (15 on 8×8) — tested robustness
3. Larger grid (12×12) — tested scalability of state representation

**PPO-Specific Validation:**
1. Verified PPO converges on 5×5 grid (~80% success in 100 iterations) as a correctness check
2. Confirmed clipped ratio stays within [1-ε, 1+ε] range during early training
3. Checked that entropy decreases over training (policy becomes more confident)
4. Verified separate optimizers produce stable training (vs. failed combined optimizer attempt)

**Note:** Environment mechanics test revealed a bug in the action-to-goal navigation test (assertion error on goal detection), which needs to be investigated further.

### Initial Results

**ES — Quick validation run (20 ES iterations on 8×8 grid, 8 obstacles):**
- Success rate: 0% — policy did not converge in 20 iterations
- Gradient norms remained high (~400) throughout, suggesting the policy is still in early exploration

**ES — Resource measurements (single ES iteration):**
- Time: ~0.65 seconds per iteration
- Memory: ~127 MB
- Model size: 8,580 parameters (0.033 MB)
- Estimated full 100-iteration training: ~1.1 minutes on CPU

**PPO — 5×5 grid correctness check (100 iterations):**
- Success rate: ~80% after 100 iterations — confirms the PPO pipeline works correctly
- Policy converges to near-optimal paths
- Entropy decreases steadily, indicating the policy is becoming more confident

**PPO — 8×8 grid (200 iterations, 128 steps/iter):**
- Success rate: 0% for first ~50 iterations (sparse reward makes initial exploration slow)
- Begins learning around iteration 100 as random exploration occasionally reaches the goal
- EMA reward shows a clear upward trend after the initial plateau
- Slower to start than ES on the same grid, but catches up around iteration 100

**PPO vs. ES — Early comparison notes:**
- ES finds non-zero rewards faster (doesn't require backprop through a reward signal)
- PPO is more sample-efficient once it starts learning (gets more out of each environment interaction)
- Both struggle with 8×8 sparse reward in short training runs; longer runs needed for definitive comparison

**Honest assessment:** Both ES and PPO implementations run end-to-end. PPO has been validated on 5×5 grid and shows signs of learning on 8×8. ES has not yet demonstrated convergence within 20 iterations on 8×8 — longer training runs and hyperparameter tuning are needed for both methods. A rigorous multi-seed comparison is the priority for next week.

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

4. **How does PPO compare to ES at scale?**
   - PPO is now implemented in `src/ppo_training.py` and validated on 5×5 grid
   - Early observations: PPO starts slower on 8×8 sparse reward (needs random goal discovery) but learns more efficiently once signal is found
   - Need: Rigorous multi-seed comparison (5-10 seeds) on same environments with matched compute budgets
   - Open: Is ES's advantage in sparse settings maintained as grid size increases, or does PPO's sample efficiency dominate?

5. **What is the optimal entropy coefficient for PPO on sparse rewards?**
   - Current setting (0.01) works on 5×5 but causes slow starts on 8×8
   - Need: Sweep over entropy_coef in [0.005, 0.01, 0.02, 0.05] on 8×8 grid
   - Related: Should entropy coefficient be annealed over training?

### Resource Usage

**Computational:**
- Single ES iteration: ~0.65 seconds (CPU only)
- Single PPO iteration (128 steps + 4 epochs of updates): ~0.3 seconds (CPU only)
- Memory: ~127 MB peak (ES), ~140 MB peak (PPO — additional buffer and value network)
- No GPU required for small networks
- PPO 200-iteration training: ~1 minute on CPU

**Development Time:**
- Environment (`GridWorld`, `HarderGridWorld`): 4 hours
- Network architecture (`PolicyNetwork`, `ValueNetwork`): 3 hours
- ES implementation (`es_gradient_estimate`, `train_es`): 6 hours
- PPO implementation (`RolloutBuffer`, `compute_gae`, `train_ppo`, `evaluate_policy`): 5 hours
- PPO debugging (combined optimizer issue, log prob detaching, entropy tuning): 2 hours
- Utilities (evaluation, plotting, statistics): 3 hours
g- Notebook validation & testing: 4 hours
- Documentation: 3 hours
- **Total: ~30 hours**

### Code Organization Evolution

**Initial (Jan 27):**
- Single file exploring ideas

**Iteration 1 (Jan 30):**
- Split into separate environment and training files

**Final (Feb 5-6):**
- `src/model.py` — Environments (`GridWorld`, `HarderGridWorld`) + Networks (`PolicyNetwork`, `ValueNetwork`)
- `src/utils.py` — ES Training (`train_es`, `es_gradient_estimate`) + Evaluation (`evaluate_policy`, `plot_training_curves`, `compute_statistics`)
- `src/ppo_training.py` — PPO pipeline (`RolloutBuffer`, `compute_gae`, `evaluate_policy`, `train_ppo`)
- `src/__init__.py` — Clean package exports
- `notebooks/week04_gridworld_es_vs_ppo.ipynb` — End-to-end validation

### LLM Usage Log

**ChatGPT:**
- Helped debug ES gradient estimation (Jan 30)
- Suggested reward standardization approach (Jan 31)
- Advised on PPO evaluation strategy: EMA tracking, deterministic eval, multiple metrics (Feb 6)
- Helped diagnose PPO sparse reward stalling on 8×8 grid — suggested verifying on smaller grid first (Feb 6)

**Cursor:**
- Used for refactoring code into `src/` structure (Feb 5)
- Code completion for boilerplate (throughout)
- Helped structure `ppo_training.py` module and write docstrings (Feb 5-6)

**Claude:**
- Asked about theoretical justification for ES in sparse reward settings (Feb 3)
- Got references to relevant papers (Salimans et al., Mania et al.)
- Guided PPO training loop design: RolloutBuffer, GAE implementation, separate optimizers rationale (Feb 5)
- Explained clipped surrogate objective and entropy bonus mechanics (Feb 5)

### Next Week Plan

**Immediate (Week 5):**
1. Debug environment mechanics test failure
2. Run full 100+ iteration ES training and verify convergence
3. Run rigorous ES vs. PPO comparison: 5-10 seeds, matched compute budgets, same environments
4. Tune hyperparameters — ES: grid search over σ, α, N; PPO: sweep entropy_coef, learning rates
5. Generate comparison plots (training curves, success rate vs. iterations, sample efficiency)

**Technical Improvements:**
1. Parallel ES evaluation (multiprocessing)
2. Mirrored sampling for variance reduction
3. Adaptive sigma scheduling
4. PPO entropy coefficient annealing (linear decay over training)
5. Learning rate scheduling for PPO (cosine or linear warmup)

**Exploration:**
1. Test both ES and PPO on `HarderGridWorld` (key-door task)
2. Test on larger grids (12×12, 16×16) — how do ES and PPO scale differently?
3. Try Natural ES or CMA-ES variants
4. Ablation study: with/without fitness standardization (ES), with/without GAE (PPO)
5. Investigate reward shaping (small step penalty) to help PPO overcome sparse reward plateau

### Lessons Learned

1. **Start simple:** 8×8 grid was the right scope for Week 4. Complex environments can come later once the base pipeline works.

2. **Test incrementally:** Running a quick 20-iteration validation caught issues early, even though it wasn't long enough for convergence. For PPO, validating on 5×5 before 8×8 confirmed correctness before scaling up.

3. **Hyperparameters matter:** Spent time finding workable σ and α values for ES through trial and error. PPO's entropy coefficient proved equally sensitive for sparse rewards. Should formalize both with sweeps.

4. **Separate concerns:** Using separate optimizers for policy and value networks was a crucial design lesson. A combined optimizer seemed simpler but led to unstable training — different loss scales require independent learning rates.

5. **Gradient-free vs. gradient-based trade-offs are real:** ES starts learning sooner on sparse rewards (no gradient signal needed), but PPO is more sample-efficient once it finds the reward. Neither dominates — the comparison is genuinely interesting.

6. **Documentation while coding:** Writing docstrings as we coded saved time when assembling the notebook and log.

7. **Be honest about results:** Initial results show limited convergence for both methods in short runs — this is valuable information for guiding next steps rather than something to hide.

### References Used

1. [Salimans et al., 2017] - Evolution Strategies as a Scalable Alternative to Reinforcement Learning
2. [Mania et al., 2018] - Simple random search provides a competitive approach to reinforcement learning
3. [Schulman et al., 2017] - Proximal Policy Optimization Algorithms (used for PPO implementation)
4. [Schulman et al., 2016] - High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE reference for `compute_gae`)

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

*Log updated: April 19, 2026*
