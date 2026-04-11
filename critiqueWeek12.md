# Week 12 Self-Critique

## ORIENT

### Strengths

- **Run profiles directly address Week 10 workflow pain.** The **`RUN_PROFILE`** switch (`smoke` vs `gemma_full`) separates “does the notebook run end-to-end?” from “are we doing a serious experiment?” without maintaining two separate notebooks. This is a practical response to the Week 10 critique that full LM + ES is slow and hard to iterate on.

- **Denser eval logging in smoke mode.** With **`EVAL_EVERY=1`** and a short **`N_ITERATIONS`**, the eval curves actually have enough points to **see shape** during debugging. That partially fixes the Week 10 issue where **`EVAL_EVERY=10`** and **`N_ITERATIONS=10`** produced only **two** eval checkpoints—too sparse to interpret learning dynamics.

- **Hardware-aware loading is a real upgrade.** `choose_device()` (CUDA / MPS / CPU) and **`default_model_load_kwargs`** for CUDA dtypes reduce foot-guns when moving off CPU. Week 10 assumed a single `DEVICE` string pattern; Week 12 is closer to how HF models are run in practice.

- **Gemma IT path is structurally supported.** `USE_CHAT_TEMPLATE` and **`CHAT_GENERATION_PROMPT`** wired into `WordleGPT2Policy` match how instruction-tuned models expect prompts, which matters for **`google/gemma-3-1b-it`** in **`gemma_full`**. That is a meaningful extension beyond “always format like GPT-2.”

- **Checkpoint isolation by profile.** Saving to **`models/wordle_gpt2_es_head.<RUN_PROFILE>.pt`** prevents a smoke run from clobbering a long Gemma checkpoint—a small change that avoids silent data loss.

- **Operational robustness.** Default **`HF_HUB_DISABLE_XET=1`** documents and mitigates a class of hub download failures; the notebook explicitly requires **`transformers>=4.50.0`**, which is appropriate for newer Gemma stacks.

### Areas for Improvement

- **Naming and mental model.** The policy class remains **`WordleGPT2Policy`** while **`gemma_full`** loads Gemma. That is confusing in code reviews and reports; a neutral name (e.g. `WordleHFCausalPolicy`) would scale better.

- **Smoke profile is too weak to validate ES behavior.** **`N_POP=4`**, **`N_ITERATIONS=2`**, **`n_eval_episodes=1`**, and **`WARM_START_STEPS=12`** produce almost no statistical signal. Smoke proves **plumbing**, not that **rank-normalized ES** is stable or improving the policy. Week 10’s critique about **noisy fitness** still applies; smoke amplifies variance.

- **Committed results are still smoke-tier.** The notebook output checked into the repo reflects **`RUN_PROFILE="smoke"`** on CPU. There is still **no artifact-backed** **`gemma_full`** run in the template, so claims about Gemma + ES remain **hypothetical** until someone executes and logs a full profile.

- **Core scientific gaps from Week 10 are mostly untouched.** Eight-word mock, **warm-start doing most of the work**, **duplicate-letter bugs in `wordle_hints.py`**, **Prime vocabulary vs secret misalignment**, and **no matched baseline vs Week 6 MLP** are still open. Week 12 is infrastructure, not a resolution of those issues.

- **`gemma_full` + `MOCK_ENV=True` still uses an 8-word task.** Scaling the **model** without scaling **task difficulty** risks expensive runs that only show “large LM on a tiny classification problem.” The critique from Week 10—that **79% on 8 words** is not comparable to **156-word Week 6**—still applies to any Gemma mock-only numbers.

- **MPS / CPU dtype path.** `default_model_load_kwargs` returns float32 when not on CUDA; that is fine, but very large models on MPS may need explicit documentation (memory, speed) so users do not assume Gemma will be usable on laptop GPUs without tuning.

### Critical Risks / Assumptions

- **Assumption:** Chat-templated prompts for Gemma preserve the same **semantic content** as the plain prompt path (constraints, guesses, feedback). If template tokens or role boundaries **change truncation behavior** (`MAX_PROMPT_LENGTH`), comparisons across **`smoke`** vs **`gemma_full`** are not apples-to-apples without checking truncated prompt logs.

- **Risk:** **`gemma_full`** default **`USE_LORA=False`** means ES still perturbs mainly the **head**; running Gemma largely frozen may **underuse** the larger backbone unless LoRA or longer training is added—while still paying **full forward-pass** cost.

- **Risk:** Same as Week 10: validating only on **mock eight-word** Wordle does not test whether **LM + ES** beats **MLP + ES** on a harder vocabulary. The Week 12 stack could be **engineering-complete** but **scientifically unvalidated** on the hard regime.

## DECIDE

### Concrete Next Actions

1. **Execute `gemma_full` on GPU** with fixed seed; commit or appendix **one** full `history` + plot set, or explicitly state “not yet run” in the report if compute is pending.

2. **Add a mid profile** (e.g. `distil_full`): Week 10 hyperparameters + **`EVAL_EVERY=2` or `5`** on DistilGPT-2 for **cheap** but meaningful ES curves—bridging smoke and Gemma.

3. **Warm-start-only ablation** (carried from Week 10): after warm-start, eval **before** ES vs **after** ES on the same eval episodes to quantify ES contribution for each profile.

4. **Rename policy class** or add a thin alias exported as `WordleCausalLMPolicy` with a deprecation note for `WordleGPT2Policy`.

5. **Vocabulary experiment** (highest scientific priority): align secrets to `policy.words`, **`MAX_VOCAB` ≥ 64**, multi-seed; compare DistilGPT-2 vs Gemma at **matched** task and budget.

## ACT

### Resource Needs

- **GPU with sufficient VRAM** for **`google/gemma-3-1b-it`** (bf16/fp16) plus batch-size-one forward passes over long prompts; CPU **`gemma_full`** is likely only for debugging single steps.  
- **Disk / cache** for Gemma weights; first-time download can be large.  
- Optional: **`peft`** if LoRA experiments are prioritized.  
- **Wall-clock budget:** `gemma_full` ES is roughly “Week 10 cost × (Gemma forward pass factor)”; parallelizing population eval across processes remains an undeployed speedup for both weeks.
