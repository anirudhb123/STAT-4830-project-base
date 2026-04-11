# Week 12 Self-Critique

**Scope of this critique.** Week 12 is **implementation-complete** (profiles, Gemma path, device/dtype loading, checkpoints). **Gemma full-scale ES + warm-start results are not available yet**—those training jobs are **still running**. Judgments below separate **what the code enables** from **what we have measured**; the main gap is **pending Gemma metrics**, not missing plumbing.

## ORIENT

### Strengths

- **Run profiles directly address Week 10 workflow pain.** The **`RUN_PROFILE`** switch (`smoke` vs `gemma_full`) separates “does the notebook run end-to-end?” from “are we doing a serious experiment?” without maintaining two separate notebooks. This is a practical response to the Week 10 critique that full LM + ES is slow and hard to iterate on.

- **Denser eval logging in smoke mode.** With **`EVAL_EVERY=1`** and a short **`N_ITERATIONS`**, the eval curves actually have enough points to **see shape** during debugging. That partially fixes the Week 10 issue where **`EVAL_EVERY=10`** and **`N_ITERATIONS=10`** produced only **two** eval checkpoints—too sparse to interpret learning dynamics.

- **Gemma IT path is structurally supported.** `USE_CHAT_TEMPLATE` and **`CHAT_GENERATION_PROMPT`** wired into `WordleGPT2Policy` match how instruction-tuned models expect prompts, which matters for **`google/gemma-3-1b-it`** in **`gemma_full`**. That is a meaningful extension beyond “always format like GPT-2.”

- **Checkpoint isolation by profile.** Saving to **`models/wordle_gpt2_es_head.<RUN_PROFILE>.pt`** prevents a smoke run from clobbering a long Gemma checkpoint—a small change that avoids silent data loss.


### Areas for Improvement

- **Naming and mental model.** The policy class remains **`WordleGPT2Policy`** while **`gemma_full`** loads Gemma. That is confusing in code reviews and reports; a neutral name (e.g. `WordleHFCausalPolicy`) would scale better.

- **Smoke profile may be too weak to validate ES behavior.** **`N_POP=4`**, **`N_ITERATIONS=2`**, **`n_eval_episodes=1`**, and **`WARM_START_STEPS=12`** produce almost no statistical signal. Smoke proves **plumbing**, not that **rank-normalized ES** is stable or improving the policy. Week 10’s critique about **noisy fitness** still applies; smoke amplifies variance.

- **No Gemma results in the report yet—by design for now.** The notebook **defaults to `gemma_full`**, so checked-in outputs may already show Gemma training progress; this write-up still **does not** state final success curves or multi-seed stats. **Full-scale `gemma_full` jobs are in flight**; until they finish, any performance claim in prose is **speculative**. The Week 10 critique about needing real curves applies to what we **will** add to **Initial Results** after those runs land.

- **`gemma_full` + `MOCK_ENV=True` still uses an 8-word task.** Scaling the **model** without scaling **task difficulty** risks expensive runs that only show “large LM on a tiny classification problem.” The critique from Week 10—that **79% on 8 words** is not comparable to **156-word Week 6**—still applies to any Gemma mock-only numbers.

- **MPS / CPU dtype path.** `default_model_load_kwargs` returns float32 when not on CUDA; that is fine, but very large models on MPS may need explicit documentation (memory, speed) so users do not assume Gemma will be usable on laptop GPUs without tuning.

### Critical Risks / Assumptions

- **Assumption:** Chat-templated prompts for Gemma preserve the same **semantic content** as the plain prompt path (constraints, guesses, feedback). If template tokens or role boundaries **change truncation behavior** (`MAX_PROMPT_LENGTH`), comparisons across **`smoke`** vs **`gemma_full`** are not apples-to-apples without checking truncated prompt logs.

- **Risk:** **`gemma_full`** default **`USE_LORA=False`** means ES still perturbs mainly the **head**; running Gemma largely frozen may **underuse** the larger backbone unless LoRA or longer training is added—while still paying **full forward-pass** cost.

- **Risk:** Same as Week 10: validating only on **mock eight-word** Wordle does not test whether **LM + ES** beats **MLP + ES** on a harder vocabulary. The Week 12 stack could be **engineering-complete** but **scientifically unvalidated** on the hard regime.

## DECIDE

### Concrete Next Actions

1. **Complete in-progress `gemma_full` GPU runs**; with fixed seed, commit or appendix **one** full `history` + plot set and refresh `reportWeek12.md` **Initial Results**. (The report already states results are pending while jobs run.)

2. **Add a mid profile** (e.g. `distil_full`): Week 10 hyperparameters + **`EVAL_EVERY=2` or `5`** on DistilGPT-2 for **cheap** but meaningful ES curves—bridging smoke and Gemma.

3. **Warm-start-only ablation** (carried from Week 10): after warm-start, eval **before** ES vs **after** ES on the same eval episodes to quantify ES contribution for each profile.

## ACT

### Resource Needs

- **GPU with sufficient VRAM** for **`google/gemma-3-1b-it`** (bf16/fp16) plus batch-size-one forward passes over long prompts; CPU **`gemma_full`** is likely only for debugging single steps.  
- **Disk / cache** for Gemma weights; first-time download can be large.  
- Optional: **`peft`** if LoRA experiments are prioritized.  
- **Wall-clock budget:** `gemma_full` ES is roughly “Week 10 cost × (Gemma forward pass factor)”; parallelizing population eval across processes remains an undeployed speedup for both weeks.
