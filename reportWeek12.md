# Wordle with Profiles (Smoke / Gemma), Device-Aware Loading, and ES

## Problem Statement

**What are we optimizing?** Same core objective as Week 10: a policy that plays Wordle by attaching a **linear head** to a **frozen** Hugging Face causal LM, with actions as five-letter words from a fixed list. Training still combines optional **supervised warm-start** (`wordle_gpt2_warmstart.py`) and **Evolution Strategies** (`train_es_wordle` in `src/es_wordle.py`). The policy consumes a **text prompt** (turn, guesses, feedback, optional structured constraints from `wordle_hints.py` when `RICHER_PROMPT=True`).

**What changed in Week 12?** `notebooks/week12_implementation.ipynb` adds a **run profile** switch so one notebook supports (1) a **fast smoke-test** configuration for pipeline checks and (2) a **heavier “Gemma full”** configuration aligned with the larger **`google/gemma-3-1b-it`** checkpoint. It also adds **device selection** (CUDA / MPS / CPU), **GPU-friendly model loading** (`bfloat16` or `float16` on CUDA when supported), **instruction-tuned formatting** via the tokenizer’s **chat template** when using Gemma, a configurable **`MAX_PROMPT_LENGTH`**, and **profile-specific checkpoint filenames** so quick runs do not overwrite long jobs.

**Why does this matter?** Week 10’s critique noted that full ES + LM runs are **slow** and that **sparse eval checkpoints** made curves hard to interpret. Week 12’s **smoke** profile reduces warm-start, population size, iterations, and eval episodes so the stack can be verified quickly; **gemma_full** restores a Week-10-scale budget while targeting a **stronger backbone**. The Week 10 **vocabulary / secret alignment** issue for Prime-backed games is unchanged: mock mode with **`MOCK_WORDLE_TARGETS`** still keeps secrets and actions aligned.

**How will we measure success?** Same metrics as Week 10: periodic **eval_reward**, **eval_success**, **eval_turns**; per-iteration **train_fitness**, **param_drift**, **pop_fitness_std**, and **train_es_win** inside `history`. With **`RUN_PROFILE="smoke"`**, **`EVAL_EVERY=1`** yields **dense eval points** over the (short) ES horizon for debugging plots. With **`gemma_full`**, eval spacing matches the prior **`EVAL_EVERY=10`** style run.

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
| `MAX_VOCAB` (with `MOCK_ENV=True`) | 8 | 8 |
| ES | `N_POP=4`, `N_ITERATIONS=2` | `N_POP=16`, `N_ITERATIONS=10` |
| `n_eval_episodes` | 1 | 2 |
| Logging eval | `EVAL_EVERY=1`, `eval_n_episodes=4` | `EVAL_EVERY=10`, `eval_n_episodes=24` |
| Warm-start | `WARM_START_STEPS=12` | `WARM_START_STEPS=400` |
| Env episode counts | `num_train_examples=128`, `num_eval_examples=16` | `2000` / `20` |
| `USE_LORA` (default in §2) | `False` | `False` |

Global knobs in §2 still include `SIGMA=0.02`, `ALPHA=0.12`, `RICHER_PROMPT=True`, `FITNESS_OBJECTIVE="win_plus_return"`, `WIN_FITNESS_SCALE=8.0`, `MOCK_ENV=True`, `USE_LORA=False` unless you enable PEFT.

**Artifacts.** Checkpoints save to `models/wordle_gpt2_es_head.<RUN_PROFILE>.pt` (head / optional LoRA state, `words`, `history`).

**Code paths.**

- `notebooks/week12_implementation.ipynb`  
- `src/wordle_gpt2_policy.py`, `src/wordle_gpt2_warmstart.py`, `src/es_wordle.py`, `src/wordle_env.py`, `src/wordle_hints.py`

## Initial Results

The notebook’s **checked-in execution** uses **`RUN_PROFILE="smoke"`** on **CPU** (DistilGPT-2, tiny ES budget). That run is meant to **validate imports, env reset, warm-start, ES loop, and plotting**—not to report competitive Wordle performance. For **quantitative** claims comparable to Week 10’s ~10-iteration DistilGPT-2 mock curves, run **`gemma_full`** (or **`smoke=False`** with DistilGPT-2 by adding a profile) on **GPU**, fix seeds, and record `history` and plots in the notebook or appendix.

**Expectation from Week 10:** On the **eight-word mock** with substantial warm-start and ES budget, greedy eval can start strong after warm-start; ES may add a modest success-rate lift with noisy per-iteration fitness. The same qualitative behavior should transfer once Gemma runs complete, modulo compute and possible hyperparameter retuning for the larger model.

## Next Steps

1. **Run `gemma_full` on GPU** with aligned dtype (`model_load_kwargs` on CUDA), log full `history`, and compare success / turns / wall-clock to Week 10 DistilGPT-2 at matched mock settings.  
2. **Optional LoRA** (`USE_LORA=True`, `peft`) on GPU: compare trainable parameter count vs head-only ES.  
3. **Scale vocabulary and align secrets** (Week 10 next steps): filter Prime targets to `policy.words` or grow `MAX_VOCAB` with curriculum.  
4. **`wordle_hints.py`**: fix duplicate-letter constraint edge cases before relying on `RICHER_PROMPT` at scale.  
5. **Multi-seed reporting**: mean ± std for eval success over ≥3 seeds for both profiles when reporting results.

## References

1. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. *arXiv:1703.03864*.  
2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.  
3. Gemma Team, Google DeepMind. (2025). *Gemma 3 Technical Report.* *arXiv:2503.19786*. [https://arxiv.org/abs/2503.19786](https://arxiv.org/abs/2503.19786) (PDF: [https://arxiv.org/pdf/2503.19786](https://arxiv.org/pdf/2503.19786)).

