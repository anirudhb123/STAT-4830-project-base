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

## References

1. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. *arXiv:1703.03864*.  
2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.  
3. Gemma Team, Google DeepMind. (2025). *Gemma 3 Technical Report.* *arXiv:2503.19786*. [https://arxiv.org/abs/2503.19786](https://arxiv.org/abs/2503.19786) (PDF: [https://arxiv.org/pdf/2503.19786](https://arxiv.org/pdf/2503.19786)).

