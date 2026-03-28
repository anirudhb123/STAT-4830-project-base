# Wordle with DistilGPT-2, Warm-Start, and Evolution Strategies

## Problem Statement

**What are we optimizing?** We are training a policy to play Wordle by attaching a **linear head** to a frozen **DistilGPT-2** model (Hugging Face `transformers`). Each action is still a five-letter word from a fixed list; the head produces logits over that list. We update the head with **Evolution Strategies**, and the notebook supports optional **LoRA** on the transformer via `peft` if we want to train more than the head. Unlike the Week 6 setup, the policy does not read only a 64-dimensional embedding: it sees a **text prompt** built from the turn number, previous guesses, Wordle feedback, and (optionally) a short structured constraint summary from `wordle_hints.py`.

**Why does this problem matter?** In Week 6, an MLP on the fixed embedding with a large vocabulary reached only modest success. A small pretrained language model gives a much stronger starting representation for “what kind of word fits this feedback.” ES remains useful here because we still optimize from **rollout fitness** rather than differentiating through the environment. One practical issue we hit is that **the hidden answer must appear in the policy’s vocabulary**: if the environment draws secrets from a big dataset while the policy only allows a handful of words, many games are unwinnable and both supervised warm-start and ES look like they fail.

**How will we measure success?** We follow the metrics printed by `train_es_wordle` in `notebooks/week10_implementation.ipynb`: periodic evaluation gives **success rate** (fraction of eval episodes solved within six turns), **average reward**, and **average turns**. The logger also reports **ES_win**, the average win rate across the **perturbed** policies evaluated in each ES iteration, plus training diagnostics such as mean fitness, gradient norm, and parameter drift. The notebook plots eval curves on the iterations where evaluation runs and overlays per-iteration training statistics on the full ES horizon.

**Constraints and risks.** Forward passes through DistilGPT-2 dominate runtime; **CPU** training is workable but slow when `N_POP` and the number of ES iterations are large. The default configuration in the notebook uses a **mock** Wordle wrapper and restricts both **secrets** and **legal guesses** to the same **eight** words (`MOCK_WORDLE_TARGETS`), which makes the task easier than Week 6 but gives a clean signal that the pipeline is behaving (when trying this on the full Wordle dataset of ~2k words we saw no learning even with a warm start). Moving back to the full Prime-style environment while keeping a small action set would require either enlarging the vocabulary to cover dataset targets or restricting which episodes are sampled.

## Technical Approach

**Environment and reward.** We still use `src/wordle_env.py`. Actions are XML strings containing `<guess>WORD</guess>`; feedback and rewards follow the same rubric as in Week 6 (`WordleEnvironmentWrapper._simulate_prime_step` / mock path). For mock training, `load_wordle_environment(use_prime_intellect=False)` ensures resets draw targets from `MOCK_WORDLE_TARGETS` instead of pulling a random row from the verifier dataset when `verifiers` happens to be installed locally.

**Policy.** `src/wordle_gpt2_policy.py` tokenizes the prompt, runs DistilGPT-2 with the backbone frozen unless LoRA is enabled, takes the hidden state at the last non-padding position, and applies a linear map to logits over `policy.words`. Guesses already played are masked out before sampling or taking an argmax. The policy formats guesses as XML for the environment.

**Supervised warm-start.** Before ES, `src/wordle_gpt2_warmstart.py` can run several hundred steps of supervised learning: we roll out a few random legal guesses, then minimize cross-entropy from the model logits to the **index** of the true word. The answer is never pasted into the prompt; it is only the label. Steps are skipped when there is no usable prefix or when the target is missing from `policy.word_to_idx`, and the notebook prints how many steps were skipped.

**Evolution Strategies.** We reuse the Wordle ES loop in `src/es_wordle.py`, estimating a gradient over flattened trainable parameters with Gaussian perturbations:

```
grad_theta J(theta) ≈ (1/(N*sigma)) * sum_i R(theta + sigma*epsilon_i) * epsilon_i
where epsilon_i ~ Normal(0, I)
```

Here `R` stands in for whatever scalar fitness we assign to each perturbation (mean return, win rate, or **win-plus-return** with a scale on wins). In code, those values are **z-scored or rank-normalized** across the population before they are combined with the noise vectors `epsilon_i`. We can also use a **normalized** parameter update (`normalize_gradient=True` in the notebook).

**Experimental configuration (from `notebooks/week10_implementation.ipynb`).** The hyperparameter cell is the source of truth; the table matches the **mock, `MOCK_ENV=True`** path as of the current notebook.


| Setting | `TRAIN_BUDGET="long"` | `TRAIN_BUDGET="fast"` |
| ------- | --------------------- | --------------------- |
| `MAX_VOCAB` | 8 (= `len(MOCK_WORDLE_TARGETS)`) | same when mock |
| ES population / iters | `N_POP=16`, `N_ITERATIONS=10` | `N_POP=8`, `N_ITERATIONS=40` |
| `SIGMA` / `ALPHA` | 0.02 / 0.12 | same |
| Fitness | `FITNESS_OBJECTIVE="win_plus_return"`, `WIN_FITNESS_SCALE=8.0` | same |
| `RANK_FITNESS` / `NORMALIZE_GRADIENT` | both `True` | same |
| Rollouts per ES member | `n_eval_episodes=2` | `1` |
| Logging eval | `EVAL_EVERY=10`, `EVAL_N_EPISODES=24`, `EVAL_DETERMINISTIC=True` | `EVAL_N_EPISODES=16` |
| Warm-start | `WARM_START_STEPS=400`, `WARM_START_LR=3e-4` | 200 steps |

Other toggles in the same cell include `MODEL_NAME` (default **distilgpt2**), `USE_LORA=False`, `RICHER_PROMPT=True`, and `USE_PRIME_TARGETS=False` for the common-word vocabulary builder. Setting `MOCK_ENV=False` switches to Prime loading in the wrapper and uses `MAX_VOCAB` 64 (long) or 256 (fast) unless you change the cell—you then need secrets and actions aligned, as in the problem statement above.

Code and artifacts:

- `notebooks/week10_implementation.ipynb` (training, plots, optional checkpoint save)
- `src/wordle_gpt2_policy.py`, `src/wordle_gpt2_warmstart.py`, `src/es_wordle.py`, `src/wordle_env.py`, `src/wordle_hints.py`
- `models/wordle_gpt2_es_head.ipynb_run.pt` (head weights, word list, and `history` dict)

## Initial Results

On the **eight-word mock** with warm-start enabled, **greedy** evaluation at the start of ES is already strong. The supervised phase learns to map typical feedback patterns to the correct word among eight options, so the first logged eval in a full run can show high success and fewer than six turns on average. During ES, **ES_win** varies from iteration to iteration because each population member uses perturbed weights, so some perturbations help and others hurt even when the center policy is good. When **normalize_gradient** is turned on, the printed **Step‖** stays nearly constant because each update uses a fixed learning rate times a unit-norm direction.

## Next Steps

**Immediate improvements.**

1. **Prime data with a narrow policy.** If we keep a small `MAX_VOCAB`, we should only reset episodes whose target lies in `policy.words`, or build the word list from the verifier train and eval targets so every sampled game is solvable.
2. **Curriculum Learning.** Increase vocabulary size in stages while preserving the invariant that every secret is an allowed action.
3. **Fair comparison to Week 6.** Match total environment steps and seeds where possible, and report mean and spread across seeds rather than a single long run.
4. **Larger models and longer budgets.** Move off CPU to **GPU** for DistilGPT-2 (or swap `MODEL_NAME` to full **GPT-2** or another HF causal LM), increase `N_ITERATIONS` and `N_POP`, and rerun with the same logging so we can see whether ES continues to improve once the easy eight-word mock is saturated.

**Technical improvements.**

- Turn on **LoRA** on GPU and compare sample cost and final success to head-only ES, building on the Week 8 theme of low-rank adaptation.
- Add **mirrored sampling** or a schedule on σ to reduce variance in the ES gradient estimate.


## References

1. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. *arXiv:1703.03864*.
2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS EMC² Workshop*.
3. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.
