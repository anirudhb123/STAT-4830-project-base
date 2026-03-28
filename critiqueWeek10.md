# Week 10 Self-Critique

## ORIENT

### Strengths
- **Meaningful architectural upgrade:** Moving from a 64-dimensional hand-crafted embedding with an MLP (Week 6) to a frozen DistilGPT-2 backbone with a trainable linear head is a well-motivated step. The pretrained language model provides much richer representations of Wordle feedback than the position-unaware letter-status embedding, directly addressing Week 6's biggest identified weakness.

- **Supervised warm-start is a smart addition:** The cross-entropy warm-start phase (`wordle_gpt2_warmstart.py`) gives the linear head a reasonable initialization before ES begins. The logged output shows warm-start loss decreasing from ~2.26 to ~2.08 over 400 steps, and the first ES eval already reports 70.8% greedy success — meaning ES starts from a competent policy rather than a random one. This is a practical and efficient design choice.

- **Richer, more principled prompt engineering:** `wordle_hints.py` constructs a structured constraint summary (known greens by position, yellows, grays) and feeds it into the LM prompt. This lets the pretrained model leverage its language understanding of constraints rather than requiring the policy to rediscover Wordle logic from scratch.

- **Improved ES diagnostics and logging:** The training loop now tracks per-iteration metrics (`train_fitness`, `train_es_win`, `param_drift`, `pop_fitness_std`, `train_grad_norm`) in addition to periodic eval checkpoints. The 2×3 plot grid and the verbose iteration-level printout make it much easier to diagnose training dynamics compared to Week 6.

- **Clean modular code:** The separation into `wordle_gpt2_policy.py`, `wordle_gpt2_warmstart.py`, `wordle_hints.py`, and `es_wordle.py` keeps responsibilities clear. The policy cleanly handles prompt construction, LM forward pass, logit masking of repeated guesses, and XML formatting — all without leaking environment internals.

### Areas for Improvement
- **The task is far easier than Week 6, not harder:** The default configuration restricts both the secret pool and the action space to the same 8 words (`MOCK_WORDLE_TARGETS`). Week 6 used 156 target words. An 8-word vocabulary makes Wordle a near-trivial classification problem — randomly guessing would win ~17% of the time even without feedback, and the warm-start alone reaches 70.8%. The 79.2% final success rate after 10 ES iterations cannot be meaningfully compared to Week 6's results on a 20× larger action space. The report acknowledges this mismatch but still frames the pipeline as an improvement rather than an untested hypothesis on a harder setting.

- **ES shows minimal improvement over warm-start:** Success goes from 70.8% (iteration 0, post-warm-start) to 79.2% (iteration 9) — a gain of ~8.4 percentage points over 10 iterations. Meanwhile, `ES_win` fluctuates between 62.5% and 87.5% with no clear upward trend across iterations, and `pop_fitness_std` stays in the 2–3 range throughout. With only 10 iterations and 2 eval checkpoints (iterations 0 and 9), it is impossible to determine whether ES is genuinely improving the policy or whether the gain is within noise. The report does not compute confidence intervals or multi-seed statistics.

- **Only 2 evaluation checkpoints make trend analysis impossible:** `EVAL_EVERY=10` with `N_ITERATIONS=10` means full eval runs only at iterations 0 and 9. The eval success curve in the plot has exactly 2 data points. This is insufficient to draw any conclusions about learning dynamics, convergence, or whether ES is helping at all. The "bottom row" per-iteration training metrics (fitness, param drift) are useful but do not substitute for periodic eval success measurements.

- **No comparison to Week 6 or any baseline:** Week 6 ran ES and PPO side-by-side, and the critique specifically called for a fair comparison with matched budgets. Week 10 drops the PPO comparison entirely and changes the environment difficulty simultaneously, making it impossible to isolate the effect of the DistilGPT-2 policy upgrade. Even a simple random-policy baseline or a warm-start-only baseline (no ES) on the same 8-word task would help contextualize the results.

- **Warm-start hint leaks make the task even easier:** The warm-start procedure in `wordle_gpt2_warmstart.py` resets the environment (which sets `target_word`), plays 1–4 random guesses, and then trains the head to predict the target's index from the resulting prompt. Because the action space equals the secret pool (8 words) and the constraint summary in the prompt often narrows candidates substantially after a few guesses, the warm-start is essentially fine-tuning on a near-deterministic classification task. This is not a problem per se, but it means the reported success rate conflates "the LM representation is good" with "8-word Wordle is trivially solvable with a few hundred cross-entropy steps."

- **`wordle_hints.py` does not handle duplicate-letter edge cases:** The module's docstring acknowledges this. For the 8-word mock (no duplicate-letter targets), this is fine, but it will produce incorrect constraint summaries on a real vocabulary, silently degrading prompt quality.

### Critical Risks/Assumptions
The central risk is that the entire pipeline — warm-start, ES, richer prompts, DistilGPT-2 — has only been validated on an 8-word toy problem. When the report lists 79.2% success and describes the policy as "already strong," these claims rest on a task where a well-tuned lookup table would likely reach 100%. The pipeline's value proposition (pretrained LM + ES scales better than MLP + ES) remains untested. If the approach does not generalize to 100+ word vocabularies, the DistilGPT-2 overhead (tokenization, 82M-parameter forward passes, HF dependencies) is pure cost with no benefit over the simpler Week 6 architecture.

A secondary risk is that the mock environment and the Prime-backed environment share code paths (`_simulate_prime_step` / `_mock_step` are identical), but the Prime path draws targets from a dataset that may not overlap with `policy.words`. The report notes this alignment problem but has not solved it. Switching `MOCK_ENV=False` without resolving the vocabulary mismatch will silently produce near-zero win rates and waste compute.

## DECIDE

### Concrete Next Actions
1. **Run on a non-trivial vocabulary (highest priority):** Increase `MAX_VOCAB` to at least 64–128 words, align the secret pool to the policy's action list (filter Prime dataset targets to only those in `policy.words`, or build the word list from Prime targets), and rerun. This is the minimum experiment needed to assess whether the DistilGPT-2 approach provides any advantage over Week 6.

2. **Add a warm-start-only baseline:** After the supervised phase, evaluate the policy *without* running ES. Compare this to the post-ES policy on the same eval episodes and seeds. If warm-start alone achieves similar success rates, ES is not contributing and the hyperparameters or iteration count need adjustment.

3. **Increase ES iterations and eval frequency:** With `EVAL_EVERY=10` and `N_ITERATIONS=10`, there are only 2 eval points. Either increase `N_ITERATIONS` to 50–100 (with `EVAL_EVERY=5`) or decrease `EVAL_EVERY` to 1–2 so that learning curves actually have enough resolution to show trends. Report mean ± standard deviation across at least 3 seeds.

4. **Restore a fair comparison to Week 6:** Run the Week 6 MLP+ES pipeline on the same word list, same number of environment episodes, and same seeds. Plot both methods' eval success vs. total environment interactions (not iterations, since per-iteration cost differs).

5. **Fix the duplicate-letter constraint bug in `wordle_hints.py`:** Before scaling to a real vocabulary, update `build_constraint_summary` to handle duplicate letters correctly (a letter can be GREEN in one position and GRAY in another if it appears once in the target but twice in the guess). Incorrect constraints in the prompt will mislead the LM.

## ACT

### Resource Needs
The current CPU-only setup is adequate for 8-word mock experiments but is acknowledged as slow even for `N_POP=16` and `N_ITERATIONS=10`. Scaling to 100+ words with 50+ ES iterations and multi-seed runs will require either GPU acceleration (DistilGPT-2 forward passes dominate wall-clock time) or a significant time budget. Parallelizing ES population evaluations across CPU cores (the perturbations are independent) would also help. No new dependencies are needed beyond what is already installed, unless LoRA experiments are prioritized (requires `peft`).
