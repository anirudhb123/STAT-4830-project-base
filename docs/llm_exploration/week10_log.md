# Week 10 LLM Conversation Log

**Date Range:** March 23 - April 2, 2026
**Focus:** Graduating from GridWorld to Wordle — first LM-backed policy, supervised warm-start, and the first end-to-end ES + LM run

---

## Session 1: Choosing the Next Problem (Mar 23)

### Conversation with ChatGPT

**My Question:**
> I have ES + LoRA working on GridWorld. I want a domain where ES is genuinely interesting — sparse reward, discrete action space, and where it’s natural to plug in a pretrained language model as the policy backbone. What’s a good candidate?

**Response Summary:**
- Suggested **Wordle** as a near-ideal toy problem: 5-letter discrete action space, sparse reward (you only know if you won at the end), well-defined evaluation, and a vocabulary small enough to put a softmax head on.
- Mentioned alternatives — math-token games, single-step QA — but flagged them as either too dense (math) or too unconstrained (QA).
- Recommended starting with a **mock target pool** (a few dozen secrets) before pulling in a full Wordle answer list, so I could verify the loop with a small action set and grow from there.

**Key Takeaway:**
Wordle hits the sweet spot: the policy must learn to **use feedback** (yellow/green letters) across turns, not just memorize a static state→action map, but the action space is small enough to put a categorical head on and treat as a classification problem at each turn.

**What I Did:**
- Drafted a Wordle environment spec with a `MOCK_WORDLE_TARGETS` pool of ~16 words plus a hook to load a full 2300-word answer list from Prime Intellect later.
- Reward shape: `+1` for correct guess, partial credit for green/yellow letter coverage on each turn, `0` otherwise.

---

## Session 2: Wordle Environment Design (Mar 24)

### Conversation with Cursor AI

**My Request:**
> Build a Gym-style Wordle environment that takes a string action like `<guess>CRANE</guess>`, parses it, scores it against a hidden secret, and returns turn-level structured feedback (per-letter green/yellow/grey).

**Response Summary:**
- Suggested an env that holds `current_state` with `previous_guesses`, `previous_feedback`, and `turn_number`.
- Proposed parsing the action XML strictly: `<guess>WORD</guess>` (and an optional `<think>...</think>` block that is ignored for scoring) so a future LM-backed policy could “think out loud” without polluting the action.
- Recommended an explicit `set_target_pool(words)` API so a curriculum could swap the env’s sampling distribution without rebuilding the env.

**Key Takeaway:**
Decoupling the *action vocabulary* (what words the policy can emit) from the *secret pool* (what words the env can pick) was a small interface choice that paid off all month — the curriculum, the warm-start, and the held-out eval all hang off this seam.

**What I Did:**
- Implemented `wordle_env.py` with `WordleEnvironmentWrapper`, `set_target_pool`, and a feedback record per turn.
- Added a parse-and-validate path that gracefully rejects malformed XML / non-vocab guesses (returns `done=True` with a small penalty rather than crashing the rollout).

---

## Session 3: Discrete Policy on Top of an LM (Mar 27)

### Conversation with Claude

**My Question:**
> I want the policy to be a small **linear head** over a fixed action vocabulary, sitting on top of a frozen language model’s last hidden state at the end of the prompt. The LM never gets fine-tuned in this version. What’s a clean way to wire this so I can swap backbones (distilGPT-2 ↔ Gemma-3-1b) and toggle LoRA later?

**Response Summary:**
- Recommended a `WordleGPT2Policy` class that owns three things: the LM (`AutoModel.from_pretrained(...)`), a tokenizer, and a linear `head: nn.Linear(hidden_dim, action_dim)`.
- Suggested two prompt formats:
  - **Plain text:** “Guess #2. Previous: CRANE → green/yellow/grey/grey/green …”
  - **Chat-template** (for instruction-tuned models like Gemma-3-1b-it): apply the model’s `apply_chat_template` so the prompt is wrapped in `<start_of_turn>user … <end_of_turn>` etc.
- Strongly recommended exposing a `forward_logits_batch(states)` method that batches the LM forward across many active games — without it, a 16-member ES population × 16 episodes/member would call the LM 256 times serially, which on Gemma is a wall-clock disaster.

**Key Design Choices:**
- Action space is a **fixed superset** of all curriculum stage pools, so the head dimension never changes (no per-stage `expand_vocab`); the env simply restricts the secret pool.
- `previous-guess masking`: at action selection time, set logits for already-guessed words to `−∞` so the policy can’t repeat itself.
- `richer_prompt=True` toggles in the structured constraint section (“letters known to be in word: A, R; letters known not in word: T, S; positions known: _R___”) so the LM has the per-turn deductive state without re-deriving it from raw feedback.

**What I Did:**
- Implemented `WordleGPT2Policy` with `format_action_xml`, `get_action`, and `forward_logits_batch`.
- Verified end-to-end that distilGPT-2 + 16-word vocab loads, runs a single rollout, and produces a sensible (if random) guess.

---

## Session 4: First ES Run on Wordle — “0% Forever” (Mar 29)

### Conversation with ChatGPT

**My Question:**
> ES on Wordle with distilGPT-2 + a linear head: success rate is 0% after 80 iterations and every single ES population member produces the same fitness. Same failure mode as Week 7 sparse-reward GridWorld?

**Response Summary:**
- Yes — same root cause, different domain. With Wordle’s ~1 / vocab_size random win probability, almost every perturbation produces fitness = 0 across all eval episodes, fitness standardization divides by ~0, and the gradient is noise.
- Recommended the same Week-7 fix in spirit (dense training signal) but pointed out that **fitness standardization itself can be the wrong shaping** when wins are sparse: rank-fitness (centered ranks) is more robust because ties don’t kill the signal as catastrophically as `std → 0`.
- Suggested two complementary additions:
  1. A `fitness_objective="win_plus_return"` mode that explicitly mixes win-rate (sparse) with return (dense partial credit) so even losing rollouts produce differentiated fitness.
  2. **Antithetic sampling** + **common-random-numbers (CRN)** so within an iteration every population member sees the same secret words and the same sampling RNG, isolating the policy difference.

**Key Takeaway:**
The standardization-collapse failure mode generalizes from GridWorld to Wordle but with a worse penalty: in Wordle there is *no* dense per-turn shaping that doesn’t require reward engineering, so the partial-credit return + win bonus design is what carries the per-iteration ranking.

**What I Did:**
- Added `rank_fitness=True`, `fitness_objective="win_plus_return"`, `win_fitness_scale=8.0`, `antithetic=True`, `common_random_numbers=True` to `src/es_wordle.py`.
- These four flags turned the “0% forever” run into a “bouncing 0–12% on tiny pool” run — clearly noisy, but no longer a flat line.

---

## Session 5: Hyperparameter Spelunking (Mar 31)

### Conversation with Cursor AI

**My Request:**
> Help me sweep `(σ, α, N)` for the Wordle ES head. I want to know which combinations actually move the policy and which produce a flat parameter trajectory.

**Response Summary:**
- Suggested instrumenting **`‖θ − θ₀‖`** (parameter drift), **`Step‖`** (actual update norm), and **`Grad‖`** (raw ES gradient norm) every iteration in addition to fitness.
- Pointed out that **`Step‖` is the right knob to calibrate**, not `α` directly: the same `α` produces wildly different `Step‖` depending on `(N, σ, n_params)` because `‖ĝ‖ ∝ √(n_params / N) / σ` for ES.
- Recommended targeting an **initial step of ~0.13** (LoRA-friendly Adam-equivalent magnitude) and back-solving for `α`.

**What I Did:**
- Added `param_drift`, `step_norm`, and `grad_norm` to the per-iteration history dict and printed them every iter.
- Implemented an in-cell ALPHA calibration probe that runs one ES gradient estimate at init, measures `‖ĝ‖`, and reports the suggested `α` to hit the 0.13 target.

**Failed Attempt:**
- Tried `σ = 0.1` (carryover from GridWorld). Result: even with rank fitness the head exploded after 5 iterations — `‖θ − θ₀‖ → 30+`, success collapsed to 0%. Reverted to `σ = 0.02`.

---

## Session 6: Supervised Warm-Start (Apr 2)

### Conversation with Claude

**My Question:**
> The ES head wanders for hundreds of iterations before producing any wins. Can I supervise-warm-start it on the secret word as a label, so ES has something better than random init to refine?

**Response Summary:**
- Yes — and the design discipline is to **never put the secret in the prompt**. The policy must still learn “given (history of guesses + feedback), the answer is X”, not “echo the answer”.
- Recommended a “random pre-play, then CE on the answer” loop:
  - Pick a secret, simulate 1–4 random opening guesses, record the resulting state.
  - Cross-entropy loss between the head’s logits and the secret’s vocab index.
- Warned against running too many warm-start steps: if WS saturates the head to ≈100% success on the training pool, ES has zero headroom and looks broken downstream. Suggested an explicit “post-WS success ceiling” (e.g. 0.85) and skipping WS on later curriculum stages once the ceiling is hit.

**What I Did:**
- Implemented `supervised_warm_start_wordle` in `src/wordle_gpt2_warmstart.py` with a `feedback_consistent_random=True` toggle (random pre-play words constrained by accumulated feedback so the supervised target is a sensible inference, not “predict random secret given garbage prefix”).
- Added a `quick_eval_success` helper for the pre/post-WS diagnostic.

**First End-to-End Result (16-word vocab, distilGPT-2, no LoRA, no curriculum):**
- Random init: 0% greedy success.
- After 50 WS steps: ~60% greedy success on the training pool.
- After 30 ES iterations on top: ~62% greedy success.
- ES contribution: roughly noise.

That last number became the question that defined Weeks 11 and 12: **is ES actually doing anything on top of warm-start, or just diffusing the head around the WS optimum?**

---

## Key Lessons from LLM Interactions

### What Worked Well
1. **Carrying the Week-7 diagnosis forward.** Knowing that “fitness standardization + sparse signal → zero update” was the failure mode in GridWorld saved me from re-discovering it on Wordle. The fix (rank-fitness + `win_plus_return`) was a direct port.
2. **Asking for the right diagnostic, not a fix.** Claude’s suggestion to instrument `‖θ − θ₀‖` and `Step‖` separately from `α` immediately exposed the *real* knob (initial-step magnitude) hiding behind the *named* knob (learning rate).
3. **Using Cursor for the boilerplate (env + policy class) and ChatGPT/Claude for the design tradeoffs.** Cursor wrote the parsing / batching / masking code; the LLMs argued for/against design choices.

### What Didn’t Work
1. **Open-vocabulary generation.** First attempt let the LM `generate(...)` a 5-letter completion. Result: ~30% of guesses were not in the answer list, the env rejected them, and the rollouts ended in 1–2 turns with no signal. Switched to a categorical head over a fixed vocabulary the same day.
2. **No previous-guess masking.** The agent quickly learned to repeat its first guess every turn — argmax stayed on the same word. Adding `−∞` masking on previously-guessed action indices fixed this in 5 lines.
3. **σ tuning carryover.** GridWorld `σ = 0.1` was wildly too large for the LM head (different parameter scale, different output sensitivity). The “try the prior week’s default first” heuristic burned ~3 hours of runs.

### Best Practices Learned
1. **Decouple action vocab from secret pool.** Made the curriculum trivial later.
2. **Calibrate `Step‖`, not `α`.** Lets you change `(N, σ, n_params)` without re-tuning the LR.
3. **Always have a “does it learn at all?” probe.** Even a 16-word vocab + 50 WS-only steps gives you a clear go/no-go before committing to a Gemma-scale run.

---

## Tools Used

**ChatGPT (GPT-4):**
- Choosing Wordle as the next problem
- Diagnosing the “0% forever” failure mode (rank-fitness + CRN + antithetic)
- ~10 conversations, ~2 hours total

**Claude (Sonnet):**
- Policy class design (linear head over LM hidden state)
- Supervised warm-start design (no-leak constraint, headroom ceiling)
- ~6 conversations, ~1 hour total

**Cursor AI:**
- Wordle env implementation (parsing, feedback, vocab swap)
- ES-on-Wordle module skeleton (`src/es_wordle.py`)
- Hyperparameter sweep instrumentation
- ~4 hours of pair-programming

---

## Impact Assessment

**Time Saved:** ~7 hours total
- Faster GridWorld → Wordle transition (re-used Week-7 ES knowledge instead of re-discovering it)
- Avoided two known-bad initial choices (open-vocab gen, no previous-guess masking) before they wasted compute

**Quality Improvement:**
- ES + LM pipeline now has the same diagnostic surface as Week-4 GridWorld ES (fitness, gradient norm, param drift, success), so cross-week comparison is possible.
- Warm-start was added with the post-WS ceiling discipline up front, which paid off heavily in Week 11/12.

**Open Questions Heading Into Week 11:**
1. Does warm-start memorize the training secret pool, or does it generalize?
2. With Gemma-3-1b instead of distilGPT-2, does the LM prior absorb so much of the policy that the LoRA delta + ES updates can’t move argmax?
3. What does a curriculum (vocab=16 → 32 → 64 → … → 1024) actually buy us, vs. one big stage?

---

*Log completed: April 2, 2026*
