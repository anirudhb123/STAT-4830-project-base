# Wordle with Evolution Strategies vs. PPO

## Problem Statement

**What are we optimizing?** We are training a neural network policy π_θ to play Wordle: guess a 5-letter target word within 6 attempts. Each action is a word guess chosen from a discrete vocabulary (loaded from the Prime Intellect dataset; **156 words** in our current run). The environment returns Wordle-style feedback (GREEN / YELLOW / GRAY) and a shaped reward that combines correctness, partial credit from feedback, solving speed, and format adherence.

**Why does this problem matter?** Wordle has a large discrete action space and a reward that is sparse at the task level (solving vs. not solving), which makes it a good testbed for comparing **gradient-free parameter-space optimization** (Evolution Strategies) against a standard **gradient-based RL baseline** (PPO). The comparison is especially interesting when the learning signal is weak and exploration is difficult.

**How will we measure success?** Our primary metric is *success rate*: the fraction of evaluation games solved within 6 turns. Secondary metrics include average total reward per game and average turns. We report both (1) a **post-training evaluation on 50 fresh games**, and (2) the **training-time evaluation summary** logged during training.

**Constraints and risks.** The policy is trained on a fixed-size numeric embedding of a text game state (64-dim), so representation quality is a key bottleneck. The action space is limited to a vocabulary subset (156 words here), which can cap attainable success. Finally, ES and PPO were run with very different episode budgets in this iteration, so raw success rates are not yet a fair sample-efficiency comparison.

## Technical Approach

**Environment and reward.** We use an adapter around Prime Intellect’s Wordle verifier environment (`src/wordle_env.py`). Actions are XML-formatted strings with a `<guess>WORD</guess>` tag, and feedback is generated/parsed in the Wordle format. The reward matches the rubric implemented in `WordleEnvironmentWrapper._simulate_prime_step`:

```
reward = correct_answer + 0.3*partial_answer + 0.5*length_bonus + 0.1*format_reward
```

where `partial_answer` is computed from GREEN/YELLOW feedback counts, `length_bonus` rewards solving in fewer turns, and `format_reward` rewards valid guesses.

**State representation.** We encode a `WordleState` into a fixed 64-d vector (`WordleEnvironmentWrapper.get_state_embedding`):

- Turn features: normalized turn number, guesses count, and done flag
- Letter features (52 dims): for each letter A–Z, a “tried” indicator and a “status” value (0.0 / 0.5 / 1.0 for unknown / YELLOW / GREEN)

**Policies (discrete action space).** The Wordle policy (`src/wordle_policy.py`) is a multi-layer MLP mapping the 64-d embedding to logits over the vocabulary. We support masking previously-guessed words and formatting actions as XML via `format_action_xml`. PPO uses a separate value network (`WordleValueNetwork`) with the same embedding input.

**Evolution Strategies (ES).** We implement ES for Wordle in `src/wordle_es.py`, estimating a gradient over flattened parameters using Gaussian perturbations:

```
grad_theta J(theta) ≈ (1/(N*sigma)) * sum_i R(theta + sigma*epsilon_i) * epsilon_i
where epsilon_i ~ Normal(0, I)
```

We standardize fitness across the population each iteration to reduce estimator variance.

**PPO baseline.** PPO is implemented in `src/ppo_training.py` (`train_ppo_wordle`) with:

- A rollout buffer (`RolloutBuffer`)
- GAE (`compute_gae`, with γ=0.99, λ=0.95)
- PPO-Clip objective with ε=0.2
- Separate optimizers for policy and value networks

**Experimental configuration (from `notebooks/week6_implementation.ipynb`).**


| Method | Key hyperparameters                                                                 | Episodes / iteration | Total episodes |
| ------ | ----------------------------------------------------------------------------------- | -------------------- | -------------- |
| ES     | N=80, σ=0.25, α=0.1, iterations=150, eval_episodes=10                               | 800                  | 120,000        |
| PPO    | iterations=50, episodes/iter=10, entropy=0.02, ε=0.2, lr_policy=3e-4, lr_value=1e-3 | 10                   | 500            |


Code and artifacts:

- `notebooks/week6_implementation.ipynb` (training + plots)
- Figures saved under `figures/` (e.g., `wordle_es_ppo_comparison.png`, `wordle_learning_curves.png`)
- Models saved under `models/` (e.g., `wordle_es_policy.pt`, `wordle_ppo_policy.pt`, `wordle_ppo_value.pt`)

## Initial Results

**Post-training evaluation (50 episodes each).** This is the “Final Evaluation (50 episodes each)” block printed in the notebook.


| Method | Success rate | Avg reward | Avg turns |
| ------ | ------------ | ---------- | --------- |
| ES     | 6.0%         | 1.188      | 5.94      |
| PPO    | 0.0%         | 0.926      | 6.00      |


**Training-time evaluation summary.** This is the later “FINAL RESULTS SUMMARY” block printed in the notebook, which reports the last logged training-eval values (and the best observed).


| Method | Final success | Best success | Final avg reward | Final avg turns | Total episodes |
| ------ | ------------- | ------------ | ---------------- | --------------- | -------------- |
| ES     | 10.0%         | 15.0%        | 1.134            | 5.60            | 120,000        |
| PPO    | 10.0%         | 10.0%        | 1.062            | 6.00            | 500            |


**Why do these differ?** The post-training evaluation and the training-time summary are computed from **different evaluation procedures and sample sizes** (a dedicated 50-episode eval vs. periodic eval during training). At low success rates, these estimates have high variance, so it is possible for training-time success to be non-zero while a separate 50-episode evaluation reports 0% (or vice versa).

**Sample-efficiency note.** ES achieved higher success in these runs, but used a vastly larger episode budget (120,000 vs. 500). A fair comparison requires matched interaction budgets and multi-seed averages.

## Next Steps

**Immediate improvements (Week 7):**

1. **Representation upgrade.** Replace the current “per-letter tried/status” embedding with a position-aware constraint representation (greens by position, yellows by excluded positions, grays by exclusion), and include guess count/remaining turns explicitly.
2. **Fair comparison.** Re-run ES vs. PPO with matched total episode budgets and 5–10 random seeds; report mean ± std for success and reward.
3. **Hyperparameter sweeps.** For ES: sweep σ and α (Wordle appears sensitive). For PPO: sweep entropy coefficient / annealing schedules to improve exploration.

**Technical improvements:**

- Add mirrored sampling (+ε / −ε) for ES variance reduction.
- Expand vocabulary and/or curriculum learning (start small vocab, then grow).
- Explore LoRA (low-rank adaptation) to fine-tune a small language-model-based guesser with minimal additional parameters, as a scalable alternative to the fixed-vocabulary discrete policy when we want to move toward more realistic Wordle action spaces.

## References

1. Salimans, T., Ho, J., Chen, X., Sidor, S., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. *arXiv:1703.03864*.
2. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

