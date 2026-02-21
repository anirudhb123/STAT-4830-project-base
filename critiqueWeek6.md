# Week 6 Self-Critique

## ORIENT

### Strengths
- Clear evolution in problem scope: We moved from GridWorld to Wordle, which is a harder discrete decision problem with meaningful exploration/credit-assignment challenges and a much larger (even if still truncated) action space.

- Working end-to-end Wordle pipeline: The repo now has an environment wrapper (`src/wordle_env.py`), a discrete vocabulary policy + value net (`src/wordle_policy.py`), ES training (`src/wordle_es.py`), and PPO training utilities (`src/ppo_training.py`), and the Week 6 notebook runs both methods and produces saved figures/models.

- Transparent reporting of results: `reportWeek6.md` separates (a) post-training evaluation (50 episodes) from (b) training-time logged metrics, and explicitly calls out the large ES vs PPO episode-budget mismatch.

### Areas for Improvement
- **Representation is too weak for Wordle:** The current 64-d embedding is mostly “letter tried/status” and is not position-aware (greens/yellows per slot), which is critical in Wordle. This likely caps achievable success even with good optimization.

- **Evaluation protocol is inconsistent / high-variance:** The training summary reports “final success” based on periodic training eval, while the post-training 50-episode evaluation can differ materially (e.g., PPO training-time success vs 0% in the 50-episode eval). This makes it hard to compare methods reliably.

- **Unfair compute budget comparison:** ES uses 120,000 episodes vs PPO’s 500 in the current run. Even if ES is “better” by success rate, we cannot claim sample-efficiency or overall superiority without matched budgets (or at least curves vs total episodes).

- **Action space is artificially constrained:** Using 156 target words simplifies training but may understate Wordle difficulty and limit generalization. It’s unclear how policies perform as vocabulary scales toward realistic sizes.

### Critical Risks/Assumptions
We are assuming the current reward shaping and environment simulation are sufficiently aligned with the intended Prime Intellect verifier task. If the wrapper reward/feedback differs from the real evaluation rubric or interaction protocol, we risk optimizing the wrong objective. Also, at low success rates (0–15%), single-run results can vary substantially across seeds, so conclusions without multi-seed evaluation are fragile.

## DECIDE

### Concrete Next Actions
1. **Upgrade the state embedding (highest leverage):** Add position-specific constraints (known greens by position, yellows with excluded positions, grays as exclusions), plus remaining-turn count. This should be done in `WordleEnvironmentWrapper.get_state_embedding` with a fixed-size encoding.

2. **Standardize evaluation and logging:** Define a single evaluation function and always report:
   - success rate over a fixed N (e.g., N=200)
   - avg reward and avg turns
   - confidence intervals over seeds
   Then update the notebook to compute *both* methods’ metrics the same way.

3. **Fair ES vs PPO comparison:** Re-run both methods under matched total episode budgets (e.g., 5k/10k/20k episodes), and plot success vs episodes. Include 5–10 seeds for each budget.

4. **Scale vocabulary gradually (curriculum):** Start with 100–200 words, then increase to larger sets, tracking how success changes. This tests whether our approach scales or only works in a toy action space.

## ACT

### Resource Needs
We can do multi-seed and budget-matched experiments on CPU, but we should parallelize runs (multi-processing or simple batch scripts) to keep iteration speed reasonable. If we expand vocabulary and model size significantly, a GPU would help, but it is not strictly required for the current network sizes.

