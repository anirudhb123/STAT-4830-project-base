# Week 12 Self-Critique

**Scope of this critique.** Week 12 ran two distinct Gemma 3 1B IT experiments: (A) `notebooks/week12_implementation.ipynb` — Gemma on the **8-word mock vocabulary** with **no LoRA** (head-only ES), which **worked** (~83% greedy success at iter 9, `ES_win` from 68.8% to 93.8%); and (B) `notebooks/week12_implementation_LoRARun.ipynb` — Gemma on the **full Wordle vocabulary** with a **LoRA rank sweep** (r=4, 16, 32), which **failed** (~0% greedy success across all ranks, `ES_win` essentially 0%). The DistilGPT-2 smoke profile in either notebook is plumbing-only and not a research baseline. The critique below focuses on what made (A) succeed but (B) collapse.

## ORIENT

### Strengths

- **Notebook A is a real positive result.** Gemma + head-only ES on the 8-word mock vocabulary reaches ~83% greedy success at iter 9 with rising `ES_win` (68.8% -> 93.8%). The Gemma backbone, chat-template prompts, warm-start, and ES loop all behave as intended end-to-end on this task.

- **Run profiles directly address Week 10 workflow pain.** The **`RUN_PROFILE`** switch (`smoke` vs `gemma_full`) separates "does the notebook run end-to-end?" from "are we doing a serious experiment?" without maintaining a dedicated debug notebook. DistilGPT-2 smoke is plumbing-only; the real Gemma runs are in the two `gemma_full` configurations.

- **Two-notebook split lets us compare configurations cleanly.** Keeping the head-only mock run (`week12_implementation.ipynb`) separate from the full-vocab LoRA sweep (`week12_implementation_LoRARun.ipynb`) makes the contrast between the working and failing setups easy to point at.

- **Gemma IT + LoRA path is runnable end-to-end.** `USE_CHAT_TEMPLATE`, `CHAT_GENERATION_PROMPT`, and PEFT LoRA adapters are wired and executable, so the failure of the LoRA sweep is about training behavior and experiment design, not missing integration.

- **Checkpoint isolation by profile.** Saving to **`models/wordle_gemma_es_head.<RUN_PROFILE>.pt`** prevents a smoke run from clobbering a long Gemma checkpoint — a small change that avoids silent data loss.


### Areas for Improvement

- **The full-vocabulary LoRA sweep did not work.** Under Notebook B's setting (`MOCK_ENV=False`, full vocab, Gemma + LoRA, ranks 4/16/32), greedy `Success` stayed at 0% across all ranks and `ES_win` stayed near 0%. The "best rank" reported by the sweep (r=4) is best only by tie-breaking among zero-success runs — it should not be read as a real LoRA recommendation.

- **Train/test split isolation is weak.** The process does not enforce a persistent, auditable disjoint train/eval word split for the full-vocabulary run. Without strict split isolation, future improvements on Notebook B's regime will be hard to interpret as true generalization instead of overlap or distribution leakage.

- **Curriculum learning is likely necessary between Notebook A and Notebook B.** We jumped directly from an 8-word task (where Gemma + head-only ES works) to the full ~2k-word vocabulary with LoRA enabled (where nothing learns). A staged curriculum (small vocab -> medium vocab -> full vocab), with weights carried forward, would likely produce a smoother optimization path and better signal for ES instead of collapsing two scaling decisions into one experiment.

- **Notebook B confounds vocabulary scaling and LoRA.** Going from Notebook A to Notebook B simultaneously changed the vocabulary (8 -> ~2k actions) and enabled LoRA. The failure cannot be attributed to either alone without further runs.

- **DistilGPT-2 smoke profile has low statistical power and is not a baseline.** `N_POP=4`, `N_ITERATIONS=2`, and very short warm-start make `RUN_PROFILE="smoke"` useful only for plumbing checks. It should not be cited as evidence about ES dynamics or model quality.

### Critical Risks / Assumptions

- **Risk:** Without strict train/test split isolation on the full-vocabulary task, we cannot make strong generalization claims even if Notebook B's headline metrics rise in future runs.

- **Risk:** Going straight to full-vocabulary optimization with LoRA enabled can hide whether failure comes from vocabulary size, LoRA capacity, reward shaping, or optimization horizon; the current Notebook A vs Notebook B contrast collapses several of these factors into one experiment.

- **Assumption:** The 8-word mock vocabulary success in Notebook A transfers in any informative way to the full vocabulary. Mock-task win rate may overstate how close the policy is to playing real Wordle.

- **Assumption:** Chat-templated prompting for Gemma preserves the same effective signal as plain prompts. If template overhead or truncation changes usable context at full vocabulary, training quality can degrade independently of ES.

## DECIDE

### Concrete Next Actions

1. **Create strict split artifacts first.** Generate and save fixed disjoint train/eval word lists for the full Wordle vocabulary, reuse them across seeds, and report metrics only on the held-out split.

2. **Introduce explicit curriculum stages between Notebook A and Notebook B.** Run a sequence such as 8 -> 64 -> 256 -> full vocabulary, carrying weights forward and logging where performance collapses. This bridges the gap between the working mock-vocab regime and the failing full-vocab regime.

3. **Decouple LoRA from vocabulary scaling.** Add a Gemma + LoRA run on the 8-word mock task and a Gemma head-only run on the full vocabulary so the cause of Notebook B's failure (vocabulary size vs LoRA vs both) can be isolated.

4. **Run warm-start vs ES ablations under the same split.** Evaluate after warm-start and after ES on identical held-out episodes to determine whether ES adds signal in each regime.

## ACT

### Resource Needs

- **GPU with sufficient VRAM** for **`google/gemma-3-1b-it`** (bf16/fp16) plus batch-size-one forward passes over long prompts; CPU **`gemma_full`** is likely only for debugging single steps.  
- **Disk / cache** for Gemma weights; first-time download can be large.  
- Optional: **`peft`** if LoRA experiments are prioritized.  
- **Wall-clock budget:** `gemma_full` ES is roughly “Week 10 cost × (Gemma forward pass factor)”; parallelizing population eval across processes remains an undeployed speedup for both weeks.
