# STAT 4830 Final Presentation — Google Slides Copy-Paste Pack

How to use this doc:

1. Open Google Slides → start from draft 5 (or a blank deck).
2. For each slide below, create a slide in Google Slides, paste the **TITLE** into the title box and the **BODY** into the main text box, one bullet per line.
3. Paste the **SPEAKER NOTES** into the speaker-notes pane at the bottom of each slide (`View → Show speaker notes`).
4. Build the visual described under **VISUAL** using the insert-image/insert-shape tools, or screenshot/export an existing plot from your notebooks.
5. Assign each slide to the person listed under **Speaker** and practice together end-to-end, **timed**.
6. Replace `[NAME 1]` … `[NAME 4]` with your teammates' actual names everywhere they appear.

Target total: **10:00** hard cut. Budget below sums to ~9:40, giving you a 20-second buffer.

The 30-second "what did you find most useful or surprising" round is separate from the 10 minutes. Draft answers at the bottom of this file.

---

## SLIDE 1 — Title
**Speaker:** [NAME 1] **| Target: 15 sec**

### TITLE
Evaluating Gradient-Free Optimization in Reinforcement Learning
Evolution Strategies for Sparse-Reward Tasks

### BODY
- [NAME 1], [NAME 2], [NAME 3], [NAME 4]
- STAT 4830 · Spring 2026

### SPEAKER NOTES
Hi, we're [NAMES]. Our project asks whether Evolution Strategies — a
gradient-free optimizer — can train a language-model policy to solve
a task where rewards are almost always zero. We'll walk you through
the problem, the method, and what actually happened when we scaled it
up to Wordle.

### VISUAL
Clean title slide. One cover image (optional): a filled-in Wordle grid
with a single green row at the bottom.

---

## SLIDE 2 — The Problem: Wordle + Sparse Rewards
**Speaker:** [NAME 1] **| Target: 60 sec**

### TITLE
The Problem: Wordle is a Sparse-Reward RL Task

### BODY
- Goal: train a policy to guess a hidden 5-letter word in ≤ 6 turns
- Reward: **+1 only if you guess the word**, 0 on every other turn
- Action space: any valid 5-letter English word (~2,000+ choices)
- Feedback per letter: green / yellow / gray (standard Wordle rules)
- Why this is hard: the learning signal is **0 for the first 5 turns of almost every game**

### SPEAKER NOTES
Wordle feels simple to a human, but it's a worst-case setup for
reinforcement learning. The reward is almost always zero — you get a
+1 only on the turn you guess correctly, and you fail most early
episodes, so your policy spends most of training seeing no reward
signal at all. Classical policy-gradient methods like REINFORCE or
PPO rely on informative rewards; when rewards are this sparse, their
gradient estimates are dominated by variance. That's the gap we
wanted to attack.

### VISUAL
A real Wordle board (5 rows of mostly gray tiles, one green at the
bottom). Annotate "reward = 0" on the gray rows and "reward = +1" on
the green row. A screenshot of any Wordle game works.

---

## SLIDE 3 — Evolution Strategies in One Picture
**Speaker:** [NAME 1] **| Target: 75 sec**

### TITLE
Evolution Strategies: Gradient-Free, Population-Based Search

### BODY
- Perturb the parameters N times: θ + σ · εᵢ,  εᵢ ~ N(0, I)
- Run each perturbed policy in the environment → fitness Fᵢ
- Move θ toward the direction where the high-fitness perturbations lived
- Update rule:  θ ← θ + α · (1 / (N σ)) · Σᵢ  Fᵢ · εᵢ
- **No backprop through the environment. Only rollouts.**

### SPEAKER NOTES
Here's ES in one slide. Instead of computing a gradient with
backprop, we sample N random perturbations of the parameters, run
each one through the environment, collect a fitness — just the total
reward of the rollout — and take a weighted average of those noise
vectors using the fitnesses as weights. That weighted average is our
"gradient estimate." It's gradient-free in the sense that the
environment can be a black box: no differentiable simulator, no
reward shaping required. Salimans et al. 2017 showed this is a
scalable alternative to RL.

### VISUAL
Diagram: center dot = θ. Draw 6–8 arrows outward (the perturbations),
color-coded by fitness (green = high, red = low). Draw one thicker
arrow showing the weighted-average direction. This maps 1:1 to the
formula.

---

## SLIDE 4 — Why ES? Parallelism + Sparse-Reward Robustness
**Speaker:** [NAME 1] **| Target: 45 sec**

### TITLE
Why ES for Sparse Rewards?

### BODY
- Only needs **episode-level fitness** — works fine when rewards are 0 until the last turn
- **Trivially parallel:** each worker sends one scalar + one seed, not a full gradient
- Bypasses high-variance REINFORCE-style estimators in the sparse regime
- Compatible with frozen / non-differentiable black-box policies

### SPEAKER NOTES
Two properties matter here. First, ES only cares about total
episode reward, so sparse rewards are not a special case — they're
the default. Second, ES is embarrassingly parallel: a worker only
needs to return a fitness and a random seed, not a multi-hundred-MB
gradient, so it scales to thousands of workers with almost no
communication overhead. That's the original motivation from
Salimans et al.

### VISUAL
Small two-column comparison. Left: "Policy gradient — per-worker
message: full gradient (~100 MB)." Right: "ES — per-worker message:
scalar + seed (~8 bytes)." One-line takeaway: "ES is bandwidth-free."

---

## SLIDE 5 — Proof of Concept: GridWorld
**Speaker:** [NAME 2] **| Target: 45 sec**

### TITLE
Proof of Concept: ES Solves Sparse-Reward GridWorld

### BODY
- 2-layer MLP, 8,580 parameters
- Sparse reward: +1 for goal, -0.1 for obstacle, 0 everywhere else
- ES learns to navigate to the goal from the bottom-left start
- Confirms the method is working before we scale it up

### SPEAKER NOTES
We first validated the pipeline on a small GridWorld. The policy is
a tiny MLP, the reward is genuinely sparse, and ES learns a working
navigation policy. This is our sanity check — if the method couldn't
solve this, nothing that follows would mean anything. It works.

### VISUAL
A GridWorld screenshot: agent start cell (bottom-left), green goal,
red obstacles, and a learned trajectory. If you have a reward curve
from `src/es_gridworld.py`, put it next to the grid. Caption: "ES
learns a goal-reaching policy on a sparse-reward grid."

---

## SLIDE 6 — Scaling Up: Gemma 3 1B + Linear Head
**Speaker:** [NAME 2] **| Target: 60 sec**

### TITLE
Policy: Gemma 3 1B + Trainable Linear Head

### BODY
- Frozen backbone: **Gemma 3 1B Instruct**
- Trainable: a **linear head** over the backbone's hidden state
- Head outputs a distribution over **a fixed list of valid 5-letter words**
- Guesses come from argmax over this list ⇒ guesses are **always real words**
- Prompt encodes: turn number, past guesses, per-letter feedback, constraints

### SPEAKER NOTES
Our Wordle policy is built around a pretrained language model.
We use Gemma 3 1B Instruct as the backbone and bolt a linear head
on top that scores each candidate word. Both the prompt and the
output space are designed so the policy can only pick real
5-letter English words — we never generate free text that then has
to be validated. This is important: every guess is a valid Wordle
word.

### VISUAL
Architecture diagram:
  [Prompt] → [Frozen Gemma 3 1B] → [Hidden state] → [Linear head] → [Softmax over ~N valid 5-letter words]
Highlight the linear head in a different color and label it
"trainable" — and label the Gemma block "frozen."

---

## SLIDE 7 — Supervised Warm-Start
**Speaker:** [NAME 2] **| Target: 60 sec**

### TITLE
Warm-Start: Give ES a Starting Point, Not a Blank Policy

### BODY
- Sample a secret word → play random (but **feedback-consistent**) guesses for 0..5 turns
- Train the head with cross-entropy to predict the secret from that mid-game state
- Teaches the policy "given this set of clues, the answer is probably X"
- ES then **optimizes on top** of this reasonable initialization

### SPEAKER NOTES
ES on a randomly-initialized policy almost never stumbles into a
reward — it would spend most of training in the zero-reward region.
So we warm-start with supervised learning. We generate mock game
states by playing random guesses that are consistent with the
feedback so far, then ask the model to predict the secret. This
gives us a policy that already plays reasonable Wordle before ES
starts. ES's job is then to refine that policy, not to build it from
scratch.

### VISUAL
Two-panel diagram. Panel 1: "Before warm-start" — random scoring
distribution. Panel 2: "After warm-start" — distribution peaked on
the secret given the clues. Arrow labeled "cross-entropy."

---

## SLIDE 8 — LoRA + Curriculum Learning
**Speaker:** [NAME 3] **| Target: 75 sec**

### TITLE
LoRA + Curriculum: How We Actually Scaled It

### BODY
- **LoRA:** freeze Gemma's weights, train low-rank updates (W + BA)
  - Rank 8: ~17M trainable params instead of ~1B
  - Lower-variance ES gradients (gradient variance ∝ dimensionality)
- **Curriculum:** grow the secret pool stage by stage
  - 16 → 32 → 64 → 96 → 128 → 256 → 512 → 1024 words
  - Carry weights forward; re-warm-start when the pool grows
  - Same action head throughout — only the "target set" gets harder

### SPEAKER NOTES
Two tricks made scaling to the real Wordle vocabulary tractable.
First, LoRA: we freeze the entire Gemma backbone and only train a
low-rank update. That cuts trainable parameters by about two orders
of magnitude, which matters for ES because variance grows with
dimensionality. Second, curriculum learning: we don't throw the full
word list at the policy on day one. We start with 16 secrets, get
that working, then grow to 32, 64, 96, and so on up to 1024. At each
stage we re-warm-start on the new pool, then let ES fine-tune. This
is directly inspired by curriculum learning from classical RL: train
on easy before hard.

### VISUAL
Two half-slide diagrams side by side.
Left — LoRA: a matrix W with "frozen" padlock, plus a small B·A
low-rank insert labeled "trainable."
Right — Curriculum: a horizontal bar split into 8 segments labeled
16, 32, 64, 96, 128, 256, 512, 1024, with an arrow showing weights
carried forward.

---

## SLIDE 9 — Result A: ES Produces Real Signal (Probe)
**Speaker:** [NAME 3] **| Target: 60 sec**

### TITLE
Result A: ES Alone Can Solve a Tiny Wordle

### BODY
- Isolated probe: 4-word secret pool, **no warm-start**, 20 ES iterations
- Starting success: **0%** (random-init LoRA head)
- Peak success: **86%**, final success: **66%**
- Proves ES is contributing real optimization — not just riding warm-start

### SPEAKER NOTES
Before claiming that ES helps on the full task, we ran a clean
probe. Tiny secret pool of 4 words, no warm-start, pure ES from a
random linear head. Success rate climbs from 0% to a peak of 86% in
twenty iterations. That's the experiment we point to when someone
asks "are you sure ES is doing anything, or is it all warm-start?"
Pure ES solves the problem. The signal is real.

### VISUAL
Line plot: x = ES iteration (0–19), y = greedy success rate (%).
Mark 0% at iter 0, 86% at the peak. Axes labeled, one-sentence
caption: "Pure ES, no warm-start: 0% → 86% on a 4-word pool."

---

## SLIDE 10 — Result B: Curriculum Run, 16 → 256 Words
**Speaker:** [NAME 4] **| Target: 90 sec**

### TITLE
Result B: Curriculum Scaling, Stage by Stage

### BODY
- **Stage 1** (16 words): post-warm-start **42%** → ES **46%**
- **Stage 2** (32 words): post-warm-start **66%** → ES **66%**
- **Stage 3** (64 words): post-warm-start **74%** → ES **84%**  ← +10 pp
- **Stage 4** (96 words): post-warm-start **88%** → ES **90%**  ← peak
- **Stage 5** (128 words): no warm-start → **72%**  ← drop
- **Stage 6** (256 words): no warm-start → **28%**  ← breaks

### SPEAKER NOTES
Here are the real numbers from the full curriculum run on Gemma +
LoRA rank 8. The first four stages work well — we hit 90% at 96
words, with ES adding a clear +10-point bump at stage 3 over what
warm-start alone achieves. Then something interesting happens:
stages 5 and 6, where we grow from 96 to 128 and 256 words, the
numbers collapse. We'll explain why on the next slide. The important
thing is that this drop is not random — we diagnosed it, and it
tells us something real about the interaction between warm-start
and curriculum learning.

### VISUAL
Table exactly as in the body, OR — preferred — a bar chart:
x = stage vocab size (16, 32, 64, 96, 128, 256); y = greedy
success %. Two bars per stage: "post-warm-start" and "best ES."
Color the 128/256 bars red to telegraph the failure. One-sentence
caption: "Curriculum works up to 96 words; collapses at ≥128."

---

## SLIDE 11 — Honest Failures: What Broke and Why
**Speaker:** [NAME 4] **| Target: 60 sec**

### TITLE
What Broke — and What We Learned

### BODY
- **Stage 5+ collapse:** warm-start was auto-suppressed once stage 4 hit ≥85%
- **But the action head's "new" rows (words 97–256) were never trained**
- **Result:** mechanical accuracy drop when the secret pool grew past the warm-start's coverage
- **Fix:** re-enable warm-start whenever the secret pool changes, even after the ceiling
- **Lesson:** ceilings designed for "don't overtrain" can starve later stages of signal

### SPEAKER NOTES
Here's the honest part. We had a warm-start "ceiling" — the idea was
that if the policy is already at 85% after warm-start, further
supervised training just saturates and ES has no room to improve.
That ceiling worked for stage 4. But once it fired, the code kept
warm-start off even when stage 5 grew the secret pool from 96 to 128
words. The 32 new words had never been warm-started, so the action
head couldn't score them, and accuracy mechanically dropped. We
caught this, fixed it, and the takeaway is general: global
optimization-schedule rules interact badly with local schedule
changes like curriculum transitions. You have to re-enable training
whenever the task changes.

### VISUAL
Before/after schematic. Before: green bars stages 1–4, red bars
stages 5–6 marked "WS ceiling tripped." After: green bars all the
way across, small annotation "WS re-enabled on pool growth." Keep it
simple — this is a diagnostic slide, not a headline result.

---

## SLIDE 12 — Takeaways + Next Steps
**Speaker:** [NAME 4] **| Target: 30 sec**

### TITLE
Takeaways + Next Steps

### BODY
- ES + warm-start + LoRA is a workable recipe for sparse-reward LM policies
- Curriculum learning scales ES well — up to the point where the schedule breaks
- **Next:** quantify generalization gap (train vs unseen secrets)
- **Next:** apples-to-apples REINFORCE vs ES comparison under matched warm-start
- **Next:** push past 1024 words with better mini-batch ES under common random numbers

### SPEAKER NOTES
Three takeaways. One: ES plus warm-start plus LoRA is a viable
recipe for training language-model policies in the sparse-reward
regime — we got real accuracy on real Wordle. Two: curriculum
learning is necessary for scale, and its interaction with
optimization schedules is subtle. Three: the honest open questions
are generalization to unseen secrets, and whether ES actually beats
a properly-tuned REINFORCE baseline when both start from the same
warm-start. That's where we'd go next with more time.

### VISUAL
Plain bullets. Optional small icon per bullet.

---

## BACKUP SLIDES — Keep these AFTER slide 12 in case of Q&A

### BACKUP A — Neural Thickets
TITLE: Why Does ES Work on an LLM at All?
BODY:
- Large pretrained models have **dense regions of "good" parameters**
- Random perturbations land inside these regions with non-zero probability
- ES doesn't need to find a single optimum; it needs to drift inside a thicket
- This is why a gradient-free method can optimize a 1B-parameter model

### BACKUP B — Prompt / Constraint Encoding
TITLE: How Feedback Becomes a Prompt
BODY:
- After each guess, compute per-letter feedback (green / yellow / gray)
- Accumulate constraints: known letters, must-include, must-not-at-position, min/max counts
- Render constraints as plain-text hints inside the prompt (`wordle_hints.py`)
- Correctly handles duplicate letters (we fixed a real bug here)

### BACKUP C — REINFORCE Baseline (work in progress)
TITLE: REINFORCE vs ES — Current Status
BODY:
- REINFORCE baseline is set up in `notebooks/week12_reinforce_gemma.ipynb`
- Just aligned its warm-start config to match the ES run (fair comparison)
- Results not yet rerun on the VM — reported honestly as "open"

---

## 30-SECOND CLOSING ANSWERS (required, one per person, separate from the 10:00)

Each person writes their own. Below are **starter drafts** — replace with what you actually found surprising. Aim for ~60 words so it lands in 25 seconds.

### [NAME 1] — draft
"The most surprising thing was how much of reinforcement learning is
really about **variance reduction**, not reward design. Baselines,
rank-based fitness, common random numbers — all of these exist
because the underlying signal is tiny and the noise is huge. I came
in thinking RL was about clever rewards; I left thinking it's about
making your estimator cleaner."

### [NAME 2] — draft
"I found it surprising that a **gradient-free** method could
optimize a **billion-parameter** neural net. The neural-thickets
idea — that good solutions in large overparameterized models live in
dense regions, not isolated points — reframed overparameterization
from a bug into a feature. It's why simple random perturbation can
work where you'd naively expect it to fail."

### [NAME 3] — draft
"The most useful thing for me was **warm-starting**. In theory we
want end-to-end RL; in practice, initialization dominates. Seeing
warm-start carry 70% of the final performance and ES refine the
last few points changed how I think about the split between
supervised signal and exploration signal in any RL system."

### [NAME 4] — draft
"What surprised me most was how **easy-to-write bugs can look like
algorithmic failures**. Our stage-5 collapse looked like curriculum
learning breaking — it was actually an interaction between our
warm-start ceiling and the schedule. A lot of optimization research
is just correctly attributing bad results to the right cause."

---

## Q&A CHEAT SHEET — expected questions and one-line answers

**"Why ES instead of PPO / REINFORCE?"**
→ Sparse rewards make policy-gradient variance enormous; ES only uses
episode-level fitness and parallelizes trivially. We're also running
a REINFORCE baseline under matched warm-start; that comparison is in
progress.

**"Does the policy ever guess non-words?"**
→ No. The action head scores a fixed list of real 5-letter English
words; argmax picks one of them. Every guess is valid by construction.

**"What baseline did you compare against?"**
→ Warm-start-only success is our primary baseline per stage — the
"ES adds +10 pp at stage 3" number is against exactly this baseline.
Random-word baseline is ~0% on full vocab.

**"How long did training take, on what hardware?"**
→ One A100 on a cloud VM, roughly [FILL IN]. Individual stages took
minutes to tens of minutes; full 8-stage curriculum is several hours.

**"Why Gemma 3 1B specifically?"**
→ Large enough to carry useful language priors, small enough to fit
on a single GPU with LoRA and still run N=64 perturbations per ES
iteration. DistilGPT-2 was our early prototype; Gemma replaced it
once the pipeline was stable.

**"Did you measure generalization to unseen secrets?"**
→ Partially — we report a fresh-episode eval per stage (sampled from
the full pool, not just train secrets). A formal train/held-out
split across the full vocabulary is explicitly open future work.

**"Why did stage 5+ collapse?"**
→ Our warm-start-ceiling heuristic suppressed warm-start once stage
4 saturated, but the secret pool kept growing. New words never got
warm-started, so their head rows were untrained. We diagnosed and
fixed this; re-running is next.

**"Is ES competitive with gradient-based RL, or just interesting?"**
→ Open question in our project. On Wordle with sparse rewards, ES
works where we couldn't get REINFORCE + LoRA to work at the same
compute budget. The properly controlled comparison is still in progress.

---

## FINAL CHECKLIST — before you present

- [ ] Team names on title slide (slide 1)
- [ ] Speaker assignments filled in for all 12 slides
- [ ] Real plot from `week12_implementation_LoRARun.ipynb` on slide 10 (bar chart)
- [ ] Every plot has labeled axes, a legend, and a one-sentence caption
- [ ] Export the whole deck to PDF as a **backup** (`File → Download → PDF Document`)
- [ ] 30-second closing answers written per person
- [ ] Q&A cheat sheet reviewed by all four people
- [ ] Full dry run #1 tonight, **timed** — if over 10:00, cut slides
- [ ] Full dry run #2 tonight after cuts, **timed** — must hit ≤ 10:00
- [ ] Morning dry run #3 tomorrow, **timed**
- [ ] Bring a laptop charger and a USB-C adapter
