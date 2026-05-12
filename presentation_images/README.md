# Presentation image index

All images extracted from notebooks + git history. Use these as drop-in
visuals for the Google Slides deck (see `../presentation_slides.md`).

## How to use in Google Slides

1. Open Finder → navigate to `presentation_images/` in the repo.
2. Double-click any `.png` to preview it in macOS Preview.
3. Drag the file straight onto a Google Slides slide (or use
   `Insert → Image → Upload from computer`).
4. Resize with the corner handles (hold `Shift` to preserve aspect ratio).
5. Add a one-sentence caption underneath using a text box.

## Best match per slide

| Slide (from `presentation_slides.md`)       | Suggested image                                                         | Why |
|---------------------------------------------|-------------------------------------------------------------------------|-----|
| 5 — GridWorld proof of concept              | `week6_implementation/cell20_img01.png`                                 | 6-panel ES training curves on sparse-reward GridWorld (reward / success / turns). Clean, matches the slide narrative.|
| 5 — GridWorld PoC (alternative)             | `week9_implementation/cell08_img01.png` **or** `week7_implementation/cell08_img01.png` | GridWorld layout rendering. Pair it next to the curve for a 2-up slide (what the env looks like + how ES learned it). |
| 6 — Policy architecture                     | *none available — build diagram in Slides*                              | No saved figure; draw the Gemma→head pipeline with Slides shapes. |
| 9 — Result A: ES signal probe (0%→86%)      | *no saved plot — generate from text output*                             | Probe results are printed as text in the LoRA notebook. Ask to generate a matplotlib chart. |
| 10 — Result B: curriculum stage-wise        | *no saved plot — generate from text output*                             | The LoRA curriculum run was interrupted before plot cells ran. Ask to generate a matplotlib chart from the real stage numbers. |
| 11 — Honest failures                        | `_history_lora/2026-04-18_f3383b8_cell??_img01.png` **or similar**      | One of the early failed curriculum runs — useful as "before" when paired with new results. Open each and pick whichever looks best. |
| Backup — Thickets / RandOpt vs ES           | `thickets_comp_week_12/cell18_img01.png`, `cell19_img02.png`            | RandOpt vs ES reward and success curves. Directly supports the "ES is non-trivially better than random search" claim. |
| Backup — ES internal diagnostics            | `thickets_comp_week_12/cell20_img03.png`                                | ES train fitness + population win rate over iterations. |
| Backup — Final comparison                   | `thickets_comp_week_12/cell21_img04.png` (success)                      | Bar chart: Initial vs RandOpt vs ES success rate. Punchy 1-slide story. |
| Q&A — Wordle ES diagnostics (Gemma, 8-word) | `week12_implementation/cell12_img01.png`                                | 6-panel diagnostic from the working 8-word Gemma mock. Useful if someone asks "what does training look like internally?". |
| Q&A — REINFORCE baseline                    | `week12_reinforce_gemma/cell13_img01.png`                               | The current REINFORCE curve. Useful if asked "have you compared against REINFORCE?" — caveat that warm-start was not matched at the time. |

## Full inventory

### Current-notebook images (25)

| Path                                                              | Description |
|-------------------------------------------------------------------|-------------|
| `randopt/cell19_img01.png`                                        | Random-opt 6-panel baseline diagnostic (Wordle) |
| `thickets_comp_week_12/cell18_img01.png`                          | RandOpt vs ES: reward during optimization |
| `thickets_comp_week_12/cell19_img02.png`                          | RandOpt vs ES: success rate |
| `thickets_comp_week_12/cell20_img03.png`                          | ES internal signals: fitness & pop win rate |
| `thickets_comp_week_12/cell21_img04.png`                          | Final success-rate bars: Init / RandOpt / ES |
| `thickets_comp_week_12/cell21_img05.png`                          | Final reward bars |
| `thickets_comp_week_12/cell21_img06.png`                          | Final avg-turns bars |
| `week10_implementation/cell12_img01.png`                          | 6-panel Wordle ES diagnostic (DistilGPT-2 path) |
| `week12_implementation/cell12_img01.png`                          | 6-panel Wordle ES diagnostic (Gemma + 8-word mock) |
| `week12_reinforce_gemma/cell13_img01.png`                         | REINFORCE baseline training curves |
| `week6_implementation/cell20_img01.png`                           | GridWorld ES: eval reward / success / turns (6-panel) |
| `week6_implementation/cell26_img02.png`                           | ES vs PPO bar comparison (GridWorld) |
| `week6_implementation/cell29_img03.png`                           | ES vs PPO overlaid learning curves |
| `week7_implementation/cell08_img01.png`                           | GridWorld base-layout visualization |
| `week7_implementation/cell16_img02-06.png`                        | GridWorld method comparison (full vs LoRA, multi-perturb) |
| `week9_implementation/cell08_img01.png`                           | GridWorld base-layout visualization (later revision) |
| `week9_implementation/cell16_img02-06.png`                        | GridWorld method comparison (full vs LoRA, multi-perturb) |

### Recovered-from-git-history LoRA images (6)

These are plot frames from earlier commits of `week12_implementation_LoRARun.ipynb`.
Most are from **failed / incomplete curriculum runs** — useful as "before"
images on the honest-failures slide, not as headline results.

| Path                                                              | Commit notes |
|-------------------------------------------------------------------|--------------|
| `_history_lora/2026-04-18_*.png`                                  | Early curriculum attempts — "vocab size outgrew model" |
| `_history_lora/2026-04-19_*.png`                                  | Intermediate failures — "bad but perhaps promising" |

Open each in Preview and pick whichever is visually cleanest for your
"what broke and why" slide.
