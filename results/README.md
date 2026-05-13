# results/

CSV exports from notebooks. Tracked in git so the report tables and adaptation
analysis can be regenerated without rerunning training.

| File | Source notebook | Description |
|------|-----------------|-------------|
| `es_lora_adaptation_deltas.csv`  | `notebooks/week14_wordle_es_lora_run.ipynb` | Per-(seed, perturbation) full-vs-LoRA delta rows used for the LoRA-adaptation comparison. |
| `es_lora_adaptation_runs.csv`    | `notebooks/week14_wordle_es_lora_run.ipynb` | Per-run flat metrics (one row per `(seed, perturb_std, method)`). |
| `es_lora_adaptation_summary.csv` | `notebooks/week14_wordle_es_lora_run.ipynb` | Per-(perturb_std, method) aggregates: final eval success, AUC, time-to-0.8, etc. |

These files were previously stored under `notebooks/results/`; they were moved
here to mirror the top-level `data/`, `models/`, `figures/`, and `runs/` layout.
