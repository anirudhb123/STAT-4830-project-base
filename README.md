# Repo Quickstart (uv)

This repository is set up to run on macOS, Linux, and Windows using `uv`.

## macOS/Linux quickstart

From the repository root:

```bash
bash script/install.sh
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the script:

```bash
python script/gd_1d_torch.py
```

Run tests:

```bash
pytest
```

## Windows PowerShell quickstart

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\script\install.ps1
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the script:

```powershell
python .\script\gd_1d_torch.py
```

Run tests:

```powershell
pytest
```

## Python file guide

Top-level scripts:

- `quickstart_wordle.py`: Quick demo of the Wordle environment + discrete policy (mock mode); runs a short interaction and a full episode without training.
- `run_wordle_comparison.py`: Runs an end-to-end ES vs PPO comparison on Wordle, saves plots to `figures/` and trained weights to `models/`.

`src/` modules:

- `src/__init__.py`: Package exports for the ES/GridWorld code (re-exports core classes/functions from `model.py` and `utils.py`).
- `src/model.py`: Core GridWorld environments plus generic PyTorch policy/value networks (including optional LoRA adapters).
- `src/utils.py`: ES utilities (noise sampling, score functions, gradient estimation), evaluation helpers, plotting, and comparison-table/statistics helpers.
- `src/ppo_training.py`: PPO training loop utilities (rollout buffer, GAE, policy evaluation) with Wordle-specific training support.
- `src/wordle_env.py`: Wordle environment wrapper (Prime Intellect `verifiers` if installed, otherwise mock fallback) + state embedding + word vocabulary/action mapping.
- `src/wordle_policy.py`: Wordle discrete policy/value networks; produces Wordle guesses (optionally XML-formatted for Prime Intellect) from state embeddings.
- `src/wordle_es.py`: Evolution Strategies training implementation adapted for the Wordle wrapper/policy.
- `src/train_wordle.py`: CLI entrypoint to train a Wordle agent with PPO; handles args, evaluation, and saving model checkpoints.

## Notes

- Install scripts detect whether `uv` is already installed.
- If `uv` is missing, scripts install it using Astral's official installer.
- Dependencies are installed from `requirements.txt` into `.venv` in the repo root.
