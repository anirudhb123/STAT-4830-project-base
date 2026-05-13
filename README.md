# Repo Quickstart (uv)

This repository is set up to run on macOS, Linux, and Windows using `uv`.

## STAT 4830 final project

- **Written report:** [`docs/reports/final_report.pdf`](docs/reports/final_report.pdf) 
- **Slides:** [`docs/presentation_slides.md`](docs/presentation_slides.md)
- **Main reproducible artifact:** [`notebooks/week16_wordle_es_qwen.ipynb`](notebooks/week16_wordle_es_qwen.ipynb)
- **Script for training from a GPU server (no notebook):** [`scripts/run_week16_es.py`](scripts/run_week16_es.py) — same ES loop as the notebook for long unattended runs.

### Executable demo

The graded demo path is **`notebooks/week16_wordle_es_qwen.ipynb`**: register the repo’s `.venv` as a Jupyter kernel (see [`AGENTS.md`](AGENTS.md)), open the notebook, use a **GPU** runtime (`torch.cuda.is_available()` should be `True`), and run cells in order through training and plots. To use **Google Colab**, upload the notebook or open it from a mounted GitHub copy, enable a GPU runtime (**Runtime → Change runtime type**), paste a Hugging Face **read** token when the notebook/environment needs Hub access (`hf auth login` in an initial `!pip` cell works on Colab), and rerun from the top. Full ES training is slow without a decent GPU because of repeated `model.generate` calls.

### Reproducing Week 16 results

1. Follow **macOS/Linux** or **Windows** quickstart below (`bash scripts/install.sh` / `install.ps1`, then activate `.venv`).
2. Install/upgrade tooling the notebook relies on if missing: **`jinja2>=3.1.0`** for Qwen chat templates (`python -m pip install -U "jinja2>=3.1.0"`). Optional: `peft`, `ipykernel`.
3. **Hugging Face:** complete the **[Hugging Face](#hugging-face-gated-models--login)** steps. For Week 16, accept access on the Hub for the checkpoints you actually load (typically **`PrimeIntellect/Qwen3-1.7B-Wordle-SFT`** and **`PrimeIntellect/Qwen3-1.7B-Wordle-RL`**, and any base routing checkpoint such as **`Qwen/Qwen3-1.7B`** if `from_pretrained` pulls it—use the exact IDs from notebook section 2 / the report). Downloads fail with **403** if the license step was skipped on the logged-in account.
4. Run **`notebooks/week16_wordle_es_qwen.ipynb`** from top to bottom in that environment.
5. **Optional CLI run on a GPU** (mirrors the notebook driver; edit `MODEL_ES_BASE`, `NORMALIZE_GRADIENT`, and related constants in [`scripts/run_week16_es.py`](scripts/run_week16_es.py) to match the run you want, as in [`docs/reports/week16.md`](docs/reports/week16.md)):

```bash
mkdir -p logs
python -u scripts/run_week16_es.py --artifacts-dir runs/week16_es/my_run 2>&1 | tee logs/week16_es.log
```

Committed example artifacts from the submission runs live under **`runs/week16_es/{sft_base,rl_base,rl_base_normed_gradients}/`**. Additional paths under `runs/week16_es/` may be intentionally gitignored via [`.gitignore`](.gitignore).

## macOS/Linux quickstart

From the repository root:

```bash
bash scripts/install.sh
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the script:

```bash
python scripts/gd_1d_torch.py
```

Run tests:

```bash
pytest
```

## Windows PowerShell quickstart

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run the script:

```powershell
python .\scripts\gd_1d_torch.py
```

Run tests:

```powershell
pytest
```

## Hugging Face (gated models + login)

Use this when a notebook or script loads weights from the Hub (e.g. **Gemma** in older weeks, **`Qwen/Qwen3-*`** and **`PrimeIntellect/Qwen3-1.7B-Wordle-{SFT,RL}`** in Week 16). You do **not** put a token in the repo or in a committed `.env` file.

### 1. Account and model access

1. Create a [Hugging Face](https://huggingface.co) account (or sign in).
2. For **gated** models, open that model’s page while logged in and click **Agree and access** (or equivalent). Examples: [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it), [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [PrimeIntellect/Qwen3-1.7B-Wordle-SFT](https://huggingface.co/PrimeIntellect/Qwen3-1.7B-Wordle-SFT). Your account must be allowlisted for each checkpoint or downloads will fail even with a valid token.

### 2. Create an access token

1. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens).
2. **Create new token** → choose **Read** (enough to download models).
3. Copy the token once shown: it is one line, usually starting with **`hf_`**. You cannot see it again after leaving the page; if you lose it, create a new token and revoke the old one.

### 3. Install the Hub CLI in your project venv

Activate the repo’s `.venv` (see the quickstart sections above), then install into **that** environment:

```bash
python -m pip install -U huggingface_hub
```

Using `python -m pip` ensures packages land in the active venv, not a different Python on your PATH.

### 4. Log in once (`hf`, not `huggingface-cli`)

The old `huggingface-cli` command is deprecated. Use the **`hf`** tool from `huggingface_hub`:

```bash
hf auth login
```

- When prompted **Enter your token**, paste the **full** `hf_...` string. The cursor may not show characters; that is normal.
- **Windows PowerShell / Terminal:** right-click in the window often pastes; paste **before** pressing Enter. If you press Enter with nothing pasted, login fails with errors like `Bearer` / empty token.
- For **Add token as git credential?** you can answer **`N`** if you only use Python (`transformers` / `from_pretrained`). Answer **`Y`** only if you also use `git clone` / Git LFS against Hub repos and want Git to reuse the token.

### 5. Check that it worked

```bash
hf auth whoami
```

You should see your Hugging Face username. After this, `from_pretrained(...)` can download gated weights; the token is stored in your user config (not in the repo).

### 6. Troubleshooting (short)

| Issue | What to try |
|--------|-------------|
| `Illegal header` / `Bearer` errors | Empty or partial paste — run `hf auth login` again and paste the full token once, then Enter. |
| `403` / access denied on a model | Accept the model license on its Hub page (same account as the token). |
| `hf` not found | Activate `.venv` first, then run `python -m pip install -U huggingface_hub` again. |

Never commit tokens, never paste them into Discord/slack as screenshots of terminal output.

## Python file guide

`scripts/` (run from the repo root):

- `scripts/install.sh` / `scripts/install.ps1`: Bootstraps `uv`, creates `.venv`, installs `requirements.txt`.
- `scripts/gd_1d_torch.py`: Tiny gradient-descent demo on a 1D quadratic (used by `tests/test_install_and_run.py`).
- `scripts/quickstart_wordle.py`: Quick demo of the Wordle environment + discrete policy (mock mode); runs a short interaction and a full episode without training.
- `scripts/run_wordle_comparison.py`: Runs an end-to-end ES vs PPO comparison on Wordle, saves plots to `figures/` and trained weights to `models/`.
- `scripts/run_es_signal_density_probe.py`, `scripts/run_experiment1_closed_loop.py`, `scripts/run_experiment2_minibatch_crn.py`: Headless mirrors of week-12 LoRA notebook cells used to characterize the ES signal-density bottleneck.
- `scripts/run_week16_es.py` (+ `run_week16_es_{nohup,tmux,detached}.sh`, `scripts/slurm/week16_es.sbatch`): Command-line ES (Qwen + LoRA) trainer for SSH / Slurm GPU nodes, with disconnect-safe launchers.

`src/` modules:

- `src/__init__.py`: Package exports for the ES/GridWorld code (re-exports core classes/functions from `gridworld.py` and `es_gridworld.py`).
- `src/gridworld.py`: Core GridWorld environments plus generic PyTorch policy/value networks (including optional LoRA adapters).
- `src/es_gridworld.py`: ES utilities (noise sampling, score functions, gradient estimation), evaluation helpers, plotting, and comparison-table/statistics helpers.
- `src/ppo.py`: PPO training loop utilities (rollout buffer, GAE, policy evaluation) with Wordle-specific training support.
- `src/wordle_env.py`: Wordle environment wrapper (Prime Intellect `verifiers` if installed, otherwise mock fallback) + state embedding + word vocabulary/action mapping.
- `src/wordle_qwen_policy.py`: Week 16 `WordleQwenPolicy` — LoRA on frozen Qwen3, `model.generate`-based guesses with `<guess>...</guess>` parsing (used by `notebooks/week16_wordle_es_qwen.ipynb` and `scripts/run_week16_es.py`).
- `src/wordle_networks.py`: Wordle discrete policy/value networks; produces Wordle guesses (optionally XML-formatted for Prime Intellect) from state embeddings.
- `src/es_wordle.py`: Evolution Strategies training implementation adapted for the Wordle wrapper/policy.
- `src/train_wordle.py`: CLI entrypoint to train a Wordle agent with PPO; handles args, evaluation, and saving model checkpoints.

## Notes

- Install scripts detect whether `uv` is already installed.
- If `uv` is missing, scripts install it using Astral's official installer.
- Dependencies are installed from `requirements.txt` into `.venv` in the repo root.
