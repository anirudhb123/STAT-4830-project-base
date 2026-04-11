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

## Hugging Face (gated models + login)

Use this when a notebook or script loads weights from the Hub (e.g. **Gemma**). You do **not** put a token in the repo or in a committed `.env` file.

### 1. Account and model access

1. Create a [Hugging Face](https://huggingface.co) account (or sign in).
2. For **gated** models, open that model’s page while logged in (e.g. [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)) and click **Agree and access** (or equivalent). Your account must be allowlisted for that checkpoint or downloads will fail even with a valid token.

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

Top-level scripts:

- `quickstart_wordle.py`: Quick demo of the Wordle environment + discrete policy (mock mode); runs a short interaction and a full episode without training.
- `run_wordle_comparison.py`: Runs an end-to-end ES vs PPO comparison on Wordle, saves plots to `figures/` and trained weights to `models/`.

`src/` modules:

- `src/__init__.py`: Package exports for the ES/GridWorld code (re-exports core classes/functions from `gridworld.py` and `es_gridworld.py`).
- `src/gridworld.py`: Core GridWorld environments plus generic PyTorch policy/value networks (including optional LoRA adapters).
- `src/es_gridworld.py`: ES utilities (noise sampling, score functions, gradient estimation), evaluation helpers, plotting, and comparison-table/statistics helpers.
- `src/ppo.py`: PPO training loop utilities (rollout buffer, GAE, policy evaluation) with Wordle-specific training support.
- `src/wordle_env.py`: Wordle environment wrapper (Prime Intellect `verifiers` if installed, otherwise mock fallback) + state embedding + word vocabulary/action mapping.
- `src/wordle_networks.py`: Wordle discrete policy/value networks; produces Wordle guesses (optionally XML-formatted for Prime Intellect) from state embeddings.
- `src/es_wordle.py`: Evolution Strategies training implementation adapted for the Wordle wrapper/policy.
- `src/train_wordle.py`: CLI entrypoint to train a Wordle agent with PPO; handles args, evaluation, and saving model checkpoints.

## Notes

- Install scripts detect whether `uv` is already installed.
- If `uv` is missing, scripts install it using Astral's official installer.
- Dependencies are installed from `requirements.txt` into `.venv` in the repo root.
