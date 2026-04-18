# Agent Environment Setup

This guide is for future agents working in this repository on a VM, workstation, or other remote machine. It is written to be modular: use the sections you need for the current task instead of assuming this exact system layout.

## 1. Baseline repo setup

From the repository root:

```bash
bash script/install.sh
source .venv/bin/activate
```

This project's install script uses `uv`, creates `.venv` in the repo root, and installs `requirements.txt`.

Quick verification:

```bash
python --version
python -m pip --version
pytest
```

If `pytest` is too expensive for the current task, at least verify imports:

```bash
python - <<'PY'
import torch, transformers, numpy, jinja2
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("numpy", numpy.__version__)
print("jinja2", jinja2.__version__)
PY
```

## 2. Hugging Face setup

Some notebooks and scripts load gated Hugging Face checkpoints such as Gemma.

Gemma and other instruction-tuned checkpoints in this repo also rely on `transformers` chat templates, which require `jinja2>=3.1.0` in the active Python environment.

Inside the active repo environment:

```bash
python -m pip install -U "jinja2>=3.1.0"
python -m pip install -U huggingface_hub
hf auth login
hf auth whoami
```

Rules:

- Never commit tokens.
- Never write tokens into tracked files.
- If a model is gated, the user account must also have accepted the model's license on Hugging Face.

## 3. Jupyter / notebook setup

If working from notebooks, prefer registering the repo environment as an explicit kernel instead of relying on whatever `python` happens to resolve to.

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name stat4830 --display-name "Python 3 (stat4830)"
```

Then switch the notebook kernel to `Python 3 (stat4830)`.

Quick notebook-side verification cell:

```python
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
```

If the notebook uses a different interpreter than the shell, fix the kernel before debugging anything else.

If the notebook uses Gemma or another chat-template model, also verify `jinja2>=3.1.0` inside that same kernel:

```python
import jinja2
print(jinja2.__version__)
```

## 4. GPU preflight checklist

Only do this when the task needs CUDA.

### OS-level checks

```bash
nvidia-smi
```

You want:

- an NVIDIA GPU to appear
- a valid driver version
- no obvious driver communication error

### Python-level checks

Inside the repo environment:

```bash
python - <<'PY'
import os, sys, torch
print("python:", sys.executable)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("cuda.device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
PY
```

Interpretation:

- `nvidia-smi` works but `torch.cuda.is_available()` is `False`: usually a PyTorch/driver compatibility problem.
- `CUDA_VISIBLE_DEVICES` is empty or restrictive: GPU may be hidden from the process.
- `torch.cuda.device_count() > 0` but device queries fail: CUDA initialization is partially seeing hardware but rejecting the runtime stack.
- Gemma or other chat-template models fail with `apply_chat_template requires jinja2>=3.1.0`: upgrade `jinja2` in the active environment and restart the kernel.

## 5. CUDA/PyTorch compatibility workflow

This is the most important troubleshooting section for future agents.

### Step A: compare driver support with the installed PyTorch build

Check the driver's CUDA support:

```bash
nvidia-smi
```

Check the wheel that is installed:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda:", torch.cuda.is_available())
PY
```

### Step B: decide whether the environment is mismatched

Example mismatch:

- `nvidia-smi` reports `CUDA Version: 12.6`
- installed PyTorch reports `torch.version.cuda == 13.0`
- `torch.cuda.is_available()` is `False`
- CUDA calls raise "driver too old"

In that situation, the OS sees the GPU, but the PyTorch wheel is built for a newer CUDA runtime than the installed NVIDIA driver supports.

### Step C: choose one fix

Pick one:

1. Upgrade the NVIDIA driver to support the installed PyTorch CUDA build.
2. Replace PyTorch with a wheel compiled for a CUDA version supported by the current driver.

If you do not control the VM image, replacing the PyTorch wheel is usually easier.

## 6. Known-good repair pattern for this project

This repo previously hit the following issue on an A100 VM:

- `nvidia-smi` worked
- driver exposed CUDA 12.6
- notebook environment had `torch==2.11.0+cu130`
- `torch.cuda.is_available()` returned `False`

The fix was to replace the CUDA 13 PyTorch stack with a CUDA 12.6 build.

Uninstall the incompatible stack:

```bash
python -m pip uninstall -y \
  torch triton \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-curand nvidia-cusolver nvidia-cusparse \
  nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvtx \
  nvidia-nvshmem-cu13 cuda-bindings cuda-toolkit
```

Install a driver-compatible build:

```bash
python -m pip install --upgrade pip
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu126 torch
```

Verify:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

Then restart the notebook kernel and rerun the import/setup cell.

## 7. How to adapt this to a different machine

Do not hardcode `cu126` just because it worked once.

Use this rule instead:

1. Inspect `nvidia-smi`.
2. Inspect `torch.version.cuda`.
3. If they are not compatible, install a PyTorch wheel that matches the driver's supported CUDA generation, or upgrade the driver.
4. Re-verify in both shell and notebook kernel.

If the machine has no NVIDIA GPU, expect CUDA to remain unavailable and fall back to CPU.

## 8. Minimal triage order for future agents

When a user says "CUDA is false" or "GPU is not detected", do these in order:

1. Confirm the notebook or script is using the intended interpreter.
2. Run `nvidia-smi`.
3. Print `torch.__version__`, `torch.version.cuda`, and `torch.cuda.is_available()`.
4. Check `CUDA_VISIBLE_DEVICES`.
5. If the GPU is visible to the OS but not to PyTorch, treat it as a runtime compatibility problem first.
6. Fix the environment, restart the kernel, and verify again.

For Gemma/chat-template failures specifically:

```bash
python -m pip install -U "jinja2>=3.1.0"
```

Then restart the notebook kernel before re-running the setup cell.

## 9. Project-specific reminders

- The repo quickstart is in `README.md`.
- The notebook `notebooks/week12_implementation.ipynb` selects CUDA purely through `torch.cuda.is_available()`, so environment correctness matters more than notebook edits.
- Prefer environment fixes over patching notebook logic when the GPU is present but initialization fails.
- If you change the Python environment for notebook work, restart the kernel before re-testing.
