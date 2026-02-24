#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "Repository root: ${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv with Astral installer..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is still not available on PATH after install."
  echo "Open a new shell and run this installer again."
  exit 1
fi

if [ -d ".venv" ]; then
  echo "Virtual environment already exists at .venv; reusing it."
else
  echo "Creating virtual environment at .venv ..."
  uv venv .venv
fi

echo "Installing requirements into .venv ..."
uv pip install --python ".venv/bin/python" -r requirements.txt

echo
echo "Install complete."
echo "Next steps:"
echo "  1) Activate venv: source .venv/bin/activate"
echo "  2) Run script:    python script/gd_1d_torch.py"
echo "  3) Run tests:     pytest"
