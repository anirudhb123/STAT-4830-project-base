#!/usr/bin/env bash
# Run week16 ES under nohup so it keeps going if SSH / Cursor / your laptop dies.
# The shell exits immediately; Python reparents to init and ignores SIGHUP.
#
# From repo root (prefers ./.venv/bin/python if it exists):
#   bash scripts/run_week16_es_nohup.sh
#   With normalize_gradient (default in run_week16_es.py), --alpha is step size ‖Δθ‖ (~0.05–0.15):
#   bash scripts/run_week16_es_nohup.sh --skip-alpha-probe --alpha 0.08
#
# Override output directory:
#   WEEK16_ARTIFACTS_DIR=/path/to/run bash scripts/run_week16_es_nohup.sh
#
# Watch progress:
#   tail -f runs/week16_es/nohup_<stamp>_<pid>/console.log
#
# Stop (only if you must):
#   kill "$(cat runs/week16_es/nohup_<...>/pid.txt)"
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs runs/week16_es

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
  echo "[week16_es_nohup] WARN: missing .venv/bin/python — using $(command -v python3)." >&2
  echo "[week16_es_nohup] Fix: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
fi
echo "[week16_es_nohup] PYTHON=$PY"

skip_art=false
for arg in "$@"; do
  if [[ "$arg" == "--artifacts-dir" || "$arg" == --artifacts-dir=* ]]; then
    skip_art=true
    break
  fi
done

if [[ "$skip_art" == true ]]; then
  CMD_ARGS=("$@")
  ART=""
elif [[ -n "${WEEK16_ARTIFACTS_DIR:-}" ]]; then
  ART="${WEEK16_ARTIFACTS_DIR}"
  mkdir -p "$ART"
  CMD_ARGS=("$@" --artifacts-dir "$ART")
else
  ART="$ROOT/runs/week16_es/nohup_${STAMP}_$$"
  mkdir -p "$ART"
  CMD_ARGS=("$@" --artifacts-dir "$ART")
fi

if [[ -n "$ART" ]]; then
  CONSOLE="$ART/console.log"
  echo "$$" >"$ART/parent_shell.txt"
  printf '%s\n' "$PY" >"$ART/interpreter.txt"
else
  CONSOLE="$ROOT/logs/week16_es_nohup_${STAMP}_$$.log"
fi

export PYTHONUNBUFFERED=1

MPL_FALLBACK="$ROOT/.cache/matplotlib"
mkdir -p "$MPL_FALLBACK"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$MPL_FALLBACK}"

nohup "$PY" -u scripts/run_week16_es.py "${CMD_ARGS[@]}" >"$CONSOLE" 2>&1 &
CHILD_PID=$!

if [[ -n "$ART" ]]; then
  echo "$CHILD_PID" >"$ART/pid.txt"
fi

echo "Started (nohup) PID=$CHILD_PID"
echo "Console log: $CONSOLE"
[[ -n "$ART" ]] && echo "Artifacts:   $ART"
echo "Tail: tail -f $CONSOLE"
