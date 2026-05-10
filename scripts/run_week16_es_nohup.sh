#!/usr/bin/env bash
# Run week16 ES under nohup so it keeps going if SSH / Cursor / your laptop dies.
# The shell exits immediately; Python reparents to init and ignores SIGHUP.
#
# From repo root:
#   bash scripts/run_week16_es_nohup.sh
#   bash scripts/run_week16_es_nohup.sh --skip-alpha-probe --alpha 3e-5
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
fi

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
else
  CONSOLE="$ROOT/logs/week16_es_nohup_${STAMP}_$$.log"
fi

export PYTHONUNBUFFERED=1

nohup "$PY" -u scripts/run_week16_es.py "${CMD_ARGS[@]}" >"$CONSOLE" 2>&1 &
CHILD_PID=$!

if [[ -n "$ART" ]]; then
  echo "$CHILD_PID" >"$ART/pid.txt"
fi

echo "Started (nohup) PID=$CHILD_PID"
echo "Console log: $CONSOLE"
[[ -n "$ART" ]] && echo "Artifacts:   $ART"
echo "Tail: tail -f $CONSOLE"
