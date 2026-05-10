#!/usr/bin/env bash
# Start week16 ES in a detached tmux session so SSH / Cursor disconnects do not stop training.
#
# Usage (from repo root):
#   bash scripts/run_week16_es_tmux.sh
#   bash scripts/run_week16_es_tmux.sh --skip-alpha-probe --alpha 3e-5
#
# Override output directory:
#   WEEK16_ARTIFACTS_DIR=/path/to/run bash scripts/run_week16_es_tmux.sh
#
# Attach to live output:
#   tmux attach -t week16_es
# Detach (leave running): Ctrl-b then d
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs runs/week16_es
LOG="$ROOT/logs/week16_es_tmux_${STAMP}.log"

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
elif [[ -n "${WEEK16_ARTIFACTS_DIR:-}" ]]; then
  ART="${WEEK16_ARTIFACTS_DIR}"
  mkdir -p "$ART"
  CMD_ARGS=("$@" --artifacts-dir "$ART")
else
  ART="$ROOT/runs/week16_es/tmux_${STAMP}_$$"
  mkdir -p "$ART"
  CMD_ARGS=("$@" --artifacts-dir "$ART")
fi

SESS="${WEEK16_TMUX_SESSION:-week16_es}"
if tmux has-session -t "$SESS" 2>/dev/null; then
  echo "tmux session '$SESS' already exists. Attach: tmux attach -t $SESS" >&2
  exit 1
fi

tmux new-session -d -s "$SESS" -c "$ROOT" \
  bash -c 'PY="$1"; LOG="$2"; shift 2; exec "$PY" -u scripts/run_week16_es.py "$@" 2>&1 | tee "$LOG"' \
  bash "$PY" "$LOG" "${CMD_ARGS[@]}"

echo "Started tmux session: $SESS"
echo "Console log (tee): $LOG"
[[ -n "${ART:-}" ]] && echo "Artifacts dir:     $ART"
echo "Attach: tmux attach -t $SESS"
