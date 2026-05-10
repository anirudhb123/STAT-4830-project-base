#!/usr/bin/env bash
# Dispatcher: run week16 ES in nohup (default) or tmux.
#
#   WEEK16_DETACH_MODE=nohup   bash scripts/run_week16_es_detached.sh   # default
#   WEEK16_DETACH_MODE=tmux    bash scripts/run_week16_es_detached.sh
#
# All extra args are passed through to run_week16_es.py.
#
set -euo pipefail
MODE="${WEEK16_DETACH_MODE:-nohup}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "$ROOT/scripts/run_week16_es_${MODE}.sh" "$@"
