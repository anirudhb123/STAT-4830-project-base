#!/bin/bash
#SBATCH --job-name=w16es
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=logs/week16_es-%j.out
#SBATCH --error=logs/week16_es-%j.err
#
# Week16 Wordle ES (Qwen LoRA) — one self-contained output directory per job.
#
# Before first run (creates log dir Slurm needs):
#   mkdir -p logs runs/week16_es
#
# From repo root:
#   sbatch scripts/slurm/week16_es.sbatch
#
# Edit #SBATCH lines above for your site (partition, qos, account, gpu type, mem, time).
# Optional node-local HF cache (uncomment if your cluster recommends it):
#   export HF_HOME=/scratch/$USER/.cache/huggingface
#
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"

mkdir -p logs runs/week16_es

ART="${REPO}/runs/week16_es/job_${SLURM_JOB_ID:-manual}"
mkdir -p "$ART"

echo "[slurm] SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "[slurm] ARTIFACTS_DIR=${ART}"

if [[ -x "${REPO}/.venv/bin/python" ]]; then
  PY="${REPO}/.venv/bin/python"
else
  PY="python3"
fi

export PYTHONUNBUFFERED=1

# Extra CLI args: edit here, e.g. --skip-alpha-probe --alpha 3e-5
exec "$PY" -u scripts/run_week16_es.py --artifacts-dir "$ART"
