#!/bin/bash
#SBATCH --job-name=wued-eval-brax
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# Same task layout as brax_ued.sbatch (NSEEDS, SEED0).
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# SRC=<dir>/src, a frozen copy with the built assets/ beside it (wojtek_env
# resolves assets from src/..), keeps queued jobs on the submitted code
SRC="${SRC:-$HOME/workspace/ml-wojtek-ued/src}"
export PYTHONPATH="$SRC"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
cd "$HOME/workspace/ml-wojtek-ued" || exit 1

read -ra MODES <<<"${MODES:?set MODES, e.g. 'dr accel'}"
NSEEDS="${NSEEDS:-3}"
MODE="${MODES[$((SLURM_ARRAY_TASK_ID / NSEEDS))]}"
SEED=$((${SEED0:-0} + SLURM_ARRAY_TASK_ID % NSEEDS))
PREFIX="${PREFIX:-v7_brax}"
# ROBOT=go1 runs Playground's Go1 at its size (robots/go1.py)
ROBOT="${ROBOT:-wojtek}"
RUN="${PREFIX}_${MODE}_seed${SEED}"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: eval $RUN"

uv run --no-sync python "$SRC/scripts/eval_brax.py" \
  --checkpoint "$PDW/artifacts/brax_ued/$RUN.pkl" --name "${RUN}_final_eval" \
  --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --out_dir "$PDW/runs/final_eval" --level_space "${SPACE:-grid}" \
  --robot "$ROBOT"
