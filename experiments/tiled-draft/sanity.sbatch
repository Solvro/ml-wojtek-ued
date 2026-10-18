#!/bin/bash
#SBATCH --job-name=wued-td-sanity
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# Final eval (eval_brax defaults) of one v10 checkpoint per robot on the
# tiled arena, once per backend: task = robot index * 2 + backend index
# (0 wojtek/jax, 1 wojtek/warp, 2 go1/jax, 3 go1/warp). jax must
# reproduce the v10 eval; warp vs jax is the backend gap.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
cd "$HOME/workspace/ml-wojtek-ued" || exit 1

ROBOTS=(wojtek go1)
CHECKPOINTS=(v10_wojtek_accel_seed4 v10_go1_accel_seed0)
BACKENDS=(jax warp)
R=$((SLURM_ARRAY_TASK_ID / 2))
ROBOT="${ROBOTS[$R]}"
CKPT="${CHECKPOINTS[$R]}"
BACKEND="${BACKENDS[$((SLURM_ARRAY_TASK_ID % 2))]}"
RUN="td_sanity_${ROBOT}_${BACKEND}"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: $RUN ($CKPT)"

uv run --no-sync python "$COPY/src/scripts/eval_brax.py" \
  --checkpoint "$PDW/artifacts/brax_ued/$CKPT.pkl" --name "$RUN" \
  --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --out_dir "$PDW/runs/tiled_draft" --level_space param \
  --robot "$ROBOT" --backend "$BACKEND" --tags tiled-draft-sanity
