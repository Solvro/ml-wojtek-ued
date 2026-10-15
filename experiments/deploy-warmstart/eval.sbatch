#!/bin/bash
#SBATCH --job-name=wued-ws-eval
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# eval_brax.py: tasks 0-2 = train.sbatch seed k's final checkpoint, task 3 =
# the imported A1-long baseline (no training); 512 levels, 16 suite instances
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda WANDB_MODE=online
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
cd "$HOME/workspace/ml-wojtek-ued"
if [[ "$SLURM_ARRAY_TASK_ID" == 3 ]]; then
  CKPT="$COPY/artifacts/upstream/a1long_305m.pkl"
  RUN=deploy_ws_baseline_a1long
else
  RUN="deploy_ws_accel_seed${SLURM_ARRAY_TASK_ID}"
  CKPT="$PDW/artifacts/deploy_ws/$RUN.pkl"
fi
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: eval $CKPT"

uv run --no-sync python "$COPY/src/scripts/eval_brax.py" \
  --checkpoint "$CKPT" --name "${RUN}_final_eval" \
  --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --out_dir "$PDW/runs/deploy_ws" \
  --num_levels 512 --suite_instances 16
