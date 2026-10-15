#!/bin/bash
#SBATCH --job-name=wued-ws-train
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=07:00:00
#SBATCH --array=0-2
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# ACCEL fine-tune of the imported upstream A1-long policy (seed = task id),
# deploy profile, upstream rewards and DR, clip 0.3, 150M steps
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda WANDB_MODE=online
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
cd "$HOME/workspace/ml-wojtek-ued"
SEED="$SLURM_ARRAY_TASK_ID"
RUN="deploy_ws_accel_seed${SEED}"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: $RUN"

uv run --no-sync python "$COPY/src/scripts/brax_ued.py" --mode accel --name "$RUN" \
  --seed "$SEED" --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --num_timesteps 150000000 --profile deploy --robot wojtek \
  --level_space param --n_edits 2 --edit_scale 0.1 --clipping_epsilon 0.3 \
  --init_from "$COPY/artifacts/upstream/a1long_305m.pkl" --eval_every 4 \
  --checkpoint_dir "$PDW/artifacts/deploy_ws"
