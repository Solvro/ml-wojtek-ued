#!/bin/bash
#SBATCH --job-name=wued-ab-eval
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# eval_brax.py on the final checkpoint of arms.sbatch task i (same order),
# 300-step levels for every arm; 512 levels / 16 suite instances keep it short
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda WANDB_MODE=online
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
cd "$HOME/workspace/ml-wojtek-ued"
ARMS=(a2_v10_reward a3_level1000 a4_clip03 a5_flat a6_torque_rate)
RUN="deploy_ab_${ARMS[$SLURM_ARRAY_TASK_ID]}_seed0"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: eval $RUN"

uv run --no-sync python "$COPY/src/scripts/eval_brax.py" \
  --checkpoint "$PDW/artifacts/deploy_ab/$RUN.pkl" --name "${RUN}_final_eval" \
  --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --out_dir "$PDW/runs/deploy_ab" \
  --num_levels 512 --suite_instances 16
