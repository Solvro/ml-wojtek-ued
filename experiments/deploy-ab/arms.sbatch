#!/bin/bash
#SBATCH --job-name=wued-ab-arm
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# A2-A6 of experiments/deploy-ab: brax_ued --profile deploy, seed 0, 50M
# steps, T4's ACCEL setup (param space, 2 edits of +-0.1); each task changes
# one training-only setting.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda WANDB_MODE=online
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
ARMS="$COPY/experiments/deploy-ab/arms"
cd "$HOME/workspace/ml-wojtek-ued"

MODE=(--mode accel)
case "$SLURM_ARRAY_TASK_ID" in
0) ARM=a2_v10_reward EXTRA=(--env_override "$ARMS/a2_v10_reward.json") ;;
1) ARM=a3_level1000 EXTRA=(--level_steps 1000 --eval_every 1) ;;
2) ARM=a4_clip03 EXTRA=(--clipping_epsilon 0.3) ;;
3) ARM=a5_flat MODE=(--mode dr_easy) EXTRA=(--generator_scale 0.0) ;;
4) ARM=a6_torque_rate EXTRA=(--env_override "$ARMS/a6_torque_rate.json") ;;
*) echo "no arm for task $SLURM_ARRAY_TASK_ID" >&2 && exit 1 ;;
esac
RUN="deploy_ab_${ARM}_seed0"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: $RUN"

uv run --no-sync python "$COPY/src/scripts/brax_ued.py" "${MODE[@]}" --name "$RUN" \
  --seed 0 --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --num_timesteps 50000000 --profile deploy --robot wojtek \
  --level_space param --n_edits 2 --edit_scale 0.1 --eval_every 2 \
  --checkpoint_dir "$PDW/artifacts/deploy_ab" "${EXTRA[@]}"
