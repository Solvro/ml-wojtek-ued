#!/bin/bash
#SBATCH --job-name=wued-dft-train
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=04:00:00
#SBATCH --array=0-8
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# Fine-tunes of the imported 1.5B upstream base (setup.sbatch), deploy
# profile, upstream rewards and DR, clip 0.3, warp tiled arena, 300M steps
# (the go/no-go point; NUM_TIMESTEPS overrides).
# task = arm * 3 + seed: 0-2 A accel (T6's flags), 3-5 B accel + lr 1e-4,
# 6-8 C dr (uniform levels, no curriculum).
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda WANDB_MODE=online
export XLA_PYTHON_CLIENT_PREALLOCATE=false
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
SEED=$((SLURM_ARRAY_TASK_ID % 3))
case $((SLURM_ARRAY_TASK_ID / 3)) in
0) ARM=accel ARGS=(--mode accel --n_edits 2 --edit_scale 0.1) ;;
1) ARM=accel_lr1e-4 ARGS=(--mode accel --n_edits 2 --edit_scale 0.1 --learning_rate 1e-4) ;;
2) ARM=dr ARGS=(--mode dr) ;;
esac
RUN="dft2_deploy_${ARM}_seed${SEED}"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: $RUN"
cd "$COPY"
"$PY" src/scripts/brax_ued.py "${ARGS[@]}" --name "$RUN" --seed "$SEED" \
  --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --num_timesteps "${NUM_TIMESTEPS:-300000000}" \
  --profile deploy --robot wojtek --level_space param --clipping_epsilon 0.3 \
  --init_from "$COPY/artifacts/upstream/base_1p5b.pkl" --eval_every 20 \
  --checkpoint_dir "$PDW/artifacts/deploy_accel_ft"
