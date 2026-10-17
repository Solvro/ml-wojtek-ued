#!/bin/bash
#SBATCH --job-name=wued-rft-train
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# Fine-tunes of the imported upstream base (setup.sbatch), robot profile,
# upstream rewards and DR, clip 0.3, warp tiled arena, 300M steps: T8's
# winning recipe (ACCEL SFL, mixed generator, grow mutation, lr 1e-4) and
# its DR control at the same lr. task = arm index in ARMS * 3 + seed.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda WANDB_MODE=online
export XLA_PYTHON_CLIENT_PREALLOCATE=false
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
OUT="$PDW/artifacts/robot_ft"
read -ra ARMS <<<"${ARMS:-accel_sfl_lr1e-4 dr_lr1e-4}"
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED=$((SLURM_ARRAY_TASK_ID % 3))
case "$ARM" in
accel_sfl_lr1e-4)
  ARGS=(--mode accel --n_edits 2 --edit_scale 0.1 --generator mixed
    --mutation grow --score learnability --learning_rate 1e-4)
  ;;
dr_lr1e-4) ARGS=(--mode dr --learning_rate 1e-4) ;;
*)
  echo "unknown arm $ARM" >&2
  exit 1
  ;;
esac
RUN="rft_robot_${ARM}_seed${SEED}"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: $RUN"
cd "$COPY"
"$PY" src/scripts/brax_ued.py "${ARGS[@]}" --name "$RUN" --seed "$SEED" \
  --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --num_timesteps "${NUM_TIMESTEPS:-300000000}" \
  --profile robot --robot wojtek --level_space param --clipping_epsilon 0.3 \
  --init_from "$COPY/artifacts/upstream/robot_base.pkl" --eval_every 20 \
  --checkpoint_dir "$OUT"
