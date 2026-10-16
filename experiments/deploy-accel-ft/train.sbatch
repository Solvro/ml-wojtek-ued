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
# (the go/no-go point). task = arm index in ARMS * 3 + seed; size the array
# as 3 x the number of arms. ARMS default = launch 2 ("accel accel_lr1e-4
# dr"); launch 3: "accel_hard_lr1e-4 dr_lr1e-4 accel_sfl_lr1e-4
# accel_lr1e-4_1b". accel_lr1e-4_1b continues launch 2's accel_lr1e-4 seed
# from its 300M checkpoint for 700M more (fresh Adam state and buffer).
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda WANDB_MODE=online
export XLA_PYTHON_CLIENT_PREALLOCATE=false
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
OUT="$PDW/artifacts/deploy_accel_ft"
read -ra ARMS <<<"${ARMS:-accel accel_lr1e-4 dr}"
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED=$((SLURM_ARRAY_TASK_ID % 3))
INIT="$COPY/artifacts/upstream/base_1p5b.pkl"
STEPS=300000000
ACCEL=(--mode accel --n_edits 2 --edit_scale 0.1)
HARD=("${ACCEL[@]}" --generator mixed --mutation grow --learning_rate 1e-4)
case "$ARM" in
accel) ARGS=("${ACCEL[@]}") ;;
accel_lr1e-4) ARGS=("${ACCEL[@]}" --learning_rate 1e-4) ;;
dr) ARGS=(--mode dr) ;;
dr_lr1e-4) ARGS=(--mode dr --learning_rate 1e-4) ;;
accel_hard_lr1e-4) ARGS=("${HARD[@]}") ;;
accel_sfl_lr1e-4) ARGS=("${HARD[@]}" --score learnability) ;;
accel_lr1e-4_1b)
  ARGS=("${ACCEL[@]}" --learning_rate 1e-4)
  INIT="$OUT/dft2_deploy_accel_lr1e-4_seed${SEED}.pkl"
  STEPS=700000000
  ;;
*)
  echo "unknown arm $ARM" >&2
  exit 1
  ;;
esac
RUN="dft2_deploy_${ARM}_seed${SEED}"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: $RUN from $INIT"
cd "$COPY"
"$PY" src/scripts/brax_ued.py "${ARGS[@]}" --name "$RUN" --seed "$SEED" \
  --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --num_timesteps "${NUM_TIMESTEPS:-$STEPS}" \
  --profile deploy --robot wojtek --level_space param --clipping_epsilon 0.3 \
  --init_from "$INIT" --eval_every 20 --checkpoint_dir "$OUT"
