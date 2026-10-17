#!/bin/bash
#SBATCH --job-name=wued-rft-eval
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=01:30:00
#SBATCH --array=0-6
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# eval_brax.py on warp, per policy: flat on 1024 levels (the flat guard),
# native (deploy arena geometry, the robot command mix; defaults: 2048
# levels x 4 episodes x 2 passes) and v10box (v10 6x6 m arena, a uniform
# forward command inside both command boxes, the robot's fixed 0.125 m
# height). Tasks as train.sbatch (ARMS); task 3 x |ARMS| = the imported base.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda WANDB_MODE=online
export XLA_PYTHON_CLIENT_PREALLOCATE=false
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
read -ra ARMS <<<"${ARMS:-accel_sfl_lr1e-4 dr_lr1e-4}"
if ((SLURM_ARRAY_TASK_ID == 3 * ${#ARMS[@]})); then
  RUN=rft_robot_base_seed0
  CKPT="$COPY/artifacts/upstream/robot_base.pkl"
else
  RUN="rft_robot_${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}_seed$((SLURM_ARRAY_TASK_ID % 3))"
  CKPT="$PDW/artifacts/robot_ft/$RUN.pkl"
fi
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: $CKPT"
cd "$COPY"
COMMON=(--checkpoint "$CKPT" --entity dominikgalus9-wroc-aw-university-of-science-and-technology
  --project wojtyla --out_dir "$PDW/runs/robot_ft")
"$PY" src/scripts/eval_brax.py "${COMMON[@]}" --name "${RUN}_flat_eval" \
  --flat_only --flat_levels 1024
"$PY" src/scripts/eval_brax.py "${COMMON[@]}" --name "${RUN}_final_eval"
"$PY" src/scripts/eval_brax.py "${COMMON[@]}" --name "${RUN}_v10box_eval" \
  --arena benchmark --command_box 0.2 0.6 -0.2 0.2 -0.4 0.4 0.125
