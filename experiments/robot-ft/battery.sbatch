#!/bin/bash
#SBATCH --job-name=wued-rft-battery
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --array=0-7
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# Upstream's own battery + report (w01-tek `run.sh report`, flat scene, CPU)
# per policy. Tasks as eval.sbatch (ARMS; 3 x |ARMS| = the imported base);
# the last task runs it on the upstream base run dir itself, so the
# imported base's row checks the conversion (to_upstream_run.py).
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu MUJOCO_GL=disabled
COPY="${COPY:?set COPY=<frozen copy dir>}"
UPSTREAM="${UPSTREAM:?set UPSTREAM=<frozen w01-tek copy with the base run>}"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
BASE_RUN="$UPSTREAM/training/runs/rft_upstream_seed0"
OUT="$PDW/runs/robot_ft/battery"
read -ra ARMS <<<"${ARMS:-accel_sfl_lr1e-4 dr_lr1e-4}"
N=$((3 * ${#ARMS[@]}))
if ((SLURM_ARRAY_TASK_ID == N + 1)); then
  RUN_DIR="$BASE_RUN"
else
  if ((SLURM_ARRAY_TASK_ID == N)); then
    RUN=rft_robot_base_seed0
    CKPT="$COPY/artifacts/upstream/robot_base.pkl"
  else
    RUN="rft_robot_${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}_seed$((SLURM_ARRAY_TASK_ID % 3))"
    CKPT="$PDW/artifacts/robot_ft/$RUN.pkl"
  fi
  RUN_DIR="$OUT/$RUN"
  PYTHONPATH="$COPY/src" "$PY" "$COPY/experiments/robot-ft/to_upstream_run.py" \
    --checkpoint "$CKPT" --base_run "$BASE_RUN" --out_run "$RUN_DIR"
fi
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: $RUN_DIR"
cd "$UPSTREAM/training"
PYTHONPATH=. "$PY" -m wojtek_rl.report --run "$RUN_DIR"
