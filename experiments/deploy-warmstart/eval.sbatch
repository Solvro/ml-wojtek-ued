#!/bin/bash
#SBATCH --job-name=wued-ws-eval
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# eval_brax.py, 512 levels / 16 suite instances / 300-step levels.
# Tasks 0-2: train.sbatch seed k; 3: imported upstream baseline (INIT);
# 4: v10_wojtek_accel_seed4. Each deploy policy runs twice: its native
# (deploy) eval, then "v10box" = the v10 6x6 arena with a uniform forward
# command inside both command boxes (eval-only; v10 runs v10box only).
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda WANDB_MODE=online
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
cd "$HOME/workspace/ml-wojtek-ued"
NATIVE=1
case "$SLURM_ARRAY_TASK_ID" in
3) CKPT="${INIT:?set INIT=<imported upstream .pkl>}" RUN=deploy_ws_baseline_upstream_1p5b ;;
4) CKPT="$PDW/artifacts/brax_ued/v10_wojtek_accel_seed4.pkl" RUN=v10_wojtek_accel_seed4 NATIVE=0 ;;
*)
  RUN="deploy_ws_accel_seed${SLURM_ARRAY_TASK_ID}"
  CKPT="$PDW/artifacts/deploy_ws/$RUN.pkl"
  ;;
esac
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: eval $CKPT"
COMMON=(--checkpoint "$CKPT" --entity dominikgalus9-wroc-aw-university-of-science-and-technology
  --project wojtyla --out_dir "$PDW/runs/deploy_ws" --num_levels 512 --suite_instances 16)

if [[ "$NATIVE" == 1 ]]; then
  uv run --no-sync python "$COPY/src/scripts/eval_brax.py" "${COMMON[@]}" \
    --name "${RUN}_final_eval"
fi
uv run --no-sync python "$COPY/src/scripts/eval_brax.py" "${COMMON[@]}" \
  --name "${RUN}_v10box_eval" --arena benchmark \
  --command_box 0.2 0.6 -0.2 0.2 -0.4 0.4 0.14
