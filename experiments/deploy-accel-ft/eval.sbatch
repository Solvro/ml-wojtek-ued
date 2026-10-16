#!/bin/bash
#SBATCH --job-name=wued-dft-eval
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=01:30:00
#SBATCH --array=0-9
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# eval_brax.py defaults (2048 levels x 4 episodes x 2 passes), warp, twice per
# policy: native (deploy arena and command mix) and v10box (v10 6x6 m arena,
# a uniform forward command inside both command boxes). Tasks 0-8 as
# train.sbatch; 9 = the imported base, no fine-tune.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda WANDB_MODE=online
export XLA_PYTHON_CLIENT_PREALLOCATE=false
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
ARMS=(accel accel_lr1e-4 dr)
if [[ "$SLURM_ARRAY_TASK_ID" == 9 ]]; then
  RUN=dft_deploy_base_seed0
  CKPT="$COPY/artifacts/upstream/base_1p5b.pkl"
else
  RUN="dft_deploy_${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}_seed$((SLURM_ARRAY_TASK_ID % 3))"
  CKPT="$PDW/artifacts/deploy_accel_ft/$RUN.pkl"
fi
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: $CKPT"
cd "$COPY"
COMMON=(--checkpoint "$CKPT" --entity dominikgalus9-wroc-aw-university-of-science-and-technology
  --project wojtyla --out_dir "$PDW/runs/deploy_accel_ft")
"$PY" src/scripts/eval_brax.py "${COMMON[@]}" --name "${RUN}_final_eval"
"$PY" src/scripts/eval_brax.py "${COMMON[@]}" --name "${RUN}_v10box_eval" \
  --arena benchmark --command_box 0.2 0.6 -0.2 0.2 -0.4 0.4 0.14
