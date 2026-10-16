#!/bin/bash
#SBATCH --job-name=wued-t7-eval
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# v10_wojtek_accel_seed4's final eval (eval_brax.py defaults, as the
# original v10_wojtek_accel_seed4_final_eval) on the tiled arena. Task 0:
# the sidecar's own backend (jax). Task 1: warp, from a copy of the
# sidecar with sim.backend and warp's buffers edited.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=online
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
RUN=v10_wojtek_accel_seed4
OUT="$PDW/runs/tiled_arena"
mkdir -p "$OUT/ckpt"
CKPT="$PDW/artifacts/brax_ued/$RUN.pkl"
BACKEND=jax
if [[ "$SLURM_ARRAY_TASK_ID" == 1 ]]; then
  BACKEND=warp
  cp "$CKPT" "$OUT/ckpt/${RUN}_warp.pkl"
  "$PY" - "$PDW/artifacts/brax_ued/$RUN.json" "$OUT/ckpt/${RUN}_warp.json" <<'PY'
import json, sys
spec = json.loads(open(sys.argv[1]).read())
# warp's buffers as the tiled arena sets them (utils.env_overrides)
spec["env_overrides"]["sim"].update(backend="warp", naconmax_per_env=116, njmax=1024)
open(sys.argv[2], "w").write(json.dumps(spec, indent=2) + "\n")
PY
  CKPT="$OUT/ckpt/${RUN}_warp.pkl"
fi
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: $BACKEND"
cd "$COPY"
"$PY" src/scripts/eval_brax.py --checkpoint "$CKPT" \
  --name "${RUN}_final_eval_tiled_${BACKEND}" \
  --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --out_dir "$OUT"
