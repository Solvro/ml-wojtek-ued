#!/bin/bash
#SBATCH --job-name=wued-t7-train
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
# Sanity run on the tiled arena (warp): v10_wojtek_accel_seed0's flags
# (benchmark profile, param space, 2 edits of +-0.1) for 30M steps.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=online
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
cd "$COPY"
"$PY" src/scripts/brax_ued.py --mode accel --name t7_tiled_wojtek_accel_seed0 \
  --seed 0 --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --num_timesteps 30000000 --level_space param --n_edits 2 \
  --edit_scale 0.1 --checkpoint_dir "$PDW/artifacts/tiled_arena" --robot wojtek \
  --profile benchmark
