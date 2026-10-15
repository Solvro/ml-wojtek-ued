#!/bin/bash
#SBATCH --job-name=wued-eval-sim2sim
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# Same task layout as brax_ued.sbatch: task = mode_index * 3 + seed over MODES.
# CPU-only (physics on CPU MuJoCo); Lem CPU nodes run the same AlmaLinux 9
# as the GPU nodes the venv was built for. Robot and level space come from
# each checkpoint's sidecar <run>.json.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
# SRC=<frozen copy of src/> keeps queued jobs on the code they were
# submitted with (assets built inside it)
SRC="${SRC:-$HOME/workspace/ml-wojtek-ued/src}"
export PYTHONPATH="$SRC"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
cd "$HOME/workspace/ml-wojtek-ued" || exit 1

read -ra MODES <<<"${MODES:?set MODES, e.g. 'dr accel'}"
MODE="${MODES[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED=$((SLURM_ARRAY_TASK_ID % 3))
PREFIX="${PREFIX:-v10_wojtek}"
RUN="${PREFIX}_${MODE}_seed${SEED}"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: sim2sim $RUN"

uv run --no-sync python "$SRC/scripts/eval_sim2sim.py" \
  --checkpoint "$PDW/artifacts/brax_ued/$RUN.pkl" --name "${RUN}_sim2sim" \
  --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --out_dir "$PDW/runs/sim2sim"
