#!/bin/bash
#SBATCH --job-name=wued-eval-sim2sim
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# Cluster-specific: the #SBATCH account, partition and log paths above and
# the variables below (docs/slurm.md).
# Same task layout as brax_ued.sbatch (MODES, NSEEDS, SEED0, PREFIX, ROBOT,
# SRC). CPU-only (physics on CPU MuJoCo); the CPU nodes run the same OS as
# the GPU nodes the venv was built for.
set -euo pipefail

REPO="${REPO:-$HOME/workspace/ml-wojtek-ued}"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
WB_ENTITY=dominikgalus9-wroc-aw-university-of-science-and-technology
WB_PROJECT="${WB_PROJECT:-wojtek-benchmark-iclr}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
export HYDRA_FULL_ERROR=1
# SRC=<dir>/src, a frozen copy with the built assets/ and conf/ beside it
# (wojtek_env and config.schema resolve them from src/..), keeps queued jobs
# on the submitted code
SRC="${SRC:-$REPO/src}"
export PYTHONPATH="$SRC"
cd "$REPO" || exit 1

read -ra MODES <<<"${MODES:?set MODES, e.g. 'dr accel'}"
NSEEDS="${NSEEDS:-3}"
MODE="${MODES[$((SLURM_ARRAY_TASK_ID / NSEEDS))]}"
SEED=$((${SEED0:-0} + SLURM_ARRAY_TASK_ID % NSEEDS))
PREFIX="${PREFIX:-ued}"
# ROBOT: a conf/robot/ option
ROBOT="${ROBOT:-wojtek}"
# ROBOT_ARGS: robot.* Hydra overrides, space-separated ("robot.kp=50
# robot.scale=2.0"); the evals of a run need the same ones
read -ra ROBOT_ARGS <<<"${ROBOT_ARGS:-}"
for arg in "${ROBOT_ARGS[@]}"; do
  [[ "$arg" == robot.* ]] || {
    echo "ROBOT_ARGS entry '$arg' is not a robot.* override" >&2
    exit 1
  }
done
RUN="${PREFIX}_${ROBOT}_${MODE}_seed${SEED}"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: sim2sim $RUN"

uv run --no-sync python "$SRC/scripts/eval_sim2sim.py" \
  "checkpoint=$PDW/artifacts/brax_ued/$RUN.pkl" "name=${RUN}_sim2sim" \
  "entity=$WB_ENTITY" "project=$WB_PROJECT" \
  "out_dir=$PDW/runs/sim2sim" "robot=$ROBOT" "${ROBOT_ARGS[@]}"
