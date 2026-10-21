#!/bin/bash
#SBATCH --job-name=wued-eval-brax
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# Cluster-specific: the #SBATCH account, partition, gres and log paths above
# and the variables below (docs/slurm.md).
# Same task layout and variables as brax_ued.sbatch (MODES, NSEEDS, SEED0,
# SPACE, PREFIX, ROBOT, ROBOT_ARGS, BACKEND, TAGS, SRC) plus EVAL_DIR.
set -euo pipefail

REPO="${REPO:-$HOME/workspace/ml-wojtek-ued}"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
WB_ENTITY=dominikgalus9-wroc-aw-university-of-science-and-technology
WB_PROJECT="${WB_PROJECT:-wojtek-benchmark-iclr}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# fail instead of silently falling back to CPU when CUDA init fails
# (seen: CUDA_ERROR_MPS_CONNECTION_FAILED on some nodes)
export JAX_PLATFORMS=cuda
# warp allocates its collision scratch outside JAX's pool (env/warp.py)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
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
# BACKEND: auto (warp on the GPU) | jax | warp; TAGS: extra wandb tags
read -ra TAG_LIST <<<"${TAGS:-}"
TAG_LIST_CSV="$(
  IFS=,
  echo "${TAG_LIST[*]}"
)"
RUN="${PREFIX}_${ROBOT}_${MODE}_seed${SEED}"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: eval $RUN"

uv run --no-sync python "$SRC/scripts/eval_brax.py" \
  "checkpoint=$PDW/artifacts/brax_ued/$RUN.pkl" "name=${RUN}_final_eval" \
  "entity=$WB_ENTITY" "project=$WB_PROJECT" \
  "out_dir=$PDW/${EVAL_DIR:-runs/final_eval}" "level_space=${SPACE:-param}" \
  "robot=$ROBOT" "${ROBOT_ARGS[@]}" "backend=${BACKEND:-auto}" \
  "tags=[$TAG_LIST_CSV]"
