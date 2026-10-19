#!/bin/bash
#SBATCH --job-name=wued-brax-ued
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%A_%a.out
# task = mode_index * NSEEDS + seed offset over MODES (default "dr accel");
# seed = SEED0 + offset (NSEEDS 3, SEED0 0: seeds 0-2; NSEEDS 2, SEED0 3:
# seeds 3-4). Size the array as NSEEDS * number of modes.
# Assets (robot + accel arena) must already be built; tasks start
# together and must not rebuild.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# fail instead of silently falling back to CPU when CUDA init fails
# (seen: CUDA_ERROR_MPS_CONNECTION_FAILED on some Lem nodes)
export JAX_PLATFORMS=cuda
# warp allocates its collision scratch outside JAX's pool (env/warp.py)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# SRC=<dir>/src, a frozen copy with the built assets/ and conf/ beside it
# (wojtek_env and config.schema resolve them from src/..), keeps queued jobs
# on the submitted code
SRC="${SRC:-$HOME/workspace/ml-wojtek-ued/src}"
export PYTHONPATH="$SRC"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
cd "$HOME/workspace/ml-wojtek-ued" || exit 1

read -ra MODES <<<"${MODES:-dr accel}"
NSEEDS="${NSEEDS:-3}"
VARIANT="${MODES[$((SLURM_ARRAY_TASK_ID / NSEEDS))]}"
# VARIANT = a conf/mode/ option; a "-s<scale>" suffix overrides the easy
# generator's mode.generator_scale. SPACE: conf/level_space/ option (param =
# v10's 8 parameters, 2 edits of +-0.1; grid = control points, 8 of +-0.15,
# 16 edits for the grow variants: conf/grow_edits/).
SPACE="${SPACE:-grid}"
PREFIX="${PREFIX:-v7_brax}"
# ROBOT=go1|barkour|spot|go2 runs a Playground robot at its size (robots/)
ROBOT="${ROBOT:-wojtek}"
# BACKEND: auto (warp on the GPU) | jax | warp; TAGS: extra wandb tags
read -ra TAG_LIST <<<"${TAGS:-}"
TAG_LIST_CSV="$(
  IFS=,
  echo "${TAG_LIST[*]}"
)"
MODE="${VARIANT%-s*}"
OVERRIDES=("mode=$MODE" "level_space=$SPACE")
if [[ "$VARIANT" == *-s* ]]; then
  OVERRIDES+=("mode.generator_scale=${VARIANT##*-s}")
fi
SEED=$((${SEED0:-0} + SLURM_ARRAY_TASK_ID % NSEEDS))
RUN="${PREFIX}_${VARIANT}_seed${SEED}"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: $RUN"

uv run --no-sync python "$SRC/scripts/brax_ued.py" "${OVERRIDES[@]}" \
  "name=$RUN" "seed=$SEED" \
  entity=dominikgalus9-wroc-aw-university-of-science-and-technology \
  project=wojtyla "num_timesteps=${NUM_TIMESTEPS:-300000000}" \
  "checkpoint_dir=$PDW/artifacts/brax_ued" "robot=$ROBOT" \
  "backend=${BACKEND:-auto}" "tags=[$TAG_LIST_CSV]"
