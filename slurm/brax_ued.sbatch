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
# fail instead of silently falling back to CPU when CUDA init fails
# (seen: CUDA_ERROR_MPS_CONNECTION_FAILED on some Lem nodes)
export JAX_PLATFORMS=cuda
# SRC=<dir>/src, a frozen copy with the built assets/ beside it (wojtek_env
# resolves assets from src/..), keeps queued jobs on the submitted code
SRC="${SRC:-$HOME/workspace/ml-wojtek-ued/src}"
export PYTHONPATH="$SRC"
PDW=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued
cd "$HOME/workspace/ml-wojtek-ued" || exit 1

read -ra MODES <<<"${MODES:-dr accel}"
NSEEDS="${NSEEDS:-3}"
VARIANT="${MODES[$((SLURM_ARRAY_TASK_ID / NSEEDS))]}"
# ACCEL variants share --mode accel and differ only in curriculum switches;
# a "-s<scale>" suffix sets the easy generator's --generator_scale.
# SPACE=param switches to the 8-parameter level space (2 edits of +-0.1
# per mutation instead of the grid's 16 of +-0.15).
SPACE="${SPACE:-grid}"
PREFIX="${PREFIX:-v7_brax}"
# ROBOT=go1 runs Playground's Go1 at its size (robots/go1.py)
ROBOT="${ROBOT:-wojtek}"
if [[ "$SPACE" == param ]]; then
  SPACE_ARGS=(--level_space param --n_edits 2 --edit_scale 0.1)
  GROW=(--mutation grow)
else
  SPACE_ARGS=()
  GROW=(--mutation grow --n_edits 16)
fi
BASE="${VARIANT%-s*}"
case "$BASE" in
accel_grow) MODE=accel EXTRA=("${GROW[@]}") ;;
accel_sfl) MODE=accel EXTRA=("${GROW[@]}" --score learnability) ;;
*) MODE="$BASE" EXTRA=() ;;
esac
EXTRA=("${SPACE_ARGS[@]}" "${EXTRA[@]}")
if [[ "$VARIANT" == *-s* ]]; then
  EXTRA+=(--generator_scale "${VARIANT##*-s}")
fi
SEED=$((${SEED0:-0} + SLURM_ARRAY_TASK_ID % NSEEDS))
RUN="${PREFIX}_${VARIANT}_seed${SEED}"
echo "job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID} on ${SLURM_JOB_NODELIST}: $RUN"

uv run --no-sync python "$SRC/scripts/brax_ued.py" --mode "$MODE" --name "$RUN" \
  --seed "$SEED" --entity dominikgalus9-wroc-aw-university-of-science-and-technology \
  --project wojtyla --num_timesteps "${NUM_TIMESTEPS:-300000000}" "${EXTRA[@]}" \
  --checkpoint_dir "$PDW/artifacts/brax_ued" --robot "$ROBOT"
