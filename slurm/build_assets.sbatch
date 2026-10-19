#!/bin/bash
#SBATCH --job-name=wued-build-assets
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
# Cluster-specific: the #SBATCH account, partition and log paths above and
# REPO below (README "Cluster-specific values").
# Robot model and the accel arena into SRC/.. (the frozen copy the GPU
# arrays run), then mujoco_menagerie for the Playground robots; the arrays
# depend on this job and never rebuild.
set -euo pipefail

REPO="$HOME/workspace/ml-wojtek-ued"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
export HYDRA_FULL_ERROR=1
SRC="${SRC:-$REPO/src}"
export PYTHONPATH="$SRC"
cd "$REPO" || exit 1
echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}: assets into $SRC/.."

uv run --no-sync python -m wojtek_env.tools.build_model
uv run --no-sync python -m hfield.arena
uv run --no-sync python -c "from mujoco_playground._src.mjx_env import \
ensure_menagerie_exists; ensure_menagerie_exists()"
