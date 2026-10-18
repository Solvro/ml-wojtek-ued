#!/bin/bash
#SBATCH --job-name=wued-td-setup
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
# Robot model and the v10 tiled accel arena into the frozen copy COPY; the
# GPU arrays depend on this job and never rebuild.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}: assets into $COPY"
cd "$COPY"
"$PY" -m wojtek_env.tools.build_model
"$PY" -m hfield.arena hfield_params.amplitude=0.3 hfield_params.coarse_n=13 \
  hfield_params.pad_radius=0.55 hfield_params.pad_taper=0.2
