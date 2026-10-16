#!/bin/bash
#SBATCH --job-name=wued-ab-build
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
# Assets for both frozen copies before the arms start: our robot model and
# accel_deploy arena into COPY, upstream's model and train arena (flags from
# the terrain_blind_v4_2 preset header) into UPSTREAM.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
COPY="${COPY:?set COPY=<frozen copy dir>}"
UPSTREAM="${UPSTREAM:?set UPSTREAM=<frozen w01-tek copy>}"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
cd "$COPY"
PYTHONPATH="$COPY/src" "$PY" -m wojtek_env.tools.build_model
PYTHONPATH="$COPY/src" "$PY" -m hfield.arena arena=accel_deploy \
  hfield_params.patch_radius=6.0 hfield_params.cell_size=0.05 \
  hfield_params.amplitude=0.3 hfield_params.coarse_n=25 \
  hfield_params.pad_radius=0.55 hfield_params.pad_taper=0.2
cd "$UPSTREAM/training"
PYTHONPATH=. "$PY" -m wojtek_rl.build_model
PYTHONPATH=. "$PY" -m wojtek_rl.build_terrain --arena train --flat-row \
  --pad-radius 0.3 --stair-tread 0.25 0.45 \
  --type-caps pyramid_stairs=0.7,inverted_pyramid_stairs=0.7,discrete_obstacles=0.8,random_grid=0.45
