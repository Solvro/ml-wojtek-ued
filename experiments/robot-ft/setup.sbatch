#!/bin/bash
#SBATCH --job-name=wued-rft-setup
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
# Robot model and both tiled arenas (deploy geometry: training + native eval;
# benchmark: v10box eval) into the frozen copy COPY, then the last checkpoint
# of the upstream base run (upstream_base.sbatch) imported as
# $COPY/artifacts/upstream/robot_base.pkl with the robot profile and the
# fine-tune flags, exported once as a contract check.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
COPY="${COPY:?set COPY=<frozen copy dir>}"
UPSTREAM="${UPSTREAM:?set UPSTREAM=<frozen w01-tek copy with the base run>}"
export PYTHONPATH="$COPY/src"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
RUNS="$UPSTREAM/training/runs/rft_upstream_seed0/checkpoints"
LAST="$RUNS/$(find "$RUNS" -mindepth 1 -maxdepth 1 -name '[0-9]*' -printf '%f\n' | sort -n | tail -1)"
echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}: import $LAST"
cd "$COPY"
"$PY" -m wojtek_env.tools.build_model
"$PY" -m hfield.arena arena=accel hfield_params.patch_radius=3.0 \
  hfield_params.cell_size=0.05 hfield_params.amplitude=0.3 \
  hfield_params.coarse_n=13 hfield_params.pad_radius=0.55 hfield_params.pad_taper=0.2
"$PY" -m hfield.arena arena=accel_deploy hfield_params.patch_radius=6.0 \
  hfield_params.cell_size=0.05 hfield_params.amplitude=0.3 \
  hfield_params.coarse_n=25 hfield_params.pad_radius=0.55 hfield_params.pad_taper=0.2
"$PY" src/scripts/import_upstream.py --checkpoint "$LAST" \
  --out "$COPY/artifacts/upstream/robot_base.pkl" --mode accel --name robot_base \
  --profile robot --level_space param --n_edits 2 --edit_scale 0.1 \
  --clipping_epsilon 0.3
"$PY" src/scripts/export_policy.py --checkpoint artifacts/upstream/robot_base.pkl
