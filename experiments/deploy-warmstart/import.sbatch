#!/bin/bash
#SBATCH --job-name=wued-ws-import
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
# Last checkpoint of upstream_xl.sbatch -> $COPY/artifacts/upstream/a1xl.pkl
# (+ its export), and the v10 "accel" arena that eval.sbatch's v10box uses
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
COPY="${COPY:?set COPY=<frozen copy dir>}"
UPSTREAM="${UPSTREAM:?set UPSTREAM=<frozen w01-tek copy>}"
export PYTHONPATH="$COPY/src"
RUNS="$UPSTREAM/training/runs/deploy_ws_upstream_1p5b_seed0/checkpoints"
LAST="$RUNS/$(find "$RUNS" -mindepth 1 -maxdepth 1 -name '[0-9]*' -printf '%f\n' | sort -n | tail -1)"
echo "job ${SLURM_JOB_ID}: import $LAST"
cd "$HOME/workspace/ml-wojtek-ued"
uv run --no-sync python -m hfield.arena hfield_params.amplitude=0.3 \
  hfield_params.coarse_n=13 hfield_params.pad_radius=0.55 hfield_params.pad_taper=0.2
uv run --no-sync python "$COPY/src/scripts/import_upstream.py" --checkpoint "$LAST" \
  --out "$COPY/artifacts/upstream/a1xl.pkl" --mode accel --name a1xl \
  --profile deploy --level_space param --n_edits 2 --edit_scale 0.1 \
  --clipping_epsilon 0.3
cd "$COPY"
uv run --no-sync --project "$HOME/workspace/ml-wojtek-ued" python \
  src/scripts/export_policy.py --checkpoint artifacts/upstream/a1xl.pkl
