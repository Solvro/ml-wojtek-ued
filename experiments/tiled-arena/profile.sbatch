#!/bin/bash
#SBATCH --job-name=wued-t7-profile
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=02:30:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
# Profile brax_ued's per-iteration cost: per-env hfield swap on jax vs one
# shared level on jax / warp, benchmark and deploy profiles, 8192 envs.
# Builds the robot model and both arenas into COPY first.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda WANDB_MODE=disabled
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$COPY"
"$PY" -m wojtek_env.tools.build_model
"$PY" -m hfield.arena arena=accel hfield_params.patch_radius=3.0 \
  hfield_params.cell_size=0.05 hfield_params.amplitude=0.3 \
  hfield_params.coarse_n=13 hfield_params.pad_radius=0.55 hfield_params.pad_taper=0.2
"$PY" -m hfield.arena arena=accel_deploy hfield_params.patch_radius=6.0 \
  hfield_params.cell_size=0.05 hfield_params.amplitude=0.3 \
  hfield_params.coarse_n=25 hfield_params.pad_radius=0.55 hfield_params.pad_taper=0.2
for profile in benchmark deploy; do
  for variant in shared_warp shared_jax cur_jax; do
    echo "=== $profile $variant"
    "$PY" "$COPY/src/scripts/profile_brax_ued.py" --variant "$variant" \
      --profile "$profile" 2>&1 | grep -v -i warning || echo "FAILED $profile $variant"
  done
done
