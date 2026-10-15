#!/bin/bash
#SBATCH --job-name=wued-t7-check
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
# Tiled arena on the GPU: flat/pit tiles on jax and warp with the max
# base-z gap between them, then profile_brax_ued.py (8192 envs) for both
# profiles on warp and the benchmark profile on jax.
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 JAX_PLATFORMS=cuda WANDB_MODE=disabled
COPY="${COPY:?set COPY=<frozen copy dir>}"
export PYTHONPATH="$COPY/src"
PY="$HOME/workspace/ml-wojtek-ued/.venv/bin/python"
echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$COPY"
status=0
"$PY" src/scripts/check_hfield_batch.py --compare --tiles 16 --envs_per_tile 4 \
  --steps 150 || status=1
for run in "warp benchmark" "warp deploy" "jax benchmark"; do
  read -r backend profile <<<"$run"
  echo "=== profile $backend $profile"
  "$PY" src/scripts/profile_brax_ued.py --backend "$backend" --profile "$profile" ||
    status=1
done
exit "$status"
