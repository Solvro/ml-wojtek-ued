#!/bin/bash
#SBATCH --job-name=wued-ab-upstream
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
# A1 of experiments/deploy-ab: w01-tek's own trainer, terrain_blind_v4_4d,
# no warm start, 50M steps; num_envs/batch = training/jobs/train.sh defaults.
set -euo pipefail

# PREALLOC=false is mandatory on terrain (upstream train.sh)
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOC=false
export MUJOCO_GL=disabled WANDB_MODE=online
export WANDB_ENTITY=dominikgalus9-wroc-aw-university-of-science-and-technology
UPSTREAM="${UPSTREAM:?set UPSTREAM=<frozen w01-tek copy>}"
cd "$UPSTREAM/training"
echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}: upstream $(cat "$UPSTREAM/GIT_COMMIT")"
PYTHONPATH=. "$HOME/workspace/ml-wojtek-ued/.venv/bin/python" -m wojtek_rl.train \
  +experiment=terrain_blind_v4_4d "++ppo.num_envs=32768" "++ppo.batch_size=1024" \
  "++ppo.num_timesteps=50000000" wandb.enable=true wandb.project=wojtyla \
  run_name=deploy_ab_a1_upstream_seed0 seed=0 restore=null
