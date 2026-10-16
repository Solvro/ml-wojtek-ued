#!/bin/bash
#SBATCH --job-name=wued-ws-upstream
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=05:00:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
# Continue A1-long (deploy-ab, 305M) with w01-tek's own train.py: same
# preset and sizing, restored from its last checkpoint, 1.2B more steps
# (~1.5B total; brax counts num_timesteps from 0 after a restore).
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOC=false
export MUJOCO_GL=disabled WANDB_MODE=online
export WANDB_ENTITY=dominikgalus9-wroc-aw-university-of-science-and-technology
UPSTREAM="${UPSTREAM:?set UPSTREAM=<frozen w01-tek copy>}"
RESTORE="$UPSTREAM/training/runs/deploy_ab_a1long_upstream_seed0/checkpoints/000304742400"
cd "$UPSTREAM/training"
echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}: restore $RESTORE"
PYTHONPATH=. "$HOME/workspace/ml-wojtek-ued/.venv/bin/python" -m wojtek_rl.train \
  +experiment=terrain_blind_v4_4d "++ppo.num_envs=8192" "++ppo.batch_size=256" \
  "++ppo.num_timesteps=1200000000" "++ppo.num_evals=25" wandb.enable=true \
  wandb.project=wojtyla run_name=deploy_ws_upstream_1p5b_seed0 seed=0 \
  "restore=$RESTORE"
