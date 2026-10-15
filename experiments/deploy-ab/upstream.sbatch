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
# no warm start, 50M steps. num_envs 8192 / batch 256 = Playground's Go1
# config (build_ppo_params) and brax_ued's env count; train.sh's 32768 / 1024
# are whole-machine totals and ran out of memory on one H100 (job 6108907).
set -euo pipefail

# PREALLOC=false is mandatory on terrain (upstream train.sh)
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOC=false
export MUJOCO_GL=disabled WANDB_MODE=online
export WANDB_ENTITY=dominikgalus9-wroc-aw-university-of-science-and-technology
UPSTREAM="${UPSTREAM:?set UPSTREAM=<frozen w01-tek copy>}"
# A1: 50M, Playground's 10 evals; A1-long: STEPS=300000000 EVALS=31 RUN=...a1long...
STEPS="${STEPS:-50000000}"
EVALS="${EVALS:-10}"
RUN="${RUN:-deploy_ab_a1_upstream_seed0}"
cd "$UPSTREAM/training"
echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}: upstream $(cat "$UPSTREAM/GIT_COMMIT")"
PYTHONPATH=. "$HOME/workspace/ml-wojtek-ued/.venv/bin/python" -m wojtek_rl.train \
  +experiment=terrain_blind_v4_4d "++ppo.num_envs=8192" "++ppo.batch_size=256" \
  "++ppo.num_timesteps=$STEPS" "++ppo.num_evals=$EVALS" wandb.enable=true \
  wandb.project=wojtyla "run_name=$RUN" seed=0 restore=null
