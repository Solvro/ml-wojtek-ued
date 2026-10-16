#!/bin/bash
#SBATCH --job-name=wued-rft-base
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --partition=lem-gpu-short
#SBATCH --gres=gpu:hopper:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96gb
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
#SBATCH --error=/lustre/pd03/hpc-danbor2008-1756464546/jakgil6519/wojtek-ued/logs/%x_%j.out
# The robot base: w01-tek's own train.py with brax_ued.UPSTREAM_RUNS["robot"]
# (flat_quiet_v6 at the RPi policy's kp 40 / kd 0.8 plant + the base_accel
# dose of wojtek_flat_quiet_baseacc_s0), from scratch, to the preset's 2B
# budget; 8192 envs on warp. num_evals only sets the checkpoint cadence
# (one per 80M steps).
set -euo pipefail

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=disabled WANDB_MODE=online
export WANDB_ENTITY=dominikgalus9-wroc-aw-university-of-science-and-technology
UPSTREAM="${UPSTREAM:?set UPSTREAM=<frozen w01-tek copy>}"
cd "$UPSTREAM/training"
echo "job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}: robot base from scratch"
PYTHONPATH=. "$HOME/workspace/ml-wojtek-ued/.venv/bin/python" -m wojtek_rl.train \
  +experiment=flat_quiet_v6 ++task.env.pd_kp=40.0 ++task.env.pd_kd=0.8 \
  ++task.env.reward.scales.base_accel=-0.02 \
  ++task.env.reward.base_accel_wz_fade=0.1 \
  ++task.env.reward.base_accel_wz_floor=0.35 \
  "++ppo.num_envs=8192" "++ppo.num_evals=25" \
  wandb.enable=true wandb.project=wojtyla run_name=rft_upstream_seed0 seed=0
