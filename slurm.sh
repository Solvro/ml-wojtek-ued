#!/bin/bash -l
#SBATCH --job-name=ued-task
#SBATCH --account=hpc-danbor2008-1756464546
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:hopper:1,storage:lustre:500G
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --partition=lem-gpu-short
#SBATCH --time=3:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --extra=FORCE_RM_TMPDIR

source /etc/profile

module load Python/3.12.3-GCCcore-13.3.0

ROOT_DIR="path/to/where/ml-wojtek-ued/is"
cd "$ROOT_DIR"

#   sbatch --export=ALL,ALGO=accel slurm.sh
#   sbatch --export=ALL,MODE=eval,ALGO=accel slurm.sh
MODE="${MODE:-train}"   # train | eval
ALGO="${ALGO:-ppo}"     # ppo | ppo_dr | accel | accel_top_k

case "$ALGO" in
    ppo)          TRAIN_SCRIPT="src/scripts/ppo.py" ;;
    ppo_dr)       TRAIN_SCRIPT="src/scripts/ppo_dr.py" ;;
    accel)        TRAIN_SCRIPT="src/scripts/accel.py" ;;
    accel_top_k)  TRAIN_SCRIPT="src/scripts/accel_top_k.py" ;;
    *) echo "Unknown ALGO: $ALGO (ppo|ppo_dr|accel|accel_top_k)" >&2; exit 1 ;;
esac

case "$ALGO" in
    ppo|ppo_dr)        DEFAULT_TRAIN_ARGS="environment_params.envs_num=4096 training_params.iterations=5000000" ;;
    accel|accel_top_k) DEFAULT_TRAIN_ARGS="environment_params.envs_num=4096" ;;
esac
TRAIN_ARGS="${TRAIN_ARGS:-$DEFAULT_TRAIN_ARGS}"
EVAL_ARGS="${EVAL_ARGS:-}"

NEED_ARENA=0
[ "$MODE" = "eval" ] && NEED_ARENA=1
case "$ALGO" in accel|accel_top_k) NEED_ARENA=1 ;; esac

DONE_MARKER="$ROOT_DIR/artifacts/DONE"

export PIP_CACHE_DIR=$TMPDIR/pip-cache
export PYTHONPYCACHEPREFIX=$TMPDIR/python-cache

CACHE_ROOT=$TMPDIR
mkdir -p "$CACHE_ROOT"/{hf/hub,hf/transformers,triton}

export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export HYDRA_FULL_ERROR=1 

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=""
export WANDB_API_KEY=""

python3 -m venv $TMPDIR/.venv

source $TMPDIR/.venv/bin/activate

pip install --upgrade pip

pip install -e .

python3 -m wojtek_env.tools.build_model
[ "$NEED_ARENA" = "1" ] && python3 -m hfield.arena

if [ "$MODE" = "eval" ]; then
    python3 src/scripts/eval_levels.py algo="$ALGO" $EVAL_ARGS
else
    python3 "$TRAIN_SCRIPT" $TRAIN_ARGS
fi
