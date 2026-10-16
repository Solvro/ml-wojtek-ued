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
ALGO="${ALGO:-ppo}"     # ppo | ppo_dr | ppo_dr_terrain | accel | accel_top_k

case "$ALGO" in
    ppo)             TRAIN_SCRIPT="src/scripts/ppo.py" ;;
    ppo_dr)          TRAIN_SCRIPT="src/scripts/ppo_dr.py" ;;
    ppo_dr_terrain)  TRAIN_SCRIPT="src/scripts/ppo_dr_terrain.py" ;;
    accel)           TRAIN_SCRIPT="src/scripts/accel.py" ;;
    accel_top_k)     TRAIN_SCRIPT="src/scripts/accel_top_k.py" ;;
    *) echo "Unknown ALGO: $ALGO (ppo|ppo_dr|ppo_dr_terrain|accel|accel_top_k)" >&2; exit 1 ;;
esac

case "$ALGO" in
    ppo|ppo_dr|ppo_dr_terrain) DEFAULT_TRAIN_ARGS="environment_params.envs_num=4096 training_params.iterations=5000000" ;;
    accel|accel_top_k)         DEFAULT_TRAIN_ARGS="environment_params.envs_num=4096" ;;
esac
TRAIN_ARGS="${TRAIN_ARGS:-$DEFAULT_TRAIN_ARGS}"
EVAL_ARGS="${EVAL_ARGS:-}"

# hfield_params.* overrides (e.g. hfield_params.amplitude=0.30), shared by the
# arena build and every script that reads terrain difficulty, so the compiled
# arena's physical envelope can never drift out of sync with what the
# training/eval script thinks it's using.
HFIELD_ARGS="${HFIELD_ARGS:-}"
EVAL_ARGS="$EVAL_ARGS $HFIELD_ARGS"
case "$ALGO" in
    ppo_dr_terrain|accel|accel_top_k) TRAIN_ARGS="$TRAIN_ARGS $HFIELD_ARGS" ;;
esac

NEED_ARENA=0
[ "$MODE" = "eval" ] && NEED_ARENA=1
case "$ALGO" in ppo_dr_terrain|accel|accel_top_k) NEED_ARENA=1 ;; esac

echo "MODE=$MODE ALGO=$ALGO NEED_ARENA=$NEED_ARENA"
echo "TRAIN_ARGS=$TRAIN_ARGS"
echo "EVAL_ARGS=$EVAL_ARGS"
echo "HFIELD_ARGS=$HFIELD_ARGS"

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
# warp allocates its collision scratch outside JAX's pool (src/utils/env.py)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=""
export WANDB_API_KEY=""

python3 -m venv $TMPDIR/.venv

source $TMPDIR/.venv/bin/activate

pip install --upgrade pip

pip install -e .

python3 -m wojtek_env.tools.build_model
[ "$NEED_ARENA" = "1" ] && python3 -m hfield.arena $HFIELD_ARGS

if [ "$MODE" = "eval" ]; then
    python3 src/scripts/eval_levels.py algo="$ALGO" $EVAL_ARGS
else
    python3 "$TRAIN_SCRIPT" $TRAIN_ARGS
fi
