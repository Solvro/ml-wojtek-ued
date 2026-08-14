export PYTHONPATH := "src"

# List available recipes
default:
    @just --list

# --- environment ---

# Sync the environment to the lockfile (incl. dev extras)
[group('env')]
sync:
    uv sync --extra dev

# Resolve and update the lockfile, then sync
[group('env')]
install:
    uv lock
    uv sync --extra dev

# --- code quality ---

# Format the codebase with ruff
[group('quality')]
fmt:
    uv run ruff format src
    uv run ruff check --fix src

# Lint and check formatting without writing changes
[group('quality')]
lint:
    uv run ruff check src
    uv run ruff format --check src

# --- build ---

# Build the MJX model (wojtek_mjx.xml + scene_mjx.xml) from wojtek.xml
[group('build')]
build:
    uv run python -m wojtek_env.tools.build_model

# Write the "accel" terrain arena's static assets (scene/hfield/spec/lookup)
[group('build')]
build-accel-arena:
    uv run python -m hfield.arena

# --- training ---

# Run the PPO training script (pass hydra overrides, e.g. `just train training_params.epochs=8`)
[group('train')]
train *args:
    uv run python src/scripts/ppo.py {{ args }}

# Run PPO with baseline domain randomization (pass hydra overrides)
[group('train')]
train-dr *args:
    uv run python src/scripts/ppo_dr.py {{ args }}

# Run the ACCEL training script (pass hydra overrides); needs build-accel-arena first
[group('train')]
accel *args:
    uv run python src/scripts/accel.py {{ args }}

# Run the batched-lanes/top-k ACCEL variant (pass hydra overrides); needs build-accel-arena first
[group('train')]
accel-top-k *args:
    uv run python src/scripts/accel_top_k.py {{ args }}

# --- evaluation ---

# Evaluate a checkpoint (pass hydra overrides, e.g. `just eval algo=ppo run=my_run step=5000`)
[group('eval')]
eval *args:
    uv run python src/scripts/eval.py {{ args }}

# Evaluate a checkpoint's generalization across generated levels; needs build-accel-arena first
[group('eval')]
eval-levels *args:
    uv run python src/scripts/eval_levels.py {{ args }}
