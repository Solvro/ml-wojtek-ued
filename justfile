export PYTHONPATH := "src"

# List available recipes
default:
    @just --list

# Sync the environment to the lockfile (incl. dev extras)
sync:
    uv sync --extra dev

# Resolve and update the lockfile, then sync
install:
    uv lock
    uv sync --extra dev

# Format the codebase with ruff
fmt:
    uv run ruff format src
    uv run ruff check --fix src

# Lint and check formatting without writing changes
lint:
    uv run ruff check src
    uv run ruff format --check src

# Build the MJX model (wojtek_mjx.xml + scene_mjx.xml) from wojtek.xml
build:
    uv run python -m wojtek_env.tools.build_model

# Run the PPO training script (pass hydra overrides, e.g. `just train training_params.epochs=8`)
train *args:
    uv run python src/scripts/ppo.py {{ args }}
