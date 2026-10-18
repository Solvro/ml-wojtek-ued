export PYTHONPATH := "src"
# warp (sim.backend auto on a GPU) allocates outside JAX's pool; utils/env.py
# refuses warp without this
export XLA_PYTHON_CLIENT_PREALLOCATE := "false"

# List available recipes
default:
    @just --list

# --- environment ---

# Sync the environment to the lockfile (incl. dev extras)
[group('env')]
sync:
    uv sync --extra dev
    just fix-mjpython-macos

# Resolve and update the lockfile, then sync
[group('env')]
install:
    uv lock
    uv sync --extra dev
    just fix-mjpython-macos

# macOS only, no-op elsewhere: symlink libpythonX.Y.dylib into the venv root
# and the mjpython.app bundle. uv's relocatable CPython isn't a framework
# build, so mjpython's launcher (which dlopens .venv/bin/python to hop onto
# the main thread for Cocoa) can't resolve @rpath/libpython there out of the
# box -- see docs/setup.md#mjpython-on-macos. `uv sync` recreates .venv, so
# this reruns on every sync/install; it's a no-op if the symlinks already
# point at the right target.
[group('env')]
fix-mjpython-macos:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "$(uname)" != "Darwin" ]; then
      exit 0
    fi
    app_glob=(.venv/lib/python*/site-packages/mujoco/MuJoCo_\(mjpython\).app)
    if [ ! -d "${app_glob[0]}" ]; then
      exit 0
    fi
    real_python="$(uv run python3 -c 'import os, sys; print(os.path.realpath(sys.executable))')"
    py_version="$(uv run python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    real_lib="$(dirname "$real_python")/../lib/libpython${py_version}.dylib"
    if [ ! -f "$real_lib" ]; then
      echo "warning: $real_lib not found, skipping mjpython fixup" >&2
      exit 0
    fi
    app_lib_dir="${app_glob[0]}/Contents/lib"
    mkdir -p "$app_lib_dir"
    ln -sf "$real_lib" "$app_lib_dir/libpython${py_version}.dylib"
    ln -sf "$real_lib" ".venv/libpython${py_version}.dylib"

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
build-accel-arena *args:
    uv run python -m hfield.arena {{ args }}

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

# Watch a checkpoint walk random ACCEL levels in the MuJoCo viewer (pass argparse flags, e.g. `just eval-sim --checkpoint artifacts/accel_checkpoint`)
[group('eval')]
eval-sim *args:
    uv run mjpython src/scripts/eval_sim.py {{ args }}
