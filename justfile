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

# Write the "accel" terrain arena's static assets (scene/hfield/spec/lookup); args are Hydra overrides of conf/arena.yaml (default arena=v10)
[group('build')]
build-accel-arena *args:
    uv run python -m hfield.arena {{ args }}

# --- training ---

# Train with brax_ued.py; args are Hydra overrides of conf/train.yaml (e.g. `just train mode=accel name=my_run`)
[group('train')]
train *args:
    uv run python src/scripts/brax_ued.py {{ args }}

# Throughput profile of a brax_ued.py iteration (conf/profile.yaml, e.g. `just profile backend=warp`)
[group('train')]
profile *args:
    uv run python src/scripts/profile_brax_ued.py {{ args }}

# --- evaluation ---

# Final eval of a checkpoint (conf/eval.yaml, e.g. `just eval checkpoint=run.pkl name=run_eval out_dir=evals`)
[group('eval')]
eval *args:
    uv run python src/scripts/eval_brax.py {{ args }}

# The checkpoint on CPU MuJoCo physics variants (conf/sim2sim.yaml)
[group('eval')]
sim2sim *args:
    uv run python src/scripts/eval_sim2sim.py {{ args }}

# mp4 clips of a brax_ued.py checkpoint on generated levels (conf/render.yaml, e.g. `just render-video checkpoint=run.pkl levels=[0.75:3]`); MUJOCO_GL defaults to cgl on macOS, egl elsewhere
[group('eval')]
render-video *args:
    MUJOCO_GL="${MUJOCO_GL:-$([ "$(uname)" = Darwin ] && echo cgl || echo egl)}" uv run python src/scripts/render_video.py {{ args }}

# --- paper ---

# LaTeX tables from eval_brax.py outputs (conf/paper_tables.yaml)
[group('paper')]
paper-tables *args:
    uv run python src/scripts/paper_tables.py {{ args }}

# Figures from wandb histories and rendered terrains (conf/paper_figures.yaml); MUJOCO_GL as in render-video
[group('paper')]
paper-figures *args:
    MUJOCO_GL="${MUJOCO_GL:-$([ "$(uname)" = Darwin ] && echo cgl || echo egl)}" uv run python src/scripts/paper_figures.py {{ args }}
