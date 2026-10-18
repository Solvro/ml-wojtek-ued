# Setup

## Requirements

- Python **3.11–3.12** (`requires-python = ">=3.11,<3.13"`). Not 3.13 — the
  jax/mjx pins below do not resolve there yet.
- `uv` for the environment. CUDA extra (`jax[cuda12]`) installs only on Linux;
  macOS gets CPU jax.

## Pinned dependencies (do not bump blindly)

These three are a proven, interdependent set — the comments in `pyproject.toml`
carry the reasons:

- `jax == 0.9.2` — proven pin for this driver family.
- `mujoco-mjx == 3.10.0` — 3.11.0 needs a newer warp than we pin; 3.10.0 is the
  proven pair.
- `warp-lang == 1.13.0` — mjx 3.10.0 vendors MJWarp and requires exactly this;
  1.14 removed `warp._src.jax_experimental` and breaks `put_model(impl="warp")`.

Everything else (`brax`, `equinox`, `distrax`, `flax`, `hydra-core`, `wandb`,
`gymnasium`, `playground`, `ml-collections`) floats on `>=`.

## Install

```bash
just sync       # uv sync --extra dev   (installs pytest + ruff dev extras)
just install    # uv lock && uv sync    (relock first, then sync)
```

## Build generated assets

The MJX model and terrain arenas are **generated and gitignored** — build them
before running anything that loads the robot or a terrain scene.

```bash
just build               # wojtek.xml -> wojtek_mjx.xml + scene_mjx.xml
just build-accel-arena   # write the "accel" arena (scene/hfield/spec/lookup)
```

`build-accel-arena` is required before `brax_ued.py` and the eval scripts — they load the
`accel` terrain arena. See [environment.md](environment.md#assets) for what each
file is and why they must sit next to the robot XML.

## mjpython on macOS

`mujoco.viewer.launch_passive` (used by
`wojtek_env/tools/check_model_mjx.py`) requires running under `mjpython` on
macOS, not plain `python` — its launcher dlopens `.venv/bin/python` to hop
the interpreter onto the main thread for Cocoa. `uv`'s relocatable CPython
isn't a framework build, so out of the box that dlopen fails with `Library
not loaded: @rpath/libpython3.12.dylib`. `just sync`/`just install` run
`just fix-mjpython-macos` (no-op on non-macOS) to symlink the real
`libpythonX.Y.dylib` into `.venv/` and the `mujoco` package's
`MuJoCo_(mjpython).app` bundle, which resolves it. Re-run `just
fix-mjpython-macos` manually if `.venv` was rebuilt outside `just
sync`/`install`.

## Run

```bash
export PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python src/scripts/brax_ued.py --mode accel --name my_run \
  --checkpoint_dir artifacts/checkpoints                                  # train (dr | accel | plr | paired)
uv run python src/scripts/eval_brax.py --checkpoint artifacts/checkpoints/my_run.pkl --name my_eval
just render-video --checkpoint artifacts/checkpoints/my_run.pkl --level 0.75:3
```

All scripts take argparse flags (`--help`); the full set, with the cluster
launchers (`slurm/*.sbatch`), is in [ued-accel.md](ued-accel.md#brax-ued).

## Import root

`PYTHONPATH=src` is exported by the `justfile`, so research code imports are
top-level (`from hfield.tiles import ...`, `from utils.env import ...`). Run
scripts through `just` (or export `PYTHONPATH=src` yourself) so these resolve.

## What lands where

Gitignored outputs (`.gitignore`): `artifacts/` (`brax_ued.py` checkpoints),
`wandb/`, `runs/`, `outputs/` + `multirun/` (Hydra), and the generated `*_mjx.xml` / `terrain*` assets.
