# Setup

## Requirements

- Python **3.11–3.12** (`requires-python = ">=3.11,<3.13"`). Not 3.13 — the
  jax/mjx pins below do not resolve there yet.
- [uv](https://docs.astral.sh/uv/) for the environment, [just](https://just.systems/)
  for the recipes, `ffmpeg` on `PATH` for `render-video`. `.python-version`
  pins 3.12.
- Training and evals run on Linux x86_64 with one NVIDIA GPU per run and a
  CUDA 12 driver (the `jax[cuda12]` extra installs only on Linux; H100 used).
  macOS gets CPU jax: tests, CPU smoke runs, rendering, tables and figures.
- `uv.lock` pins every version; `uv sync --locked --extra dev` reproduces it
  exactly (the paper's environment): jax/jaxlib 0.9.2, mujoco 3.11.0,
  mujoco-mjx 3.10.0, warp-lang 1.13.0, brax 0.14.2, playground 0.2.0.
- A wandb account: training, eval and the paper dumps log to / read from
  `entity` / `project` ([configuration.md](configuration.md#trainyaml--trainconfig)).

## Pinned dependencies (do not bump blindly)

These three are a proven, interdependent set — the comments in `pyproject.toml`
carry the reasons:

- `jax == 0.9.2` — proven pin for this driver family.
- `mujoco-mjx == 3.10.0` — 3.11.0 needs a newer warp than we pin; 3.10.0 is the
  proven pair.
- `warp-lang == 1.13.0` — mjx 3.10.0 vendors MJWarp and requires exactly this;
  1.14 removed `warp._src.jax_experimental` and breaks `put_model(impl="warp")`.

`onnx == 1.23.2` and `onnxruntime == 1.31.0` (policy export,
[export.md](export.md)) are pinned exactly: `onnx` builds and checks the
graph against opset 17, `onnxruntime` runs the export-time check and is
what the robot runs, so a drifting pair would change the shipped contract.

Everything else (`brax`, `playground`, `equinox`, `distrax`, `optax`,
`hydra-core`, `wandb`, `gymnasium`, `ml-collections`, `matplotlib`,
`mediapy`) floats on `>=` in `pyproject.toml`; the lockfile fixes the
versions. Only directly imported packages are declared (`flax` comes with
brax).

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

The Playground robots (`go1`, `barkour`, `spot`, `go2`) also need
`mujoco_menagerie` (meshes) at Playground's pinned commit, cloned into the
venv:

```bash
uv run python -c "from mujoco_playground._src.mjx_env import \
ensure_menagerie_exists; ensure_menagerie_exists()"
```

On SLURM, `slurm/build_assets.sbatch` does all three in a frozen copy
([slurm.md](slurm.md#frozen-copy)).

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
just train mode=accel name=my_run checkpoint_dir=artifacts/checkpoints  # mode: conf/mode/
just eval checkpoint=artifacts/checkpoints/my_run.pkl name=my_eval out_dir=artifacts/evals
just render-video checkpoint=artifacts/checkpoints/my_run.pkl 'levels=[0.75:3]'
```

Every script is configured by Hydra from `conf/` (`key=value` overrides,
`--cfg job` prints the composed config): [configuration.md](configuration.md).
Cluster launchers (`slurm/*.sbatch`): [slurm.md](slurm.md); the full paper
grid: [README](../README.md#reproduce-the-paper).

## Import root

`PYTHONPATH=src` is exported by the `justfile`, so research code imports are
top-level (`from hfield.tiles import ...`, `from env.functional import ...`). Run
scripts through `just` (or export `PYTHONPATH=src` yourself) so these resolve.

## What lands where

Gitignored outputs (`.gitignore`): `artifacts/` (checkpoints, videos, paper
outputs, and `artifacts/hydra/` = one dir per run with its composed
`.hydra/config.yaml`), `wandb/`, `runs/`, `multirun/` (Hydra `-m` sweeps),
and the generated `*_mjx.xml` / `terrain*` assets.
