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
`orbax`, `gymnasium`, `playground`, `ml-collections`) floats on `>=`.

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

`build-accel-arena` is required before `just accel` — the ACCEL env loads the
`accel` terrain arena. See [environment.md](environment.md#assets) for what each
file is and why they must sit next to the robot XML.

## Tiled arena / warp requirements {#tiled-arena-requirements}

Every ACCEL/DR level of an iteration sits on one shared tiled arena, simulated
on warp when a GPU is present ([environment.md](environment.md#tiled-arena)).

- `XLA_PYTHON_CLIENT_PREALLOCATE=false`: warp allocates its collision scratch
  outside JAX's pool; `FunctionalEnv` refuses warp without it. The `justfile`,
  `slurm.sh`, `slurm/brax_ued.sbatch` and `slurm/eval_brax.sbatch` export it;
  set it yourself when running a script outside `just`. (`XLA_PYTHON_CLIENT_PREALLOC` is not read by JAX.)
- Rebuild arenas after any change to the geometry: `patch_radius`,
  `cell_size`, `amplitude`, `base_z`, `tiles`, `reach`, `pad_radius`/
  `pad_taper`, or the arena kind. `hfield.arena.check_arena` fails with the
  exact command when the arena differs from the run's.
  - `accel` (benchmark): `just build-accel-arena [hfield_params.*]`
  - `accel_deploy` (`--profile deploy|robot`): `just build-deploy-arena`
  - arenas are gitignored; on WCSS `slurm.sh` builds the `accel` arena
    (`HFIELD_ARGS`, [slurm_recommended.md](slurm_recommended.md)), the
    `brax_ued.sbatch` array needs its arena built beforehand.

## mjpython on macOS

`mujoco.viewer.launch_passive` (used by `just eval-sim` and
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
just train [hydra-overrides]      # plain PPO           (scripts/ppo.py, config "config")
just train-dr [hydra-overrides]   # PPO + domain rand.   (scripts/ppo_dr.py, config "config")
just accel [hydra-overrides]      # ACCEL                (scripts/accel.py, config "accel_config")
just accel-top-k [hydra-overrides] # ACCEL, batched-lanes/top-k (scripts/accel_top_k.py, config "accel_config")
just eval algo=<algo> [overrides] # evaluate a checkpoint (scripts/eval.py, config "eval_config")
just eval-levels algo=<algo> [overrides] # generalization across generated levels (scripts/eval_levels.py, config "eval_levels_config"); needs build-accel-arena
```

`eval-levels` evaluates a checkpoint across freshly generated heightfield
levels instead of one fixed course; same `algo`/`run`/`step` overrides.

Examples:

```bash
just train training_params.epochs=8 environment_params.name=jump
just train-dr environment_params.envs_num=32
just accel accel_params.replay_prob=0.7 wandb_params.enable=true
just eval algo=accel                      # latest run, latest checkpoint
just eval algo=ppo run=my_run step=5000
just eval-levels algo=accel num_levels=512
```

Overrides are Hydra dotted paths into the config dataclasses —
see [configuration.md](configuration.md).

## Import root

`PYTHONPATH=src` is exported by the `justfile`, so research code imports are
top-level (`from ued.curriculum import ...`, `from utils.env import ...`). Run
scripts through `just` (or export `PYTHONPATH=src` yourself) so these resolve.

## What lands where

Gitignored outputs (`.gitignore`): `artifacts/` (orbax checkpoints, laid out as
`artifacts/<algo>/<run>/`, see [ppo.md](ppo.md#checkpoints--utilscheckpointpy)),
`wandb/`, `logs/` (jsonl fallback), `runs/`, `outputs/` + `multirun/` (Hydra),
and the generated `*_mjx.xml` / `terrain*` / `hfield_playground_*` assets.
