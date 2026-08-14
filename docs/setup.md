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

## Run

```bash
just train [hydra-overrides]      # plain PPO           (scripts/ppo.py, config "config")
just train-dr [hydra-overrides]   # PPO + domain rand.   (scripts/ppo_dr.py, config "config")
just accel [hydra-overrides]      # ACCEL                (scripts/accel.py, config "accel_config")
just accel-top-k [hydra-overrides] # ACCEL, batched-lanes/top-k (scripts/accel_top_k.py, config "accel_config")
just eval --algo <algo> [args]    # evaluate a checkpoint (scripts/eval.py, pyrallis CLI)
```

`scripts/eval_levels.py` (no `just` recipe yet) evaluates a checkpoint's
generalization across freshly generated heightfield levels instead of one
fixed course; same `--algo`/`--run`/`--step` CLI.

Examples:

```bash
just train training_params.epochs=8 environment_params.name=jump
just train-dr environment_params.envs_num=32
just accel accel_params.replay_prob=0.7 wandb_params.enable=true
just eval --algo accel                    # latest run, latest checkpoint
just eval --algo ppo --run my_run --step 5000
uv run python src/scripts/eval_levels.py --algo accel --num_levels 512
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
