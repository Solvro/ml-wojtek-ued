# Wojtek UED Research

Unsupervised Environment Design (PAIRED/ACCEL) on the Wojtek quadruped: a PPO
student (brax PPO in `scripts/brax_ued.py`) trained against an adversarial
curriculum of procedurally generated heightfield levels. Built on `wojtek_env`, an extracted MJX simulation env +
procedural terrain generator + Gymnasium interface (env only — no ROS, no
deployment).

## Documentation

Full docs live in [`docs/`](docs/):

| Doc | Read it for |
|---|---|
| [overview.md](docs/overview.md) | architecture, repo map, end-to-end data flow, glossary |
| [setup.md](docs/setup.md) | install, dependency pins, building assets, running |
| [environment.md](docs/environment.md) | `wojtek_env` tasks/terrain/gym, `FunctionalEnv`, the shared tiled arena |
| [ued-accel.md](docs/ued-accel.md) | heightfield levels, the `brax_ued.py` trainer + curriculum modes, eval/render/paper scripts |
| [configuration.md](docs/configuration.md) | `HFieldConfig` fields + arena-builder Hydra overrides |
| [development.md](docs/development.md) | layout rules, tooling, conventions, gotchas |

Start with [overview.md](docs/overview.md).

## Quickstart

```bash
just sync                # uv sync --extra dev
just build               # wojtek.xml -> wojtek_mjx.xml + scene_mjx.xml
just build-accel-arena   # write the tiled "accel" terrain arena (needed by brax_ued.py)
export PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run python src/scripts/brax_ued.py --mode accel --name my_run --checkpoint_dir artifacts/checkpoints
just render-video --checkpoint artifacts/checkpoints/my_run.pkl --level 0.75:3
```

Cluster launchers: `slurm/*.sbatch` (see [ued-accel.md](docs/ued-accel.md#brax-ued)).

## Justfile recipes

`PYTHONPATH=src` and `XLA_PYTHON_CLIENT_PREALLOCATE=false` (warp backend) are
exported for every recipe. `just` lists them all.

| recipe | does |
|---|---|
| `just sync` | `uv sync --extra dev` |
| `just install` | `uv lock && uv sync --extra dev` (relock, then sync) |
| `just fmt` | `ruff format` + `ruff check --fix` on `src` |
| `just lint` | `ruff check` + `ruff format --check` on `src` |
| `just build` | build the MJX model (`wojtek_mjx.xml` + `scene_mjx.xml`) |
| `just build-accel-arena *args` | write the tiled `accel` arena (scene/hfield/spec/lookup); args are `hfield_params.*` Hydra overrides |
| `just render-video *args` | mp4 clips of a `brax_ued.py` checkpoint walking generated levels (`--level D:SEED`, repeatable) |

## Env usage

```python
from wojtek_env.gymnasium import WojtekGymEnv

env = WojtekGymEnv(task="joystick")
obs, info = env.reset(seed=0)
```

`main.py` is a random-agent MuJoCo-viewer preview.
