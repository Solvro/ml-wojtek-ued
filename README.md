# Wojtek UED Research

Unsupervised Environment Design (PAIRED/ACCEL) on the Wojtek quadruped: a PPO
student trained against an adversarial curriculum of procedurally generated
heightfield levels. Built on `wojtek_env`, an extracted MJX simulation env +
procedural terrain generator + Gymnasium interface. Trained policies export
to the robot's `policy.npz` + `policy_meta.json` ([docs/deploy.md](docs/deploy.md)).

## Documentation

Full docs live in [`docs/`](docs/):

| Doc | Read it for |
|---|---|
| [overview.md](docs/overview.md) | architecture, repo map, end-to-end data flow, glossary |
| [setup.md](docs/setup.md) | install, dependency pins, building assets, running |
| [environment.md](docs/environment.md) | `wojtek_env` tasks/terrain/gym, `FunctionalEnv`, the shared tiled arena |
| [ppo.md](docs/ppo.md) | A2C model, rollout, GAE, PPO loss/step, checkpoints, logging |
| [ued-accel.md](docs/ued-accel.md) | heightfield levels + the ACCEL curriculum + convergence |
| [configuration.md](docs/configuration.md) | every Hydra config field + override examples |
| [development.md](docs/development.md) | layout rules, tooling, conventions, gotchas |
| [slurm_recommended.md](docs/slurm_recommended.md) | recommended `sbatch` commands for training + eval |
| [deploy.md](docs/deploy.md) | `brax_ued.py --profile deploy`, checkpoint sidecar, export to the robot's `policy.npz` + `policy_meta.json` |

Start with [overview.md](docs/overview.md).

## Quickstart

```bash
just sync                # uv sync --extra dev
just build               # wojtek.xml -> wojtek_mjx.xml + scene_mjx.xml
just build-accel-arena   # write the "accel" terrain arena (needed for ACCEL; see docs/setup.md#tiled-arena-requirements)
just train               # plain PPO
just train-dr            # plain PPO + baseline domain randomization
just accel               # ACCEL curriculum
just eval algo=ppo       # evaluate the latest checkpoint of an algo
```

## Justfile recipes

`PYTHONPATH=src` is exported for every recipe. `just` lists them all.

| recipe | does |
|---|---|
| `just sync` | `uv sync --extra dev` |
| `just install` | `uv lock && uv sync --extra dev` (relock, then sync) |
| `just fix-mjpython-macos` | macOS: make `mjpython` find libpython (run by `sync`/`install`; [docs](docs/setup.md#mjpython-on-macos)) |
| `just fmt` | `ruff format` + `ruff check --fix` on `src` |
| `just lint` | `ruff check` + `ruff format --check` on `src` |
| `just build` | build the MJX model (`wojtek_mjx.xml` + `scene_mjx.xml`) |
| `just build-accel-arena *args` | write the `accel` arena (scene/hfield/spec/lookup); `hfield_params.*` overrides set its geometry, `arena=<kind>` the file set |
| `just build-deploy-arena` | write the 12x12 m `accel_deploy` arena `brax_ued.py --profile deploy` loads |
| `just train *args` | PPO training; args are Hydra overrides (e.g. `just train training_params.epochs=8`) |
| `just train-dr *args` | PPO training with baseline domain randomization (`scripts/ppo_dr.py`) |
| `just accel *args` | ACCEL training, single-level-per-iteration variant; needs `build-accel-arena` first |
| `just accel-top-k *args` | ACCEL training, batched-lanes/top-k variant; needs `build-accel-arena` first |
| `just eval *args` | evaluate a checkpoint (`algo=` required; `run=`/`step=` default to latest) |
| `just eval-levels *args` | evaluate a checkpoint's generalization across generated levels; needs `build-accel-arena` first |
| `just eval-sim *args` | watch a checkpoint walk random ACCEL levels in the MuJoCo viewer; needs `build-accel-arena` first |
| `just render-video *args` | mp4 clips of a `brax_ued.py` checkpoint walking generated levels ([docs](docs/ued-accel.md#render-video)) |
| `just write-sidecar *args` | backfill the `<run>.json` sidecar of a pre-sidecar `brax_ued.py` checkpoint from its training flags ([docs](docs/deploy.md#sidecar)) |
| `just export *args` | `--checkpoint <run>.pkl [--out dir]`: deploy-profile checkpoint → `policy.npz` + `policy_meta.json` for the robot, validated against the robot runtime ([docs](docs/deploy.md#export)) |
| `just import-upstream *args` | `--checkpoint <w01-tek run>/checkpoints/<step> --out <run>.pkl <brax_ued flags>`: upstream `train.py` checkpoint → brax_ued `.pkl` + sidecar, for export or `--init_from` ([docs](docs/deploy.md#warm-start)) |

## Env usage

```python
from wojtek_env.gymnasium import WojtekGymEnv

env = WojtekGymEnv(task="joystick")
obs, info = env.reset(seed=0)
```

`main.py` is a random-agent MuJoCo-viewer preview.
