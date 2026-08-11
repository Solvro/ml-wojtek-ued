# wojtek-ued

Wojtek quadruped: MJX/MJWarp simulation env + procedural terrain generator,
extracted from [w01-tek](https://github.com/) for Unsupervised Environment
Design research (PAIRED, ACCEL, and similar). No ROS, no deployment stack —
just the model, the Brax-compatible env, domain randomization, and a Hydra +
Brax PPO trainer to sanity-check it end to end.

## What's here

- `wojtek_ued/base.py`, `env.py` — `WojtekJoystick`, the MJX env (velocity
  tracking task). `env_getup.py` / `env_jump.py` are two smaller tasks on the
  same base class.
- `wojtek_ued/terrain.py` — deterministic procedural arena generator: a grid
  of tiles, rows of increasing difficulty, one column per terrain type
  (`rough_uniform`, `pyramid_slope`, `inverted_pyramid_slope`,
  `pyramid_stairs`, `inverted_pyramid_stairs`, `discrete_obstacles`,
  `random_grid`, `wave`). One heightfield + static box geoms, seeded and
  fully reproducible from a `TerrainSpec`. This is the hook to plug a UED
  level generator/adversary/mutation operator into — `generate(seed=...,
  difficulties=..., type_caps=..., stair_tread=...)` already takes the shape
  of a level-design parameter vector.
- `wojtek_ued/terrain_env.py`, `terrain_wrapper.py` — arena loading + an
  auto-curriculum Brax wrapper (promotes/demotes each env's difficulty level
  per episode). A starting point for regret-based level replay.
- `wojtek_ued/randomize.py` — Brax-convention domain randomization
  (`randomization_fn`): floor friction, mass, PD gains, DOF, foot friction,
  motor strength.
- `wojtek_ued/registry.py` — `make_env(task, overrides)`, independent of PPO.
- `wojtek_ued/train.py` — Hydra-configured Brax PPO trainer, only needed to
  validate the env; swap it out for a UED training loop once one exists.
- `assets/wojtek_description/` — source MJCF (`wojtek.xml`) + meshes. Never
  edit `wojtek_mjx.xml` / `scene_mjx.xml` / `scene_terrain*.xml` by hand —
  they're generated (gitignored) build outputs.

## Setup

```bash
uv sync
./run.sh build             # wojtek.xml -> wojtek_mjx.xml + scene_mjx.xml
./run.sh check              # sanity-check the generated MJX model
```

## MVP: get PPO training running on a vmapped batch

```bash
# CPU smoke run: tiny batch, ~100k steps, no wandb -- confirms the env
# builds, vmaps, and steps through brax PPO end to end.
./run.sh smoke

# Real GPU run (MJWarp backend auto-selected on CUDA):
./run.sh train task=joystick run_name=smoke_gpu ++ppo.num_envs=2048 wandb.enable=false
```

`./run.sh train --help` is Hydra's own `--help`; inspect a resolved config
without launching anything with `--cfg job --resolve`.

## Terrain training

```bash
./run.sh build-terrain --arena train
./run.sh check-terrain --arena train
./run.sh train +task.env.terrain.enable=true task.env.terrain.arena=train
```

## Deliberately left out

Everything ROS/deployment/navigation-specific from the source repo:
policy export, np_policy runtime, VLM navigation, room scanning, sysid,
course/battery eval harnesses, video rendering. Pull any of those back in
from `w01-tek/training/wojtek_rl` if a later experiment needs them — the
module boundaries (`from wojtek_ued import ...`) are the same shape.
