# wojtek-env

Wojtek quadruped: MJX/MJWarp simulation env + procedural terrain generator +
a Gymnasium interface, extracted from [w01-tek](https://github.com/) for
Unsupervised Environment Design research (PAIRED, ACCEL, and similar). Env
only — no PPO trainer, no ROS, no deployment stack.

```
src/wojtek_env/
  base.py, env.py, env_getup.py, env_jump.py   # the MJX env (registry.make_env)
  terrain.py, terrain_env.py, terrain_wrapper.py  # procedural terrain + curriculum
  randomize.py                                  # domain randomization
  gymnasium/                                    # the Gymnasium interface
    env.py         WojtekGymEnv        (single instance)
    vector_env.py  WojtekVectorGymEnv  (batched, vmapped)
  build_model.py, build_terrain.py, check_*.py  # setup + validation CLIs
```

## What's here

- `wojtek_env/base.py`, `env.py` — `WojtekJoystick`, the MJX env (velocity
  tracking task). `env_getup.py` / `env_jump.py` are two smaller tasks on the
  same base class.
- `wojtek_env/terrain.py` — deterministic procedural arena generator: a grid
  of tiles, rows of increasing difficulty, one column per terrain type
  (`rough_uniform`, `pyramid_slope`, `inverted_pyramid_slope`,
  `pyramid_stairs`, `inverted_pyramid_stairs`, `discrete_obstacles`,
  `random_grid`, `wave`). One heightfield + static box geoms, seeded and
  fully reproducible from a `TerrainSpec`. This is the hook to plug a UED
  level generator/adversary/mutation operator into — `generate(seed=...,
  difficulties=..., type_caps=..., stair_tread=...)` already takes the shape
  of a level-design parameter vector.
- `wojtek_env/terrain_env.py`, `terrain_wrapper.py` — arena loading + an
  auto-curriculum Brax wrapper (promotes/demotes each env's difficulty level
  per episode). A starting point for regret-based level replay.
- `wojtek_env/randomize.py` — Brax-convention domain randomization
  (`randomization_fn`): floor friction, mass, PD gains, DOF, foot friction,
  motor strength.
- `wojtek_env/registry.py` — `make_env(task, overrides)`, the plain
  functional entry point every wrapper below is built on.
- `wojtek_env/gymnasium/` — the Gymnasium interface, split so it's easy to
  extend on its own (add wrappers, more surfaces, without touching the env):
  - `env.py: WojtekGymEnv(gymnasium.Env)` — single instance. No auto-reset
    (the modern Gymnasium contract): `step()` after termination/truncation
    raises until `reset()` is called again.
  - `vector_env.py: WojtekVectorGymEnv(gymnasium.vector.VectorEnv)` —
    batched, vmapped through the same brax wrapper stack the original PPO
    trainer used. Auto-resets sub-envs on done
    (`metadata["autoreset_mode"] = SAME_STEP`): the observation on a done
    step is already the next episode's first observation. The true terminal
    observation is not separately exposed.
  - Observation is the actor's flat `"state"` vector; the critic-only
    `"privileged_state"` rides along under `info["privileged_obs"]`.
- `assets/wojtek_description/` — source MJCF (`wojtek.xml`) + meshes. Never
  edit `wojtek_mjx.xml` / `scene_mjx.xml` / `scene_terrain*.xml` by hand —
  they're generated (gitignored) build outputs.

## Setup

```bash
uv sync
./run.sh build             # wojtek.xml -> wojtek_mjx.xml + scene_mjx.xml
./run.sh check              # sanity-check the generated MJX model
./run.sh check-gym          # build both Gymnasium wrappers, roll out a few steps
```

## Usage

```python
from wojtek_env.gymnasium import WojtekGymEnv, WojtekVectorGymEnv

# Single instance, classic Gymnasium loop.
env = WojtekGymEnv(task="joystick")
obs, info = env.reset(seed=0)
for _ in range(1000):
    action = env.action_space.sample()          # Box(-1, 1, (12,))
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

# Batched: gymnasium.vector.VectorEnv, vmapped on device.
envs = WojtekVectorGymEnv(num_envs=2048, task="joystick", domain_randomize=True)
obs, infos = envs.reset(seed=0)
actions = envs.action_space.sample()
obs, rewards, terminations, truncations, infos = envs.step(actions)
```

`task` is `joystick` (velocity tracking, terrain-capable), `getup`, or
`jump`. `env_overrides` is a plain nested dict matching each task's
`default_config()` in `env.py`/`env_getup.py`/`env_jump.py` — e.g.
`env_overrides={"terrain": {"enable": True, "arena": "train"}}` after
`./run.sh build-terrain --arena train`.

## Deliberately left out

Everything PPO/ROS/deployment/navigation-specific from the source repo: the
Hydra+Brax PPO trainer, hydra/wandb config plumbing, policy export, np_policy
runtime, VLM navigation, room scanning, sysid, course/battery eval harnesses,
video rendering. Pull any of those back in from `w01-tek/training/wojtek_rl`
if a later experiment needs them — the module boundaries (`from wojtek_env
import ...`) are the same shape.
