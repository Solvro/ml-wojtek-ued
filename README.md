# Wojtek UED Research

Wojtek quadruped: MJX/MJWarp simulation env + procedural terrain generator +
a Gymnasium interface. Extracted from w01-tek for UED research (PAIRED,
ACCEL). Env only — no PPO trainer, no ROS, no deployment stack.

```
src/wojtek_env/
  envs/        the MJX env (envs.make_env)
  terrain/     procedural terrain generator + curriculum wrapper
  gymnasium/   WojtekGymEnv
  tools/       build/check CLIs
  paths.py
```

## Setup

```bash
just sync            # uv sync --extra dev
just build           # wojtek.xml -> wojtek_mjx.xml + scene_mjx.xml
```

Other recipes (`just` to list all):

```bash
just install         # relock + sync
just fmt             # ruff format + check --fix
just lint            # ruff check + format --check
just train [args]    # PPO training (hydra overrides, e.g. just train training_params.epochs=8)
```

## Usage

```python
from wojtek_env.gymnasium import WojtekGymEnv

env = WojtekGymEnv(task="joystick")
obs, info = env.reset(seed=0)
```

See `main.py` for a random-agent MuJoCo-viewer preview.
