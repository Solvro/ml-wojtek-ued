# Wojtek UED Research

Wojtek quadruped: MJX/MJWarp simulation env + procedural terrain generator +
a Gymnasium interface. Extracted from w01-tek for UED research (PAIRED,
ACCEL). Env only — no PPO trainer, no ROS, no deployment stack.

```
src/wojtek_env/
  envs/        the MJX env (envs.make_env)
  terrain/     procedural terrain generator + curriculum wrapper
  gymnasium/   WojtekGymEnv, WojtekVectorGymEnv
  tools/       build/check CLIs
  paths.py
```

## Setup

```bash
uv sync
./run.sh build       # wojtek.xml -> wojtek_mjx.xml + scene_mjx.xml
./run.sh check-gym   # sanity-check both Gymnasium wrappers
```

## Usage

```python
from wojtek_env.gymnasium import WojtekGymEnv, WojtekVectorGymEnv

env = WojtekGymEnv(task="joystick")
obs, info = env.reset(seed=0)

envs = WojtekVectorGymEnv(num_envs=2048, task="joystick")
obs, infos = envs.reset(seed=0)
```

See `main.py` for a random-agent MuJoCo-viewer preview.
