import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from wojtek_env.gymnasium import WojtekGymEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="joystick", choices=["joystick", "getup", "jump"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = WojtekGymEnv(task=args.task, seed=args.seed)
    env.reset(seed=args.seed)

    mj_model = env._env.mj_model
    mj_data = mujoco.MjData(mj_model)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()

            mj_data.qpos[:] = np.asarray(env._state.data.qpos)
            mj_data.qvel[:] = np.asarray(env._state.data.qvel)
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            time.sleep(max(0.0, env._env.dt - (time.time() - step_start)))


if __name__ == "__main__":
    main()
