import argparse
import time
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import mujoco
import mujoco.viewer
import numpy as np
import orbax.checkpoint as ocp

from config.hfield_config import HFieldConfig
from hfield.manager import HFieldManager
from modules.ppo_modules import A2C
from ued.levels import level_to_hfield_data
from utils.env import FunctionalEnv
from utils.env_overrides import accel_arena_overrides, merge_overrides


def restore_model(model_dir: Path, template: A2C) -> A2C:
    array_template, static = eqx.partition(template, eqx.is_array)
    restored = ocp.StandardCheckpointer().restore(
        model_dir.resolve(), target=array_template
    )
    return eqx.combine(restored, static)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="artifacts/accel_checkpoint")
    p.add_argument("--task", default="joystick", choices=["joystick", "getup", "jump"])
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    p.add_argument("--activation", default="elu", choices=["relu", "tanh", "elu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="0 = run forever, new level each episode",
    )
    p.add_argument("--steps_in_episode", type=int, default=1000)
    args = p.parse_args()

    hfield_config = HFieldConfig()
    hfield_manager = HFieldManager(hfield_config)
    level_rng = np.random.default_rng(args.seed)

    env = FunctionalEnv(
        task=args.task,
        env_overrides=merge_overrides(
            accel_arena_overrides(hfield_config.pad_radius), {"sim": {"num_envs": 1}}
        ),
        episode_length=args.steps_in_episode,
        seed=args.seed,
    )

    template = A2C(
        hidden_dims=args.hidden_dims,
        actor_obs_dim=env.observation_size.actor,
        critic_obs_dim=env.observation_size.critic,
        action_dim=env.action_size,
        action_high=1,
        key=jax.random.key(args.seed),
        activation=args.activation,
    )
    model = restore_model(Path(args.checkpoint) / "model", template)

    jit_step = eqx.filter_jit(env.step)
    jit_act = eqx.filter_jit(model.deterministic_action)

    base = cast(Any, env._env.unwrapped)
    mj_model = base.mj_model
    mj_data = mujoco.MjData(mj_model)

    key = jax.random.key(args.seed)
    episode = 0
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running() and (args.episodes == 0 or episode < args.episodes):
            level = hfield_manager.generate(level_rng)
            hfield_data = level_to_hfield_data(hfield_manager, level)
            mj_model.hfield_data[:] = np.asarray(hfield_data)

            key, reset_key = jax.random.split(key)
            obs, state = env.reset(reset_key, hfield_data=hfield_data)

            done = False
            while not done and viewer.is_running():
                step_start = time.time()
                action = jit_act(obs)
                key, step_key = jax.random.split(key)
                obs, state, _reward, done_arr, _info = jit_step(
                    step_key, state, action, hfield_data=hfield_data
                )
                done = bool(done_arr)

                mj_data.qpos[:] = np.asarray(state.data.qpos)
                mj_data.qvel[:] = np.asarray(state.data.qvel)
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()
                time.sleep(max(0.0, base.dt - (time.time() - step_start)))

            episode += 1


if __name__ == "__main__":
    main()
