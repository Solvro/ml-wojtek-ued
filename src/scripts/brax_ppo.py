"""Reference student: brax PPO on the benchmark env, flat ground.

Same task, observations and domain randomization as `ppo_dr_terrain.py`, but
trained with brax's PPO (mujoco_playground locomotion recipe: observation
normalization, persistent episodes with short unrolls) on the accel arena's
static flat ground. Answers whether the env is learnable at all, separately
from our PPO. Ends with a tracking eval: commanded vs body-frame forward
velocity, falls and travel over `--eval_steps`.
"""

import argparse
import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from utils.env_overrides import (
    accel_arena_overrides,
    merge_overrides,
    movement_overrides,
    terrain_overrides,
    terrain_task_overrides,
)
from utils.ppo_helpers import body_velocity
from wojtek_env.envs import randomize, registry


def make_env(pad_radius: float, num_envs: int) -> Any:
    overrides = merge_overrides(
        accel_arena_overrides(pad_radius),
        movement_overrides(),
        terrain_overrides(),
        terrain_task_overrides(),
        {"sim": {"num_envs": num_envs}},
    )
    return registry.make_env("joystick", overrides)


def tracking_eval(
    env: Any, inference_fn: Any, num_envs: int, steps: int, seed: int
) -> dict[str, float]:
    def rollout(key: jax.Array) -> tuple[jax.Array, ...]:
        state = env.reset(key)
        start_xy = state.data.qpos[:2]

        def step(carry: Any, _: None) -> tuple[Any, tuple[jax.Array, ...]]:
            state, fell, key = carry
            key, act_key = jax.random.split(key)
            action, _ = inference_fn(state.obs, act_key)
            cmd_vx = state.info["command"][0]
            state = env.step(state, action)
            fell = jnp.maximum(fell, state.done)
            return (state, fell, key), (cmd_vx, body_velocity(state.data)[0])

        (last, fell, _), (cmd_vx, forward_vel) = jax.lax.scan(
            step, (state, jnp.zeros(()), key), None, steps
        )
        travel = jnp.linalg.norm(last.data.qpos[:2] - start_xy)
        return fell, travel, cmd_vx.mean(), forward_vel.mean()

    keys = jax.random.split(jax.random.key(seed), num_envs)
    fell, travel, cmd_vx, forward_vel = (
        np.asarray(x) for x in jax.jit(jax.vmap(rollout))(keys)
    )
    standing = fell == 0
    return {
        "track/fall_rate": float(fell.mean()),
        "track/mean_travel": float(travel.mean()),
        "track/mean_travel_no_fall": float(travel[standing].mean())
        if standing.any()
        else float("nan"),
        "track/mean_cmd_vx": float(cmd_vx.mean()),
        "track/mean_forward_vel": float(forward_vel.mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="wandb run name")
    p.add_argument("--entity", default="")
    p.add_argument("--project", default="wojtyla")
    p.add_argument("--num_timesteps", type=int, default=200_000_000)
    p.add_argument("--num_envs", type=int, default=8192)
    p.add_argument("--pad_radius", type=float, default=0.55)
    p.add_argument("--eval_steps", type=int, default=400)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # mujoco_playground's Go1 joystick recipe, with our network widths
    ppo_params = {
        "num_timesteps": args.num_timesteps,
        "num_evals": 10,
        "reward_scaling": 1.0,
        "episode_length": 1000,
        "normalize_observations": True,
        "action_repeat": 1,
        "unroll_length": 20,
        "num_minibatches": 32,
        "num_updates_per_batch": 4,
        "discounting": 0.97,
        "learning_rate": 3e-4,
        "entropy_cost": 1e-2,
        "num_envs": args.num_envs,
        "batch_size": 256,
        "max_grad_norm": 1.0,
        "clipping_epsilon": 0.2,
        "seed": args.seed,
    }
    run = wandb.init(
        entity=args.entity or None,
        project=args.project,
        name=args.name,
        tags=["brax-reference", "flat"],
        config={**vars(args), "ppo": ppo_params},
    )

    env = make_env(args.pad_radius, args.num_envs)
    eval_env = make_env(args.pad_radius, 128)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        activation=jax.nn.elu,
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        scalars = {k: float(v) for k, v in metrics.items() if np.ndim(v) == 0}
        run.log({"env_steps": num_steps, **scalars})
        print(num_steps, scalars.get("eval/episode_reward"), flush=True)

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomize.make_domain_randomize(env.mj_model),
        network_factory=network_factory,
        progress_fn=progress,
        save_checkpoint_path=args.checkpoint_dir,
        **ppo_params,
    )

    inference_fn = make_inference_fn(params, deterministic=True)
    track_env = make_env(args.pad_radius, 256)
    run.log(tracking_eval(track_env, inference_fn, 256, args.eval_steps, args.seed))
    run.finish()


if __name__ == "__main__":
    main()
