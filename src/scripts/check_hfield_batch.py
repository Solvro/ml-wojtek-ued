"""Check that per-env hfield_data reaches the physics under vmap.

Envs alternate between a flat level (0.5 everywhere) and a pit (0.0
everywhere, `amplitude` m below the flat plane). Spawn reads the terrain
lookup and contact reads the physics heightfield, so when both follow each
env's own level a pit env stands `amplitude` lower than a flat env with no
more dones. Prints the mean base z over the last quarter and the number of
dones over `--steps` of zero action. Run once per backend: pit envs standing
at the flat height, or pit envs that keep terminating, mean the lookup or the
physics ignores the per-env level.
"""

import argparse
import time
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from config.hfield_config import HFieldConfig
from utils.env import FunctionalEnv
from utils.env_overrides import merge_overrides, movement_overrides, terrain_overrides


def make_env(backend: str, num_envs: int, pad_radius: float) -> FunctionalEnv:
    overrides = merge_overrides(
        {"terrain": {"enable": True, "arena": "accel", "pad_radius": pad_radius}},
        movement_overrides(),
        terrain_overrides(),
        # only falls may end an episode, so dones count physical drops
        {
            "sim": {"num_envs": num_envs, "backend": backend},
            "no_progress": {"enable": False},
            "push": {"enable": False},
            "command": {"zero_prob": 1.0},
        },
    )
    return FunctionalEnv(task="joystick", env_overrides=overrides)


def settle(
    env: FunctionalEnv, keys: jax.Array, hfield: jax.Array, steps: int
) -> jax.Array:
    action = jnp.zeros(env.action_size)

    def step(state: Any, _: None) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        z = state.data.qpos[2]
        _obs, state, _reward, done, _info = env.step(keys[0], state, action, hfield)
        return state, (z, done)

    _, state = env.reset(keys[0], hfield)
    _, (z, done) = jax.lax.scan(step, state, None, steps)
    return jnp.stack([z[-steps // 4 :].mean(), done.sum()])


def per_env_z(env: FunctionalEnv, hfield: jax.Array, steps: int) -> np.ndarray:
    keys = jax.random.split(jax.random.key(0), hfield.shape[0])[:, None]
    run = jax.jit(jax.vmap(lambda k, h: settle(env, k, h, steps)))
    return np.asarray(run(keys, hfield))


def per_lane_z(
    env: FunctionalEnv, hfield: jax.Array, envs_per_lane: int, steps: int
) -> np.ndarray:
    num_lanes = hfield.shape[0]
    keys = jax.random.split(jax.random.key(0), num_lanes * envs_per_lane)
    keys = keys.reshape(num_lanes, envs_per_lane, 1)

    def lane(lane_keys: jax.Array, h: jax.Array) -> jax.Array:
        return jax.vmap(lambda k: settle(env, k, h, steps))(lane_keys)

    return np.asarray(jax.jit(jax.vmap(lane))(keys, hfield))


def bench(backend: str, num_envs: int, steps: int, pad_radius: float) -> None:
    env = make_env(backend, num_envs, pad_radius)
    n = HFieldConfig().grid_size ** 2
    hfield = jax.random.uniform(jax.random.key(1), (num_envs, n))
    keys = jax.random.split(jax.random.key(0), num_envs)[:, None]
    run = jax.jit(jax.vmap(lambda k, h: settle(env, k, h, steps)))
    jax.block_until_ready(run(keys, hfield))
    start = time.perf_counter()
    jax.block_until_ready(run(keys, hfield))
    rate = num_envs * steps / (time.perf_counter() - start)
    print(f"bench backend={backend} envs={num_envs}: {rate:,.0f} env-steps/s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="auto", choices=["auto", "jax", "warp"])
    p.add_argument("--amplitude", type=float, default=0.3)
    p.add_argument(
        "--pad_radius",
        type=float,
        default=HFieldConfig().pad_radius,
        help="must match the built accel arena",
    )
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--envs_per_lane", type=int, default=2)
    p.add_argument("--bench_envs", type=int, default=0, help="0 = skip timing")
    args = p.parse_args()

    if args.bench_envs:
        bench(args.backend, args.bench_envs, args.steps, args.pad_radius)
        return

    config = HFieldConfig(amplitude=args.amplitude, pad_radius=args.pad_radius)
    n = config.grid_size**2
    flat = jnp.full((n,), 0.5, dtype=jnp.float32)
    pit = jnp.zeros((n,), dtype=jnp.float32)

    env = make_env(args.backend, 4 * args.envs_per_lane, config.pad_radius)
    print(
        f"backend={cast(Any, env._env.unwrapped)._backend} amplitude={args.amplitude}"
    )

    for name, levels in (
        ("flat,pit,flat,pit", [flat, pit, flat, pit]),
        ("pit,flat,pit,flat", [pit, flat, pit, flat]),
    ):
        hfield = jnp.stack(levels)
        envs = per_env_z(env, hfield, args.steps)
        print(f"per-env vmap [{name}] (late z, dones) per env:\n{np.round(envs, 3)}")
        lanes = per_lane_z(env, hfield, args.envs_per_lane, args.steps)
        print(
            f"lanes vmap [{name}] (late z, dones) per lane x env:\n{np.round(lanes, 3)}"
        )
    print(
        f"expected: pit late z ~{args.amplitude:.2f} m below flat, "
        "dones similar for both"
    )


if __name__ == "__main__":
    main()
