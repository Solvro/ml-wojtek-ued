import functools
from collections.abc import Callable
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from brax.training import types

from env.functional import FunctionalEnv


class Unroll(Protocol):
    def __call__(
        self,
        inference: Any,
        state: Any,
        key: jax.Array,
        hfield: jax.Array,
        dr_model: Any,
        *,
        deterministic: bool,
        length: int,
    ) -> tuple[Any, types.Transition]: ...


def make_unroll(
    env: FunctionalEnv, make_policy: Callable[..., Any], dr_in_axes: Any
) -> Unroll:
    """Jitted policy rollout of a batch of envs on one shared arena;
    transitions come back env-major, with per-step logs in `extras`."""

    def step_one(
        state: Any, action: jax.Array, hfield: jax.Array, dr_model: Any
    ) -> tuple[Any, ...]:
        _, state, reward, done, info = env.step(
            jax.random.key(0), state, action, hfield, dr_model
        )
        return state, reward, done, info["truncation"], info["diverged"]

    # warp collides every env with the first env's hfield_data, so the arena
    # is shared, not batched
    v_step = jax.vmap(step_one, in_axes=(0, 0, None, dr_in_axes))

    @functools.partial(jax.jit, static_argnames=("deterministic", "length"))
    def unroll(
        inference: Any,
        state: Any,
        key: jax.Array,
        hfield: jax.Array,
        dr_model: Any,
        *,
        deterministic: bool,
        length: int,
    ) -> tuple[Any, types.Transition]:
        policy = make_policy(inference, deterministic=deterministic)

        def step(carry: Any, _: None) -> tuple[Any, types.Transition]:
            state, key = carry
            key, act_key = jax.random.split(key)
            action, extras = policy(state.obs, act_key)
            cmd_vx = state.info["command"][:, 0]
            cmd_speed = jnp.linalg.norm(state.info["command"][:, :2], axis=-1)
            next_state, reward, done, truncation, diverged = v_step(
                state, action, hfield, dr_model
            )
            forward_vel = jax.vmap(env.forward_velocity)(next_state.data)
            transition = types.Transition(
                observation=state.obs,
                action=action,
                reward=reward,
                discount=1.0 - done,
                next_observation=next_state.obs,
                # brax types extras as an array; a nested dict is what it reads
                extras={  # ty: ignore[invalid-argument-type]
                    "policy_extras": extras,
                    "state_extras": {"truncation": truncation},
                    "log": {
                        "fell": done * (1.0 - truncation),
                        "diverged": diverged,
                        "reward": reward,
                        "cmd_vx": cmd_vx,
                        "cmd_speed": cmd_speed,
                        "forward_vel": forward_vel,
                        "xy": next_state.data.qpos[:, :2],
                    },
                },
            )
            return (next_state, key), transition

        (state, _), data = jax.lax.scan(step, (state, key), None, length)
        return state, jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), data)

    return unroll
