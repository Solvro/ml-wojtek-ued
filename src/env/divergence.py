from typing import Any

import jax
import jax.numpy as jnp

# m/s or rad/s: random actions on full-difficulty levels peak at ~75 (joints),
# a blowing-up env passes 1e3-1e7 before it turns nan
QVEL_DIVERGED = 200.0


def end_diverged(state: Any) -> Any:
    """End the episode of an env whose step blew up or turned nan: such obs
    poison brax's running obs normalizer for good."""
    done = state.done

    def per_env(ok: jax.Array) -> jax.Array:
        return ok.reshape(*done.shape, -1).all(axis=-1)

    leaves = [state.data.qpos, state.data.qvel, state.reward]
    leaves += jax.tree.leaves(state.obs)
    finite = jnp.all(jnp.stack([per_env(jnp.isfinite(x)) for x in leaves]), axis=0)
    bounded = per_env(jnp.abs(state.data.qvel) <= QVEL_DIVERGED)
    bad = ~(finite & bounded)
    state.info["diverged"] = bad.astype(done.dtype)
    metrics = {
        k: jnp.where(bad, 0.0, v) if v.shape == done.shape else v
        for k, v in state.metrics.items()
    }
    return state.replace(
        done=jnp.where(bad, jnp.ones_like(done), done),
        reward=jnp.where(bad, 0.0, state.reward),
        metrics=metrics,
    )
