"""`end_diverged` on a hand-built batched state (no env, no model)."""

from typing import Any, NamedTuple, cast

import jax.numpy as jnp
import numpy as np
from mujoco_playground._src import mjx_env

from env.divergence import end_diverged


class Data(NamedTuple):
    qpos: Any
    qvel: Any


def state(qvel: Any, info: dict[str, Any]) -> mjx_env.State:
    n = qvel.shape[0]
    return mjx_env.State(
        # end_diverged reads qpos and qvel only
        data=cast(Any, Data(qpos=jnp.zeros((n, 3)), qvel=qvel)),
        obs={"state": jnp.zeros((n, 2))},
        reward=jnp.ones(n),
        done=jnp.zeros(n),
        metrics={"reward/x": jnp.ones(n)},
        info=info,
    )


def test_a_diverged_env_gets_finite_info_and_ends() -> None:
    nan = jnp.nan
    info = {
        "swing_peak": jnp.array([[nan, 0.1], [nan, 0.2]]),
        "first_obs": {"state": jnp.full((2, 2), nan)},
        "steps": jnp.array([3, 4]),
    }
    qvel = jnp.array([[1e4, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out = end_diverged(state(qvel, info))
    np.testing.assert_array_equal(out.done, [1.0, 0.0])
    np.testing.assert_array_equal(out.info["diverged"], [1.0, 0.0])
    np.testing.assert_array_equal(out.reward, [0.0, 1.0])
    # only the diverged env is scrubbed; a nan in a healthy env stays visible
    peak = np.asarray(out.info["swing_peak"])
    np.testing.assert_array_equal(peak[0], np.array([0.0, 0.1], np.float32))
    assert np.isnan(peak[1, 0]) and peak[1, 1] == np.float32(0.2)
    assert np.isnan(np.asarray(out.info["first_obs"]["state"])).all()
    np.testing.assert_array_equal(out.info["steps"], [3, 4])


def test_a_finite_state_passes_through_unchanged() -> None:
    info = {"swing_peak": jnp.array([[0.3, 0.1]]), "steps": jnp.array([7])}
    out = end_diverged(state(jnp.zeros((1, 3)), info))
    np.testing.assert_array_equal(
        out.info["swing_peak"], np.array([[0.3, 0.1]], np.float32)
    )
    np.testing.assert_array_equal(out.done, [0.0])
    np.testing.assert_array_equal(out.reward, [1.0])
