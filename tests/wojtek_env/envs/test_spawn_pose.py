"""Local spawn-pose patch: reset poses every leg joint on the settled
closed-loop pose, so each four-bar starts closed."""

from typing import Any

import mujoco
import numpy as np
import pytest

from wojtek_env.envs import registry

HEIGHTS = (0.09, 0.11, 0.13, 0.15, 0.17)


@pytest.fixture(scope="module", params=[False, True], ids=["legacy", "real"])
def env(request: pytest.FixtureRequest) -> Any:
    return registry.make_env(
        "joystick", {"sim": {"backend": "jax"}, "real_pose_ref": request.param}
    )


def _closure_gap(env: Any, qpos: np.ndarray) -> float:
    m = env.mj_model
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    equality = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    return float(np.abs(d.efc_pos[equality]).max())


@pytest.mark.parametrize("height", HEIGHTS)
def test_spawn_pose_closes_every_leg_loop(env: Any, height: float) -> None:
    qpos = np.asarray(env._home_qpos).copy()
    qpos[7:] = np.asarray(env._spawn_joint_qpos(height))
    qpos[2] = height
    assert _closure_gap(env, qpos) < 2e-3


@pytest.mark.parametrize("height", HEIGHTS)
def test_spawn_pose_sits_near_the_pose_reference(env: Any, height: float) -> None:
    spawn = np.asarray(env._spawn_joint_qpos(height))[np.asarray(env._qadr) - 7]
    np.testing.assert_allclose(spawn, np.asarray(env._pose_ref(height)), atol=0.02)
