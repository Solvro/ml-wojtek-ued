import math

import jax.numpy as jnp
import numpy as np
import pytest

from training.iteration import commanded_motion

DT = 0.02


def motion(
    command: list[float], steps: int, heading: float = 0.0
) -> tuple[float, np.ndarray]:
    commands = jnp.tile(jnp.array(command), (1, steps, 1))
    turned, displacement = commanded_motion(
        jnp.array([heading]), jnp.zeros((1, 2)), commands, DT
    )
    return float(turned[0]), np.asarray(displacement[0])


def test_straight_command_covers_speed_times_time() -> None:
    _, displacement = motion([0.6, 0.8, 0.0], 100)
    np.testing.assert_allclose(displacement, [1.2, 1.6], rtol=1e-5)


def test_heading_rotates_the_body_frame_command() -> None:
    _, displacement = motion([1.0, 0.0, 0.0], 50, heading=math.pi / 2)
    np.testing.assert_allclose(displacement, [0.0, 1.0], atol=1e-5)


def test_a_full_commanded_circle_ends_where_it_started() -> None:
    steps = 400
    wz = 2 * math.pi / (steps * DT)
    turned, displacement = motion([1.0, 0.0, wz], steps)
    assert turned == pytest.approx(2 * math.pi, rel=1e-5)
    assert np.linalg.norm(displacement) < 1e-3


def test_unrolls_chain_through_heading_and_displacement() -> None:
    command = jnp.tile(jnp.array([0.5, 0.0, 0.3]), (1, 60, 1))
    whole = commanded_motion(jnp.zeros(1), jnp.zeros((1, 2)), command, DT)
    heading, displacement = jnp.zeros(1), jnp.zeros((1, 2))
    for part in (command[:, :20], command[:, 20:]):
        heading, displacement = commanded_motion(
            heading, displacement, part, DT
        )
    np.testing.assert_allclose(heading, whole[0], rtol=1e-5)
    np.testing.assert_allclose(displacement, whole[1], rtol=1e-5)
