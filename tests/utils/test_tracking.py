import numpy as np
import pytest

from scripts.brax_ued import ROBOT_SCALE
from utils.tracking import (
    TRACK_FLOOR_LIN,
    TRACK_FLOOR_YAW,
    TRACK_REL_TOL,
    settled_steps,
    tracked,
    tracking_errors,
)


def constant(
    cmd: list[float], vel: list[float], steps: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    return np.tile(np.array(cmd), (1, steps, 1)), np.tile(np.array(vel), (1, steps, 1))


def judged(
    cmd: np.ndarray,
    vel: np.ndarray,
    skip: int = 0,
    robot: str = "wojtek",
    fell: float = 0.0,
) -> dict[str, np.ndarray]:
    floor = TRACK_FLOOR_LIN * ROBOT_SCALE[robot]
    errors = tracking_errors(cmd, vel, skip, floor)
    return {"fell": np.full(len(cmd), fell), "travel": np.zeros(len(cmd)), **errors}


def is_tracked(cmd: list[float], vel: list[float], robot: str = "wojtek") -> bool:
    return bool(tracked(judged(*constant(cmd, vel), robot=robot)).all())


def test_tracking_errors_skip_the_start_up_transient() -> None:
    cmd, vel = constant([0.3, 0.4, 1.0], [0.3, 0.4, 0.5])
    vel[:, :2] = 0.0
    settled = tracking_errors(cmd, vel, 2, TRACK_FLOOR_LIN)
    assert settled["lin_err"] == pytest.approx([0.0])
    assert settled["yaw_err"] == pytest.approx([0.5])
    assert settled["yaw_rel"] == pytest.approx([0.5])
    whole = tracking_errors(cmd, vel, 0, TRACK_FLOOR_LIN)
    assert whole["lin_err"] == pytest.approx([0.25])


def resampled(lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Command 0.5 m/s forward for 6 steps, then 1 rad/s spin for 6; the
    robot follows each command `lag` steps late, standing in between."""
    cmd = np.zeros((1, 12, 3))
    cmd[:, :6, 0] = 0.5
    cmd[:, 6:, 2] = 1.0
    vel = np.zeros_like(cmd)
    vel[:, lag:6, 0] = 0.5
    vel[:, 6 + lag :, 2] = 1.0
    return cmd, vel


def test_settled_steps_restart_the_skip_at_every_command_change() -> None:
    settled = settled_steps(resampled(0)[0], skip=2)
    expected = [False, False, True, True, True, True] * 2
    assert settled[0].tolist() == expected


def test_settled_steps_ignore_a_resample_that_draws_the_same_command() -> None:
    cmd, _ = constant([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], steps=6)
    assert settled_steps(cmd, skip=2)[0].tolist() == [False] * 2 + [True] * 4


def test_post_resample_transient_is_not_scored() -> None:
    cmd, vel = resampled(lag=2)
    settled = judged(cmd, vel, skip=2)
    assert settled["lin_rel"] == pytest.approx([0.0])
    assert settled["yaw_rel"] == pytest.approx([0.0])
    whole = judged(cmd, vel, skip=0)
    assert whole["lin_rel"] == pytest.approx([2 / 12])
    assert whole["yaw_rel"] == pytest.approx([2 / 12])


def test_an_episode_with_no_settled_step_is_never_tracked() -> None:
    cmd, vel = constant([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], steps=3)
    run = judged(cmd, vel, skip=3)
    assert np.isnan(run["lin_rel"]).all()
    assert not tracked(run).any()


@pytest.mark.parametrize(
    ("cmd", "vel", "expected"),
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], True),
        ([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], False),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], True),
        ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], False),
        ([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], False),
        ([-0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], True),
        ([-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], False),
    ],
    ids=[
        "zero-stands",
        "zero-walks-off",
        "spin-tracked",
        "spin-stands",
        "spin-wrong-way",
        "backward-tracked",
        "backward-walks-forward",
    ],
)
def test_command_kinds_are_judged_by_signed_velocity_error(
    cmd: list[float], vel: list[float], expected: bool
) -> None:
    assert is_tracked(cmd, vel) is expected


def test_tolerance_is_relative_to_the_commanded_speed() -> None:
    assert is_tracked([1.2, 0.0, 0.0], [0.8, 0.0, 0.0])
    assert not is_tracked([0.2, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert is_tracked([0.0, 0.0, 1.5], [0.0, 0.0, 1.0])


def test_floors_let_standing_pass_only_near_zero_commands() -> None:
    lin_ok = 0.9 * TRACK_REL_TOL * TRACK_FLOOR_LIN
    yaw_ok = 0.9 * TRACK_REL_TOL * TRACK_FLOOR_YAW
    assert is_tracked([lin_ok, 0.0, yaw_ok], [0.0, 0.0, 0.0])
    assert not is_tracked([1.1 * TRACK_FLOOR_LIN, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert not is_tracked([0.0, 0.0, 1.1 * TRACK_FLOOR_YAW], [0.0, 0.0, 0.0])


@pytest.mark.parametrize("axis", [0, 2])
def test_relative_error_at_the_tolerance_fails(axis: int) -> None:
    cmd = [2.0, 0.0, 2.0]
    vel = list(cmd)
    vel[axis] = cmd[axis] * (1 - TRACK_REL_TOL)
    assert not is_tracked(cmd, vel)


def test_a_fall_is_never_tracked() -> None:
    run = judged(*constant([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]), fell=1.0)
    assert not tracked(run).any()


def test_linear_floor_scales_with_the_robot() -> None:
    drift = [0.0, 0.0, 0.0], [0.6 * TRACK_FLOOR_LIN, 0.0, 0.0]
    assert not is_tracked(*drift, robot="wojtek")
    assert is_tracked(*drift, robot="go1")
