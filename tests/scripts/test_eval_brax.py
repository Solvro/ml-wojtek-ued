import numpy as np
import pytest

from scripts.brax_ued import ROBOT_SCALE
from scripts.eval_brax import (
    TRACK_FLOOR_LIN,
    TRACK_FLOOR_YAW,
    TRACK_REL_TOL,
    stack_passes,
    tracked,
    tracking_errors,
    tracking_summary,
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


def episodes(fell: float, lin: float, yaw: float) -> dict[str, np.ndarray]:
    return {
        "fell": np.full((1, 1), fell),
        "travel": np.zeros((1, 1)),
        "lin_err": np.full((1, 1), lin),
        "yaw_err": np.full((1, 1), yaw),
        "lin_rel": np.full((1, 1), lin),
        "yaw_rel": np.full((1, 1), yaw),
    }


def test_tracking_summary_averages_errors_over_standing_episodes() -> None:
    out = tracking_summary("set", [episodes(0, 0.1, 0.2), episodes(1, 9.0, 9.0)])
    assert out["set/cmd_success_rate"] == pytest.approx(0.5)
    assert out["set/track_lin_err"] == pytest.approx(0.1)
    assert out["set/track_yaw_err"] == pytest.approx(0.2)


def test_tracking_summary_without_a_standing_episode_omits_errors() -> None:
    out = tracking_summary("set", [episodes(1, 0.0, 0.0)] * 2)
    assert out == {"set/cmd_success_rate": 0.0}


def test_stack_passes_names_every_key_per_set() -> None:
    raw = stack_passes("random_d1.0", [episodes(0, 0, 0), episodes(1, 0, 0)])
    assert set(raw) == {
        f"random_d1.0_{k}"
        for k in ("fell", "travel", "lin_err", "yaw_err", "lin_rel", "yaw_rel")
    }
    assert raw["random_d1.0_fell"].shape == (2, 1, 1)
