import jax.numpy as jnp
import numpy as np
import pytest

from scripts.eval_brax import (
    TRACK_LIN_TOL,
    TRACK_YAW_TOL,
    stack_passes,
    tracked,
    tracking_errors,
    tracking_summary,
)


def episodes(fell: float, lin_err: float, yaw_err: float) -> dict[str, np.ndarray]:
    return {
        "fell": np.full((1, 1), fell),
        "travel": np.zeros((1, 1)),
        "lin_err": np.full((1, 1), lin_err),
        "yaw_err": np.full((1, 1), yaw_err),
    }


def test_tracking_errors_skip_the_start_up_transient() -> None:
    cmd = jnp.tile(jnp.array([0.3, 0.4, 1.0]), (1, 4, 1))
    vel = jnp.zeros((1, 4, 3)).at[:, 2:].set(jnp.array([0.3, 0.4, 0.5]))
    lin, yaw = tracking_errors(cmd, vel, skip=2)
    assert lin == pytest.approx([0.0])
    assert yaw == pytest.approx([0.5])
    lin_all, _ = tracking_errors(cmd, vel, skip=0)
    assert lin_all == pytest.approx([0.25])


def rollout_tracked(cmd: list[float], vel: list[float]) -> bool:
    lin, yaw = tracking_errors(
        jnp.tile(jnp.array(cmd), (1, 4, 1)), jnp.tile(jnp.array(vel), (1, 4, 1)), 0
    )
    episode = {
        "fell": np.zeros((1, 1)),
        "lin_err": np.asarray(lin).reshape(1, 1),
        "yaw_err": np.asarray(yaw).reshape(1, 1),
    }
    return bool(tracked(episode, "wojtek").all())


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
    assert rollout_tracked(cmd, vel) is expected


def test_standing_still_on_a_stand_command_counts_as_tracked() -> None:
    assert tracked(episodes(fell=0, lin_err=0.0, yaw_err=0.0), "wojtek").all()


def test_a_fall_is_never_tracked() -> None:
    assert not tracked(episodes(fell=1, lin_err=0.0, yaw_err=0.0), "wojtek").any()


@pytest.mark.parametrize(
    ("lin_err", "yaw_err"),
    [(TRACK_LIN_TOL, 0.0), (0.0, TRACK_YAW_TOL)],
)
def test_error_at_the_tolerance_fails(lin_err: float, yaw_err: float) -> None:
    assert not tracked(episodes(0, lin_err, yaw_err), "wojtek").any()


def test_linear_tolerance_scales_with_the_robot() -> None:
    walk = episodes(fell=0, lin_err=1.5 * TRACK_LIN_TOL, yaw_err=0.0)
    assert not tracked(walk, "wojtek").any()
    assert tracked(walk, "go1").all()


def test_tracking_summary_averages_errors_over_standing_episodes() -> None:
    passes = [episodes(0, 0.1, 0.2), episodes(1, 9.0, 9.0)]
    out = tracking_summary("set", passes, "wojtek")
    assert out["set/cmd_success_rate"] == pytest.approx(0.5)
    assert out["set/track_lin_err"] == pytest.approx(0.1)
    assert out["set/track_yaw_err"] == pytest.approx(0.2)


def test_tracking_summary_without_a_standing_episode_omits_errors() -> None:
    out = tracking_summary("set", [episodes(1, 0.0, 0.0)] * 2, "wojtek")
    assert out == {"set/cmd_success_rate": 0.0}


def test_stack_passes_names_every_key_per_set() -> None:
    raw = stack_passes("random_d1.0", [episodes(0, 0, 0), episodes(1, 0, 0)])
    assert set(raw) == {
        "random_d1.0_fell",
        "random_d1.0_travel",
        "random_d1.0_lin_err",
        "random_d1.0_yaw_err",
    }
    assert raw["random_d1.0_fell"].shape == (2, 1, 1)
