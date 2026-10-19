from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from rendering.video import frame_steps, hold_command, parse_level, render


def test_parse_level_reads_difficulty_and_seed() -> None:
    assert parse_level("0.75:3") == (0.75, 3)


@pytest.mark.parametrize("text", ["0.5", "0.5:x", "a:1", "1.5:2", "-0.1:2", ""])
def test_parse_level_rejects_malformed(text: str) -> None:
    with pytest.raises(ValueError, match="DIFFICULTY:SEED"):
        parse_level(text)


def test_frame_steps_resamples_control_rate_to_fps() -> None:
    steps = frame_steps(n_steps=50, ctrl_dt=0.02, fps=25)
    assert len(steps) == 25
    np.testing.assert_array_equal(steps, np.arange(0, 50, 2))


def test_frame_steps_stays_within_rollout() -> None:
    steps = frame_steps(n_steps=1000, ctrl_dt=0.02, fps=30)
    assert len(steps) == 600
    assert steps[0] == 0 and steps.max() <= 999
    assert np.all(np.diff(steps) >= 1)


def test_hold_command_keeps_height_and_blocks_resample() -> None:
    info = {"command": jnp.array([0.5, -0.1, 0.3, 0.12]), "steps_since_cmd": 240}
    held = hold_command(info, jnp.array([0.3, 0.0, 0.2]))
    np.testing.assert_allclose(held["command"], [0.3, 0.0, 0.2, 0.12])
    assert held["steps_since_cmd"] == 0
    assert info["command"][0] == 0.5


def test_hold_command_blocks_go1_countdown() -> None:
    info = {"command": jnp.zeros(3), "steps_until_next_cmd": jnp.int32(1)}
    held = hold_command(info, jnp.ones(3))
    np.testing.assert_allclose(held["command"], np.ones(3))
    assert held["steps_until_next_cmd"] > 1000


def test_hold_command_blocks_barkour_and_spot_step_counter() -> None:
    info = {"command": jnp.zeros(3), "step": jnp.int32(499)}
    held = hold_command(info, jnp.ones(3))
    np.testing.assert_allclose(held["command"], np.ones(3))
    assert held["step"] == 0


def test_hold_command_without_resample_counter() -> None:
    held = hold_command({"command": jnp.zeros(3)}, jnp.ones(3))
    np.testing.assert_allclose(held["command"], np.ones(3))
    assert set(held) == {"command"}


def test_render_refuses_a_command_without_three_entries(
    compose_cfg: Callable[..., Any],
) -> None:
    cfg = compose_cfg(
        "render", "checkpoint=none.pkl", "levels=[0.5:1]", "command=[0.3,0.0]"
    )
    with pytest.raises(ValueError, match="VX, VY, WZ"):
        render(cfg)


def test_render_refuses_a_malformed_level(compose_cfg: Callable[..., Any]) -> None:
    cfg = compose_cfg("render", "checkpoint=none.pkl", "levels=[0.5]")
    with pytest.raises(ValueError, match="DIFFICULTY:SEED"):
        render(cfg)
