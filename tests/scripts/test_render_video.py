import argparse
import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from config.hfield_config import HFieldConfig
from scripts.render_video import check_arena, frame_steps, hold_command, parse_level

RUN = HFieldConfig(amplitude=0.3, coarse_n=13, pad_radius=0.55, pad_taper=0.2)


def write_spec(path: Path, config: HFieldConfig) -> Path:
    hfield = {"elevation_z": 2 * config.amplitude, "nrow": config.grid_size}
    path.write_text(json.dumps({"pad_radius": config.pad_radius, "hfield": hfield}))
    return path


def test_parse_level_reads_difficulty_and_seed() -> None:
    assert parse_level("0.75:3") == (0.75, 3)


@pytest.mark.parametrize("text", ["0.5", "0.5:x", "a:1", "1.5:2", "-0.1:2", ""])
def test_parse_level_rejects_malformed(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="DIFFICULTY:SEED"):
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


def test_hold_command_without_resample_counter() -> None:
    held = hold_command({"command": jnp.zeros(3)}, jnp.ones(3))
    np.testing.assert_allclose(held["command"], np.ones(3))
    assert set(held) == {"command"}


def test_check_arena_accepts_matching_build(tmp_path: Path) -> None:
    check_arena(write_spec(tmp_path / "spec.json", RUN), RUN)


def test_check_arena_rejects_default_build_with_rebuild_command(
    tmp_path: Path,
) -> None:
    spec = write_spec(tmp_path / "spec.json", HFieldConfig())
    with pytest.raises(ValueError, match="hfield_params.pad_radius=0.55"):
        check_arena(spec, RUN)


def test_check_arena_rejects_amplitude_mismatch(tmp_path: Path) -> None:
    spec = write_spec(tmp_path / "spec.json", dataclasses.replace(RUN, amplitude=0.12))
    with pytest.raises(ValueError, match=r"'amplitude': 0\.12"):
        check_arena(spec, RUN)


def test_check_arena_missing_build(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="just build-accel-arena"):
        check_arena(tmp_path / "spec.json", RUN)
