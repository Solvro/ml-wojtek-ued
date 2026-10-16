import dataclasses
import json
from pathlib import Path

import pytest

from config.hfield_config import HFieldConfig
from hfield.arena import arena_spec, build_accel_arena, check_arena
from hfield.tiles import TileLayout
from scripts.brax_ued import hfield_config, parse_args
from wojtek_env import paths

RUN = HFieldConfig(amplitude=0.3, coarse_n=13, pad_radius=0.55, pad_taper=0.2)


def write_spec(path: Path, config: HFieldConfig) -> Path:
    spec = arena_spec(config, build_accel_arena(config))
    path.write_text(json.dumps(spec))
    return path


def test_check_arena_accepts_matching_build(tmp_path: Path) -> None:
    check_arena(write_spec(tmp_path / "spec.json", RUN), RUN, "accel")


def test_check_arena_rejects_default_build_with_rebuild_command(
    tmp_path: Path,
) -> None:
    spec = write_spec(tmp_path / "spec.json", HFieldConfig())
    with pytest.raises(ValueError, match="hfield_params.pad_radius=0.55"):
        check_arena(spec, RUN, "accel")


def test_check_arena_rejects_amplitude_mismatch(tmp_path: Path) -> None:
    spec = write_spec(tmp_path / "spec.json", dataclasses.replace(RUN, amplitude=0.12))
    with pytest.raises(ValueError, match=r"'amplitude': 0\.12"):
        check_arena(spec, RUN, "accel")


def test_check_arena_missing_build(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="just build-accel-arena"):
        check_arena(tmp_path / "spec.json", RUN, "accel")


def profile_config(profile: str) -> HFieldConfig:
    flags = ["--mode", "accel", "--name", "t", "--profile", profile]
    return hfield_config(parse_args(flags))


def test_deploy_arena_is_12m_with_half_metre_features(tmp_path: Path) -> None:
    config = profile_config("deploy")
    assert config.grid_size == 241
    assert 2 * config.patch_radius / (config.coarse_n - 1) == pytest.approx(0.5)
    check_arena(write_spec(tmp_path / "spec.json", config), config, "accel_deploy")


def test_benchmark_arena_rejected_for_deploy_with_deploy_rebuild(
    tmp_path: Path,
) -> None:
    spec = write_spec(tmp_path / "spec.json", profile_config("benchmark"))
    with pytest.raises(ValueError, match="arena=accel_deploy.*patch_radius=6.0"):
        check_arena(spec, profile_config("deploy"), "accel_deploy")


def test_deploy_arena_files_do_not_overwrite_benchmark() -> None:
    deploy = paths.terrain_paths("accel_deploy")
    benchmark = paths.terrain_paths("accel")
    assert set(deploy.values()).isdisjoint(benchmark.values())


def test_check_arena_rejects_single_level_arena_with_rebuild(tmp_path: Path) -> None:
    spec = arena_spec(RUN, build_accel_arena(RUN))
    del spec["tile_layout"]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="hfield_params.tiles=16"):
        check_arena(path, RUN, "accel")


def test_check_arena_rejects_other_tile_count(tmp_path: Path) -> None:
    spec = write_spec(tmp_path / "spec.json", dataclasses.replace(RUN, tiles=8))
    with pytest.raises(ValueError, match="'tiles': 8"):
        check_arena(spec, RUN, "accel")


def test_static_arena_spans_every_tile_and_its_gap() -> None:
    arena = build_accel_arena(RUN)
    layout = TileLayout(RUN)
    assert arena.hfield_data.shape == (layout.nodes, layout.nodes)
    assert arena.spec.x_max == pytest.approx(layout.half)
    # pinned corners keep MuJoCo's [0, 1] rescale the identity
    assert arena.hfield_data.min() == 0.0 and arena.hfield_data.max() == 1.0
