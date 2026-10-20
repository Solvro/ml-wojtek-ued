from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import numpy as np
import onnx
import pytest
from brax.io import model as brax_model
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from config.schema import CONF_DIR, TrainConfig
from export.run import (
    OUTPUT_FILES,
    export_policy,
    make_export_env,
    training_config,
)
from export.verify import TOLERANCE, session
from training.student import Student
from training.summary import TrainingSummary, write_summary
from wojtek_env import paths

Compose = Callable[..., Any]

# a 2 x 2 tile arena keeps the Playground env small; Wojtek needs the built
# benchmark arena
SMALL_ARENA = ("arena.tiles=2", "arena.reach=2.0", "arena.patch_radius=1.5")


def tiny_checkpoint(cfg: TrainConfig, path: Path) -> None:
    env, _, _ = make_export_env(cfg)
    student = Student(env, jax.random.key(0), 0.01, cfg.ppo, cfg.network)
    brax_model.save_params(str(path), student.inference_params())
    summary = TrainingSummary(
        env_steps=640,
        wall_time_s=12.0,
        window_iterations=2,
        reward_per_step=0.1,
        train_success=0.0,
        falls_per_step=0.0,
        forward_vel=0.05,
        cmd_vx=0.3,
    )
    write_summary(path, summary)


def composed(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        return compose("export", overrides=["checkpoint=c.pkl", *overrides])


def test_policy_name_composes_conf_policy() -> None:
    cfg = training_config(composed("policy=template"))
    assert cfg.name == "my_policy"
    assert cfg.checkpoint_dir == "artifacts/checkpoints"
    assert not hasattr(cfg, "checkpoint")


def test_overrides_alone_define_the_config() -> None:
    cfg = training_config(composed("mode=accel", "name=bench", "robot=go1"))
    assert (cfg.name, cfg.robot.name) == ("bench", "go1")
    assert cfg.mode.curriculum == "accel"


def test_config_file_replaces_the_composed_values(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {"name": "run", "seed": 3, "mode": {"curriculum": "plr"}}
            )
        )
    )
    cfg = training_config(composed("mode=dr", f"config={path}"))
    assert (cfg.name, cfg.seed, cfg.mode.curriculum) == ("run", 3, "plr")
    # fields the file lacks (an older run) keep the conf/train.yaml values
    assert cfg.ppo.unroll_length == 20
    assert cfg.network.policy_hidden == [512, 256, 128]


def test_missing_config_file_is_named(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="no such file"):
        training_config(composed("mode=dr", "name=t", f"config={missing}"))


def test_missing_checkpoint_is_named(tmp_path: Path) -> None:
    cfg = training_config(composed("policy=template"))
    with pytest.raises(FileNotFoundError, match="checkpoint .* not found"):
        export_policy(cfg, tmp_path / "missing.pkl", tmp_path)


def test_go1_export_writes_three_files_and_matches_jax(
    compose_cfg: Compose, tmp_path: Path
) -> None:
    cfg = compose_cfg(
        "train",
        "mode=dr",
        "name=tiny",
        "robot=go1",
        "backend=jax",
        *SMALL_ARENA,
    )
    checkpoint = tmp_path / "tiny.pkl"
    tiny_checkpoint(cfg, checkpoint)
    out_dir, ver = export_policy(cfg, checkpoint, tmp_path / "export")
    assert out_dir == tmp_path / "export" / "tiny"
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(OUTPUT_FILES)
    assert max(ver.random_error, ver.env_error) <= TOLERANCE
    model = onnx.load(out_dir / "policy.onnx")
    obs = np.zeros((3, 48), np.float32)
    action, targets = (
        np.asarray(x) for x in session(model).run(None, {"obs": obs})
    )
    assert action.shape == targets.shape == (3, 12)
    assert np.abs(action).max() <= 1.0
    readme = (out_dir / "README.md").read_text()
    assert "`linvel`" in readme and "reward per step" in readme
    saved = OmegaConf.load(out_dir / "config.yaml")
    assert saved.name == "tiny" and saved.robot.name == "go1"


def test_export_refuses_a_checkpoint_of_another_robot(
    compose_cfg: Compose, tmp_path: Path
) -> None:
    go1 = compose_cfg(
        "train",
        "mode=dr",
        "name=tiny",
        "robot=go1",
        "backend=jax",
        *SMALL_ARENA,
    )
    checkpoint = tmp_path / "tiny.pkl"
    tiny_checkpoint(go1, checkpoint)
    spot = compose_cfg(
        "train",
        "mode=dr",
        "name=tiny",
        "robot=spot",
        "backend=jax",
        *SMALL_ARENA,
    )
    with pytest.raises(ValueError, match="takes 48 actor observations"):
        export_policy(spot, checkpoint, tmp_path / "export")
    assert not (tmp_path / "export").exists()


@pytest.mark.slow
@pytest.mark.skipif(
    not paths.terrain_paths("accel")["spec"].exists(),
    reason="accel arena not built; run `just build-accel-arena`",
)
def test_wojtek_export_end_to_end(compose_cfg: Compose, tmp_path: Path) -> None:
    cfg = compose_cfg("train", "mode=dr", "name=tiny", "backend=jax")
    checkpoint = tmp_path / "tiny.pkl"
    tiny_checkpoint(cfg, checkpoint)
    out_dir, ver = export_policy(cfg, checkpoint, tmp_path / "export")
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(OUTPUT_FILES)
    assert max(ver.random_error, ver.env_error) <= TOLERANCE
    readme = (out_dir / "README.md").read_text()
    assert "`phase` | 46:54 | 8 |" in readme
    assert "FILTER = 0.2" in readme
