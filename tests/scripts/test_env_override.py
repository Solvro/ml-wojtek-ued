import json
from pathlib import Path
from typing import Any

import pytest

from deploy.contract import CONSUMED_KEYS
from scripts.brax_ued import parse_args, run_spec
from wojtek_env.envs import registry

ROOT = Path(__file__).parents[2]
SNAPSHOT = json.loads(
    (ROOT / "tests/deploy/upstream_terrain_blind_v4_4d.json").read_text()
)
ARMS = sorted((ROOT / "experiments/deploy-ab/arms").glob("*.json"))
DEPLOY = ["--mode", "accel", "--name", "t", "--profile", "deploy"]


def write(tmp_path: Path, override: Any) -> str:
    path = tmp_path / "override.json"
    path.write_text(json.dumps(override))
    return str(path)


def resolved(flags: list[str]) -> dict[str, Any]:
    config = registry.TASKS["joystick"][1]()
    registry._apply_overrides(config, run_spec(parse_args(flags))["env_overrides"])
    return json.loads(json.dumps(config.to_dict()))


@pytest.mark.parametrize("key", sorted(CONSUMED_KEYS))
def test_consumed_keys_are_refused(
    tmp_path: Path, key: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, {key: 1.0})
    with pytest.raises(SystemExit):
        parse_args([*DEPLOY, "--env_override", path])
    assert "not TRAINING_ONLY" in capsys.readouterr().err


def test_unclassified_keys_are_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args([*DEPLOY, "--env_override", write(tmp_path, {"made_up": 1})])


def test_malformed_json_names_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{reward")
    with pytest.raises(SystemExit):
        parse_args([*DEPLOY, "--env_override", str(path)])
    assert "broken.json" in capsys.readouterr().err


def test_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args([*DEPLOY, "--env_override", str(tmp_path / "none.json")])


def test_non_object_json_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args([*DEPLOY, "--env_override", write(tmp_path, ["reward"])])


def test_go1_has_no_env_config_to_override(tmp_path: Path) -> None:
    path = write(tmp_path, {"reward": {}})
    with pytest.raises(SystemExit):
        parse_args(
            ["--mode", "accel", "--name", "t", "--robot", "go1", "--env_override", path]
        )


def test_override_merges_onto_the_profile(tmp_path: Path) -> None:
    path = write(tmp_path, {"reward": {"scales": {"torque_rate": -1e-4}}})
    base = resolved(DEPLOY)
    ours = resolved([*DEPLOY, "--env_override", path])
    assert ours["reward"]["scales"]["torque_rate"] == -1e-4
    del ours["reward"]["scales"]["torque_rate"]
    del base["reward"]["scales"]["torque_rate"]
    assert ours == base


def test_every_arm_file_exists() -> None:
    assert ARMS, "experiments/deploy-ab/arms/*.json missing"


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: p.stem)
def test_arm_keeps_inference_equal_to_upstream(arm: Path) -> None:
    flags = [*DEPLOY, "--env_override", str(arm)]
    ours = resolved(flags)
    for key in CONSUMED_KEYS:
        assert ours[key] == SNAPSHOT["env_config"][key], key
    assert run_spec(parse_args(flags))["network"] == SNAPSHOT["network"]


@pytest.mark.parametrize(
    "flag", [["--clipping_epsilon", "0.3"], ["--level_steps", "1000"]]
)
def test_ppo_and_episode_arms_keep_inference_equal(flag: list[str]) -> None:
    ours = resolved([*DEPLOY, *flag])
    for key in CONSUMED_KEYS:
        assert ours[key] == SNAPSHOT["env_config"][key], key


def test_clipping_epsilon_defaults_to_the_benchmark_value() -> None:
    assert parse_args(DEPLOY).clipping_epsilon == 0.2
