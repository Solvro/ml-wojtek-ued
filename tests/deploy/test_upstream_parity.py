import json
from pathlib import Path
from typing import Any

from deploy.contract import CONSUMED_KEYS, TRAINING_ONLY_KEYS
from scripts.brax_ued import parse_args, run_spec
from wojtek_env.envs import registry

SNAPSHOT = json.loads(
    (Path(__file__).parent / "upstream_terrain_blind_v4_4d.json").read_text()
)
# training-only keys --profile deploy sets its own way (docs/deploy.md); sim
# only in num_envs, which brax_ued sets per run
DEVIATIONS = {"terrain", "sim"}


def deploy_spec() -> dict[str, Any]:
    flags = ["--mode", "accel", "--name", "t", "--profile", "deploy"]
    return run_spec(parse_args(flags))


def resolved_env_config(overrides: dict[str, Any]) -> dict[str, Any]:
    config = registry.TASKS["joystick"][1]()
    registry._apply_overrides(config, overrides)
    return json.loads(json.dumps(config.to_dict()))


def test_consumed_keys_match_upstream() -> None:
    ours = resolved_env_config(deploy_spec()["env_overrides"])
    upstream = SNAPSHOT["env_config"]
    for key in CONSUMED_KEYS:
        assert ours[key] == upstream[key], key


def test_training_only_keys_match_upstream_except_documented() -> None:
    ours = resolved_env_config(deploy_spec()["env_overrides"])
    upstream = SNAPSHOT["env_config"]
    assert set(ours) == set(upstream) == CONSUMED_KEYS | TRAINING_ONLY_KEYS
    for key in TRAINING_ONLY_KEYS - DEVIATIONS:
        assert ours[key] == upstream[key], key


def test_sim_matches_upstream_but_the_batch_size() -> None:
    # the shared tiled arena lets every env run on upstream's warp backend
    ours = resolved_env_config(deploy_spec()["env_overrides"])["sim"]
    del ours["num_envs"]
    assert ours == SNAPSHOT["env_config"]["sim"]


def test_terrain_is_the_deploy_accel_arena() -> None:
    terrain = resolved_env_config(deploy_spec()["env_overrides"])["terrain"]
    assert terrain["enable"] is True
    assert terrain["arena"] == "accel_deploy"


def test_gyro_bias_and_vibration_stay_off() -> None:
    noise = resolved_env_config(deploy_spec()["env_overrides"])["obs_noise"]
    assert noise["gyro_bias"] == 0.0
    assert noise["gyro_vib"] == 0.0


def test_network_matches_upstream() -> None:
    assert deploy_spec()["network"] == SNAPSHOT["network"]


def test_dr_matches_upstream() -> None:
    assert json.loads(json.dumps(deploy_spec()["dr_config"])) == SNAPSHOT["dr"]


def test_cmd_vel_box_contains_zero() -> None:
    # the robot clips cmd_vel to the box: standing still must stay reachable
    command = resolved_env_config(deploy_spec()["env_overrides"])["command"]
    for key in ("vx", "vy", "wz"):
        lo, hi = command[key]
        assert lo <= 0.0 <= hi, key
