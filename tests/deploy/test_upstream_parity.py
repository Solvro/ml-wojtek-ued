import json
from pathlib import Path
from typing import Any

import pytest

from deploy.contract import CONSUMED_KEYS, TRAINING_ONLY_KEYS
from scripts.brax_ued import UPSTREAM_RUNS, parse_args, run_spec
from wojtek_env.envs import registry

PROFILES = list(UPSTREAM_RUNS)
# training-only keys every robot profile sets its own way (docs/deploy.md);
# sim only in num_envs (deploy) or the terrain contact buffers (robot)
DEVIATIONS = {"terrain", "sim"}


def snapshot(profile: str) -> dict[str, Any]:
    path = Path(__file__).parent / f"upstream_{profile}.json"
    return json.loads(path.read_text())


def profile_spec(profile: str) -> dict[str, Any]:
    flags = ["--mode", "accel", "--name", "t", "--profile", profile]
    return run_spec(parse_args(flags))


def resolved_env_config(profile: str) -> dict[str, Any]:
    config = registry.TASKS["joystick"][1]()
    registry._apply_overrides(config, profile_spec(profile)["env_overrides"])
    return json.loads(json.dumps(config.to_dict()))


@pytest.mark.parametrize("profile", PROFILES)
def test_consumed_keys_match_upstream(profile: str) -> None:
    ours = resolved_env_config(profile)
    upstream = snapshot(profile)["env_config"]
    for key in CONSUMED_KEYS:
        assert ours[key] == upstream[key], key


@pytest.mark.parametrize("profile", PROFILES)
def test_training_only_keys_match_upstream_except_documented(profile: str) -> None:
    ours = resolved_env_config(profile)
    upstream = snapshot(profile)["env_config"]
    assert set(ours) == set(upstream) == CONSUMED_KEYS | TRAINING_ONLY_KEYS
    for key in TRAINING_ONLY_KEYS - DEVIATIONS:
        assert ours[key] == upstream[key], key


def test_deploy_sim_matches_upstream_but_the_batch_size() -> None:
    # the shared tiled arena lets every env run on upstream's warp backend
    ours = resolved_env_config("deploy")["sim"]
    del ours["num_envs"]
    assert ours == snapshot("deploy")["env_config"]["sim"]


def test_robot_sim_is_upstream_backend_with_terrain_contact_buffers() -> None:
    # flat_quiet_v6 trains on the flat floor (32 contacts, 320 rows per
    # world); the heightfield arena needs terrain_blind_v4_4d's buffers
    ours = resolved_env_config("robot")["sim"]
    del ours["num_envs"]
    terrain = snapshot("deploy")["env_config"]["sim"]
    buffers = {k: terrain[k] for k in ("naconmax_per_env", "njmax")}
    assert ours == {**snapshot("robot")["env_config"]["sim"], **buffers}


@pytest.mark.parametrize("profile", PROFILES)
def test_terrain_is_the_deploy_accel_arena(profile: str) -> None:
    terrain = resolved_env_config(profile)["terrain"]
    assert terrain["enable"] is True
    assert terrain["arena"] == "accel_deploy"


@pytest.mark.parametrize("profile", PROFILES)
def test_gyro_bias_and_vibration_stay_off(profile: str) -> None:
    noise = resolved_env_config(profile)["obs_noise"]
    assert noise["gyro_bias"] == 0.0
    assert noise["gyro_vib"] == 0.0


@pytest.mark.parametrize("profile", PROFILES)
def test_network_matches_upstream(profile: str) -> None:
    assert profile_spec(profile)["network"] == snapshot(profile)["network"]


@pytest.mark.parametrize("profile", PROFILES)
def test_dr_matches_upstream(profile: str) -> None:
    dr = json.loads(json.dumps(profile_spec(profile)["dr_config"]))
    assert dr == snapshot(profile)["dr"]


@pytest.mark.parametrize("profile", PROFILES)
def test_cmd_vel_box_contains_zero(profile: str) -> None:
    # the robot clips cmd_vel to the box: standing still must stay reachable
    command = resolved_env_config(profile)["command"]
    for key in ("vx", "vy", "wz"):
        lo, hi = command[key]
        assert lo <= 0.0 <= hi, key


def test_robot_profile_has_the_rpi_policys_established_keys() -> None:
    # w01-tek 833b9ca (origin/michal/baseacc-default-policy) on
    # wojtek_flat_quiet_baseacc_s0, the policy the RPi drop-in loads: schema
    # 2, joystick, 4-D command, kp=40 kd=0.8 max_torque=12, tau_ff +-3, no
    # action filter, IMU-blind obs (docs/deploy.md#robot-profile)
    contract = snapshot("robot")["contract"]
    assert contract["schema_version"] == 2
    assert contract["task"] == "joystick"
    assert contract["pd"] == {"kp": 40.0, "kd": 0.8, "max_torque": 12.0}
    assert contract["tau_ff"] == {"enable": True, "scale": 3.0}
    assert contract["action_size"] == 24
    assert contract["action_filter"] == 0.0
    assert contract["obs_layout"] == [
        "joint_pos:12",
        "joint_vel:12",
        "last_act:12",
        "command:4",
    ]
