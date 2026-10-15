"""Deployment contract (policy_meta.json, schema 2) from a live env.

Port of w01-tek `training/wojtek_rl/deploy_contract.py` @ e44c1a9 onto
`wojtek_env`; the key classification and every contract field are
upstream's. `runtime.py` (the robot's interpreter) reads the result:

  obs = concat(obs_layout components)
  filt = action_filter * filt + (1 - action_filter) * tanh_mlp(obs)
  motor_targets = clip(anchor_ctrl + filt * action_scale,
                       target_low, target_high)

Every env-config key is CONSUMED (shapes a contract field) or TRAINING_ONLY;
`build_contract` refuses a key it does not know, so a new env option with
deploy implications cannot ship unclassified.
"""

from typing import Any

import jax
import numpy as np

from wojtek_env.envs.env import DSECOND_TABLE, HEIGHT_TABLE

SCHEMA_VERSION = 2

# Past the knee (third-joint) singularity the four-bar can snap through and
# break the linkage; the runtime's clamp_knee safety option clips there.
KNEE_SINGULARITY = 3.2

CONSUMED_KEYS = {
    "ctrl_dt",
    "action_scale",
    "action_filter",
    "abduction_ctrl_limit",
    "knee_target_max",
    "max_torque",
    "pd_kp",
    "pd_kd",
    "obs",
    "command",
    # the height->anchor table, exported as "height_table"
    "real_pose_ref",
    # feed-forward torque head: 24 actions and a torque channel on the robot
    "tau_ff",
}

# Physics stepping, randomization, episode structure, rewards: nothing here
# changes what the robot does with the exported network at 50 Hz.
TRAINING_ONLY_KEYS = {
    "sim",
    "sim_dt",
    "episode_length",
    "obs_noise",
    "push",
    "latency",
    "encoder",
    "action_delay",
    "fall",
    "gait",
    "reward",
    "symmetry",
    # the actor sees no world position or heading, so the scene is free
    "terrain",
    # a scan-observing policy names its component in obs_layout instead
    "height_scan",
    "no_progress",
}

COMMAND_BOX_KEYS = ("vx", "vy", "wz", "height")
# sampling curriculum inside the trained box; the box itself is unchanged
COMMAND_TRAINING_KEYS = {
    "resample_steps",
    "zero_prob",
    "pure_wz_prob",
    "pure_vy_prob",
    "arc_prob",
    "arc_vx",
    "pure_slow_prob",
    "slow_vx",
    "pure_fast_prob",
    "fast_vx",
    "pure_back_prob",
    "back_vx",
    "terrain_bias",
    "pure_wz_sticky",
}


def check_config_covered(env_config: dict[str, Any]) -> None:
    unknown = set(env_config) - CONSUMED_KEYS - TRAINING_ONLY_KEYS
    if unknown:
        raise ValueError(
            f"env config key(s) {sorted(unknown)} are not classified in "
            "src/deploy/contract.py -- decide whether they affect deployment "
            "(contract field + runtime support, upstream first) or not (add to "
            "TRAINING_ONLY_KEYS) before exporting"
        )
    unknown_cmd = (
        set(env_config.get("command", {}))
        - set(COMMAND_BOX_KEYS)
        - COMMAND_TRAINING_KEYS
    )
    if unknown_cmd:
        raise ValueError(
            f"command config key(s) {sorted(unknown_cmd)} are not classified "
            "in src/deploy/contract.py (COMMAND_BOX_KEYS / "
            "COMMAND_TRAINING_KEYS)"
        )


def obs_layout(env: Any) -> list[tuple[str, int]]:
    state = env.reset(jax.random.PRNGKey(0))
    catalog = env._obs_catalog(state.data, state.info)
    return [(n, int(np.asarray(catalog[n]).shape[0])) for n in env.actor_obs_names]


def height_table(env: Any) -> dict[str, list[float]]:
    # real_pose_ref anchors with a table measured at construction; the
    # runtime must re-anchor live-height commands with the same one
    if env._config.get("real_pose_ref", False):
        return {
            "heights": np.asarray(env._anchor_heights, np.float32).tolist(),
            "dsecond": np.asarray(env._anchor_dsecond, np.float32).tolist(),
        }
    return {"heights": list(HEIGHT_TABLE), "dsecond": list(DSECOND_TABLE)}


def build_contract(env: Any, run: dict[str, Any], checkpoint: str = "") -> dict:
    """Schema-2 policy_meta for a WojtekJoystick built with the run's config.

    `run` holds run_name, task and env_config (the env's full resolved
    config, as upstream's run.json).
    """
    check_config_covered(run.get("env_config", {}))
    task = run.get("task", "joystick")
    if task != "joystick":
        raise NotImplementedError(
            f"deploy contract is defined for the joystick task only, got "
            f"'{task}' -- the robot runtime has no target pipeline for it"
        )

    layout = obs_layout(env)
    if any(n == "phase" for n, _ in layout):
        raise NotImplementedError(
            "this policy observes the gait clock ('phase'); the deploy "
            "runtime has no implementation of the speed-blended walk/trot "
            "clock -- train with --profile deploy"
        )

    c = env._config.command
    box = [tuple(getattr(c, k)) for k in COMMAND_BOX_KEYS]
    cmd_width = dict(layout).get("command", 0)
    if cmd_width != len(box):
        raise ValueError(
            f"command obs width {cmd_width} != command box dims {len(box)} "
            "-- COMMAND_BOX_KEYS is out of sync with _sample_command"
        )
    # dims past (vx, vy, wz) have no /cmd_vel source on the robot: the
    # runtime fills them with the box midpoint
    command_fill = [(lo + hi) / 2.0 for lo, hi in box[3:]]
    anchor = np.asarray(env._height_ctrl(command_fill[0]), np.float32)

    scale = np.asarray(env._config.action_scale, np.float32)
    if scale.ndim == 0:
        scale = np.full(12, float(scale), np.float32)
    elif scale.size == 3:
        scale = np.tile(scale, 4)

    m = env.mj_model
    tau_ff = bool(getattr(env, "_tau_ff_enabled", False))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": run["run_name"],
        "checkpoint": str(checkpoint),
        "task": task,
        "obs_size": int(sum(w for _, w in layout)),
        "action_size": int(env.action_size),
        "obs_layout": [f"{n}:{w}" for n, w in layout],
        "actuator_names": [m.actuator(i).name for i in range(m.nu)],
        "home_ctrl": np.asarray(env._home_ctrl, np.float32).tolist(),
        "anchor_ctrl": anchor.tolist(),
        "action_scale": scale.tolist(),
        "target_low": np.asarray(env._target_lo, np.float32).tolist(),
        "target_high": np.asarray(env._target_hi, np.float32).tolist(),
        # the anchor clips to the model ctrlrange, not the target bounds
        "ctrl_low": np.asarray(env._ctrlrange[:, 0], np.float32).tolist(),
        "ctrl_high": np.asarray(env._ctrlrange[:, 1], np.float32).tolist(),
        "command_low": [float(lo) for lo, _ in box],
        "command_high": [float(hi) for _, hi in box],
        "command_fill": [float(v) for v in command_fill],
        "height_table": height_table(env),
        "action_filter": float(env._config.action_filter),
        "ctrl_dt": float(env._config.ctrl_dt),
        "knee_singularity": KNEE_SINGULARITY,
        # the robot's servo config; the real driver must match it
        "pd": {
            "kp": float(m.actuator_gainprm[0, 0]),
            "kd": float(-m.actuator_biasprm[0, 2]),
            "max_torque": float(m.actuator_forcerange[0, 1]),
        },
        "tau_ff": {
            "enable": tau_ff,
            "scale": float(env._config.tau_ff.scale if tau_ff else 0.0),
        },
    }
