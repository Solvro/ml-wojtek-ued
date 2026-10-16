from typing import Any


def merge_overrides(*overrides: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for override in overrides:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge_overrides(result[key], value)
            else:
                result[key] = value
    return result


def accel_arena_overrides(pad_radius: float, arena: str = "accel") -> dict[str, Any]:
    # every env shares the tiled arena's hfield_data, so warp is valid (auto:
    # warp on a GPU); its contact/constraint buffers are deploy's (upstream
    # terrain_blind_v4_4d), sized for terrain
    return {
        "terrain": {"enable": True, "arena": arena, "pad_radius": pad_radius},
        "sim": {"naconmax_per_env": 116, "njmax": 1024},
    }


def deploy_overrides() -> dict[str, Any]:
    # task.env of w01-tek e44c1a9 `+experiment=terrain_blind_v4_4d`, resolved
    # by its Hydra config (docs/deploy.md); keep verbatim, the deploy runtime
    # and tests/deploy/upstream_terrain_blind_v4_4d.json depend on it
    return {
        "obs": {
            "state": (
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_act",
                "command",
                "phase",
            ),
            "privileged": (
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_act",
                "command",
                "linvel",
                "foot_contact",
                "height_scan_clean",
            ),
            "include": (
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_act",
                "command",
            ),
        },
        "latency": {"enable": True, "min_substeps": 0, "max_substeps": 5},
        "encoder": {"enable": True, "range": 0.02},
        "terrain": {
            "enable": True,
            "arena": "train",
            "flat_row": True,
            "spawn_yaw": True,
            "pad_jitter": 0.15,
            "init_level_frac": 0.5,
            "spawn_level": -1,
            "demote_fraction": 0.5,
            "spawn_mode": "feature",
            "spawn_grace_sec": 1.0,
            "pinned_frac": 0.2,
            "pinned_flat_frac": 0.1,
            "demote_strikes": 3,
            "stair_tread_range": (0.25, 0.45),
            "type_caps": {
                "pyramid_stairs": 0.7,
                "inverted_pyramid_stairs": 0.7,
                "discrete_obstacles": 0.8,
                "random_grid": 0.45,
            },
            "pad_radius": 0.3,
        },
        "height_scan": {
            "enable": True,
            "x_range": (0.65, 1.45),
            "y_range": (-0.3, 0.3),
            "nx": 5,
            "ny": 5,
            "clip": 0.3,
            "hold_steps": 3,
            "delay_steps": 1,
            "dark": False,
            "mask": {
                "enable": True,
                "mount": (0.32, 0.0, 0.07),
                "pitch_deg": 15.0,
                "hfov_deg": 90.7,
                "vfov_deg": 61.2,
                "min_depth": 0.3,
                "max_depth": 3.0,
                "occlusion": True,
                "occlusion_samples": 8,
                "occlusion_margin": 0.02,
            },
            "corrupt": {
                "enable": False,
                "noise_prob": 0.6,
                "drift_prob": 0.3,
                "blackout_prob": 0.1,
                "noise_std": 0.02,
                "drift_z": 0.05,
                "drift_tilt": 0.05,
                "dropout_prob": 0.05,
                "pitch_jitter_deg": 0.0,
                "mount_jitter": 0.0,
            },
        },
        "action_scale": (0.25, 0.5, 1.35),
        "pd_kp": 40.0,
        "pd_kd": 1.6,
        "max_torque": 32.0,
        "abduction_ctrl_limit": 0.44,
        "knee_target_max": 4.0,
        "symmetry": {"enable": True, "mirror_prob": 0.5},
        "no_progress": {
            "enable": True,
            "grace_sec": 2.0,
            "ema_sec": 1.0,
            "risk_below": 0.5,
            "p_max": 0.02,
            "terrain_grace_sec": 4.0,
            "terrain_p_max_scale": 0.5,
        },
        "command": {
            "vx": (-0.8, 1.2),
            "vy": (-0.5, 0.5),
            "wz": (-1.5, 1.5),
            "height": (0.1, 0.18),
            "zero_prob": 0.25,
            "pure_wz_prob": 0.4,
            "pure_wz_sticky": True,
            "pure_vy_prob": 0.2,
            "arc_prob": 0.2,
            "arc_vx": (0.3, 0.8),
            "pure_slow_prob": 0.1,
            "slow_vx": (0.15, 0.35),
            "pure_back_prob": 0.2,
            "back_vx": (-0.8, -0.2),
            "terrain_bias": {
                "enable": True,
                "zero_prob": 0.1,
                "pure_wz_prob": 0.1,
                "pure_vy_prob": 0.05,
                "pure_slow_prob": 0.3,
                "pure_back_prob": 0.1,
            },
        },
        "push": {"vel": 0.8},
        "gait": {"swing_height": 0.08, "air_time_cap": 0.35},
        "reward": {
            "tracking_rel_sigma": 0.5,
            "tracking_rel_floor_ang": 0.7,
            "tracking_far_weight": 0.25,
            "tracking_far_sigma": 2.5,
            "pose_leg_weight": 0.1,
            "tracking_product": True,
            "tracking_relative": True,
            "shaping_tracking_gate": True,
            "torque_limit_frac": 0.25,
            "orientation_tol_deg": 20.0,
            "apex_target": 0.05,
            "terrain_gate": {
                "enable": True,
                "floor": 0.15,
                "rough_ref": 0.05,
                "landing_soften": 0.5,
                "flat_pitch_tol_deg": 2.0,
                "flat_pitch_rough_cut": 0.15,
            },
            "scales": {
                "tracking_lin_vel": 4.0,
                "tracking_ang_vel": 2.4,
                "height_tracking": 1.0,
                "lin_vel_z": 0.0,
                "ang_vel_xy": 0.0,
                "orientation": -5.0,
                "flat_pitch": -10.0,
                "torques": -0.0006,
                "torque_rate": -0.02,
                "torque_limit": -0.2,
                "action_rate": -0.25,
                "action_accel": 0.0,
                "energy": 0.0,
                "pose": -1.0,
                "feet_air_time": 4.0,
                "feet_slip": -0.35,
                "feet_phase": 0.0,
                "contact_match": 0.0,
                "feet_apex": 5.0,
                "feet_landing": -0.5,
                "high_step": 4.0,
                "stand_still": -2.5,
                "stand_feet_down": -30.0,
                "termination": -1.0,
            },
        },
        "fall": {"on_base_contact": True},
        "sim": {"naconmax_per_env": 116, "njmax": 1024},
    }


def deploy_dr_config() -> dict[str, Any]:
    # top-level `dr` block of the same resolved experiment
    return {
        "com_offset": {"enable": False, "xy": 0.02, "z": 0.01},
        "joint_gains": {"enable": True, "gain_pct": 0.2, "kd_pct": 0.2},
        "dof": {
            "enable": False,
            "damping": (0.9, 1.1),
            "armature": (0.9, 1.1),
            "frictionloss": (0.9, 1.1),
        },
        "foot_friction": {"enable": True, "range": (0.4, 1.35)},
        "motor_strength": {"enable": True, "range": (0.5, 1.1)},
    }


def terrain_task_overrides() -> dict[str, Any]:
    # walk forward so the robot leaves the flat spawn pad and meets the level
    # within an episode; standing, spinning and backing up never reach it
    return {
        "command": {
            "vx": (0.2, 0.6),
            "vy": (-0.2, 0.2),
            "wz": (-0.4, 0.4),
            "terrain_bias": {
                "enable": True,
                "zero_prob": 0.05,
                "pure_wz_prob": 0.0,
                "pure_vy_prob": 0.0,
                "pure_slow_prob": 0.15,
                "pure_back_prob": 0.0,
            },
        },
    }


def movement_overrides() -> dict[str, Any]:
    return {
        "latency": {"enable": True},
        "encoder": {"enable": True},
        "symmetry": {"enable": True},
        "real_pose_ref": True,
        "action_filter": 0.2,
        "no_progress": {"enable": True},
        "gait": {"air_time_cap": 0.3},
        "command": {
            "pure_wz_prob": 0.1,
            "pure_vy_prob": 0.1,
            "arc_prob": 0.1,
            "pure_slow_prob": 0.1,
            "pure_fast_prob": 0.1,
            "pure_back_prob": 0.1,
        },
        "reward": {
            "tracking_relative": True,
            "tracking_product": True,
            "shaping_tracking_gate": True,
            "tracking_far_weight": 0.3,
            "orientation_tol_deg": 5.0,
            "scales": {
                "torque_rate": -1e-4,
                "feet_landing": -0.25,
                "stand_feet_down": -0.5,
                "torque_limit": -1e-2,
                "toggle_flat": -0.5,
            },
        },
    }


def terrain_overrides() -> dict[str, Any]:
    return {
        "command": {"terrain_bias": {"enable": True}},
        "fall": {"on_base_contact": True},
        "height_scan": {"enable": True, "corrupt": {"enable": True}},
        "obs": {
            "state": (
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_act",
                "command",
                "phase",
            ),
            "privileged": (
                "gyro",
                "gravity",
                "joint_pos",
                "joint_vel",
                "last_act",
                "command",
                "phase",
                "linvel",
                "foot_contact",
                "height_scan_clean",
            ),
        },
        "reward": {
            "terrain_gate": {"enable": True},
            "scales": {"flat_pitch": -2.0},
        },
    }
