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


def accel_arena_overrides(
    pad_radius: float, backend: str = "auto", arena: str = "accel"
) -> dict[str, Any]:
    # shared hfield_data makes warp valid; 116 clears the 84-contact floor of
    # 21 ground-colliding geoms (base.py), njmax leaves room for rough terrain
    return {
        "terrain": {"enable": True, "arena": arena, "pad_radius": pad_radius},
        "sim": {"backend": backend, "naconmax_per_env": 116, "njmax": 1024},
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
