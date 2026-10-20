"""What a deployed policy has to know: observation layout, action mapping,
joints, gains, commands. Built from the live env, never hand-written."""

import dataclasses
import math
from typing import Any

import numpy as np

from config.schema import Robot
from env.functional import FunctionalEnv
from wojtek_env.envs import env as wojtek_joystick
from wojtek_env.envs.symmetry import COMPONENT_SIZES

LEG_OFFSETS = (0.0, 1.0, 2.0)  # dsecond multiplier of a leg's three joints
GAIT_STAND_SPEED = 0.05  # m/s: below this the Wojtek gait clock is frozen
YAW_SPEED_WEIGHT = 0.3  # wz share of the Wojtek gait-clock speed


@dataclasses.dataclass(frozen=True)
class ObsComponent:
    name: str
    size: int
    units: str
    source: str
    noise: float | None = None


@dataclasses.dataclass(frozen=True)
class Joint:
    name: str
    default: float
    target_min: float
    target_max: float
    kp: float
    kd: float
    torque_limit: float | None
    action_scale: float


@dataclasses.dataclass(frozen=True)
class HeightStance:
    """Wojtek: the zero-action pose follows the commanded standing height."""

    height_index: int
    knots_height: tuple[float, ...]
    knots_dsecond: tuple[float, ...]
    offsets: tuple[float, ...]
    home: tuple[float, ...]
    stance_min: tuple[float, ...]
    stance_max: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class GaitClock:
    freq: tuple[float, float]
    trot_band: tuple[float, float]
    walk_phase: tuple[float, ...]
    trot_phase: tuple[float, ...]
    max_speed: float
    dt: float


@dataclasses.dataclass(frozen=True)
class Command:
    names: tuple[str, ...]
    units: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    notes: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Contract:
    robot: str
    obs: tuple[ObsComponent, ...]
    frames: int
    joints: tuple[Joint, ...]
    ctrl_dt: float
    command: Command
    conditions: tuple[str, ...]
    stance: HeightStance | None = None
    gait_clock: GaitClock | None = None
    action_filter: float = 0.0

    @property
    def frame_dim(self) -> int:
        return sum(c.size for c in self.obs)

    @property
    def obs_dim(self) -> int:
        return self.frames * self.frame_dim

    @property
    def action_dim(self) -> int:
        return len(self.joints)


def index_ranges(contract: Contract) -> list[tuple[ObsComponent, int, int]]:
    """Each component with its [start, stop) in (the newest) frame."""
    out, start = [], 0
    for component in contract.obs:
        out.append((component, start, start + component.size))
        start += component.size
    return out


def build_contract(robot: Robot, env: FunctionalEnv) -> Contract:
    if robot == Robot.wojtek:
        contract = _wojtek_contract(env)
    else:
        contract = _playground_contract(robot, env)
    if contract.obs_dim != env.observation_size.actor:
        raise ValueError(
            f"{robot}: the contract describes {contract.obs_dim} actor "
            f"observations, the env produces {env.observation_size.actor}; "
            "update export/contract.py for the robot's observation layout"
        )
    return contract


def _scale_vector(scale: Any, n: int) -> np.ndarray:
    scale = np.asarray(scale, dtype=np.float64)
    if scale.ndim > 0 and scale.size == 3:
        scale = np.tile(scale, n // 3)
    return np.broadcast_to(scale, (n,)).copy()


def _torque_limits(m: Any) -> list[float | None]:
    return [
        float(m.actuator_forcerange[i, 1])
        if m.actuator_forcelimited[i]
        else None
        for i in range(m.nu)
    ]


def _gains(m: Any) -> tuple[np.ndarray, np.ndarray]:
    """kp and the velocity damping: the actuator's kv plus joint damping."""
    kp = np.asarray(m.actuator_gainprm[:, 0], dtype=np.float64)
    dof = np.asarray(
        [m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]
    )
    kd = -np.asarray(m.actuator_biasprm[:, 2]) + np.asarray(m.dof_damping)[dof]
    return kp, np.asarray(kd, dtype=np.float64)


def _joints(
    m: Any,
    default: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    scale: np.ndarray,
) -> tuple[Joint, ...]:
    kp, kd = _gains(m)
    torque = _torque_limits(m)
    return tuple(
        Joint(
            name=m.actuator(i).name,
            default=float(default[i]),
            target_min=float(lo[i]),
            target_max=float(hi[i]),
            kp=float(kp[i]),
            kd=float(kd[i]),
            torque_limit=torque[i],
            action_scale=float(scale[i]),
        )
        for i in range(m.nu)
    )


# --- Wojtek ---------------------------------------------------------------

WOJTEK_OBS = {
    "gyro": (
        "rad/s",
        "IMU angular velocity, body frame (x, y, z)",
    ),
    "gravity": (
        "unit vector",
        (
            "gravity direction in the body frame: rotate (0, 0, -1) by the "
            "inverse of the IMU orientation quaternion; upright = (0, 0, -1)"
        ),
    ),
    "joint_pos": (
        "rad",
        (
            "joint encoder angle minus `default` of the joint table "
            "(joint order below)"
        ),
    ),
    "joint_vel": ("rad/s", "joint encoder velocity (joint order below)"),
    "last_act": (
        "[-1, 1]",
        "the `action` output of the previous control step (zeros at start)",
    ),
    "command": (
        "m/s, m/s, rad/s, m",
        (
            "operator setpoint [vx, vy, wz, height] in the body frame "
            "(see Commands)"
        ),
    ),
    "phase": (
        "unitless",
        (
            "gait clock: cos and sin of the 4 leg phases, computed on the "
            "robot (see Gait clock)"
        ),
    ),
}


def _wojtek_contract(env: FunctionalEnv) -> Contract:
    base = env.base
    cfg = base._config
    m = base.mj_model
    names = base.actor_obs_names
    unknown = [n for n in names if n not in WOJTEK_OBS]
    if unknown:
        raise ValueError(
            f"obs.state components {unknown} have no export description; "
            f"supported: {sorted(WOJTEK_OBS)}. Add them to "
            "export/contract.py WOJTEK_OBS"
        )
    obs = tuple(
        ObsComponent(
            name=n,
            size=COMPONENT_SIZES[n],
            units=WOJTEK_OBS[n][0],
            source=WOJTEK_OBS[n][1],
            noise=_noise(cfg.obs_noise.get(n, 0.0)),
        )
        for n in names
    )
    scale = _scale_vector(cfg.action_scale, m.nu)
    home = np.asarray(base._home_ctrl, dtype=np.float64)
    lo = np.asarray(base._target_lo, dtype=np.float64)
    hi = np.asarray(base._target_hi, dtype=np.float64)
    command = _wojtek_command(cfg)
    height_index = _command_index(obs)
    return Contract(
        robot=Robot.wojtek.value,
        obs=obs,
        frames=1,
        joints=_joints(m, home, lo, hi, scale),
        ctrl_dt=float(base.dt),
        command=command,
        conditions=_wojtek_conditions(cfg, float(m.opt.timestep)),
        stance=_height_stance(base, height_index + 3),
        gait_clock=_gait_clock(base) if "phase" in names else None,
        action_filter=float(cfg.action_filter),
    )


def _noise(value: float) -> float | None:
    return float(value) if value else None


def _command_index(obs: tuple[ObsComponent, ...]) -> int:
    start = 0
    for component in obs:
        if component.name == "command":
            return start
        start += component.size
    raise ValueError("the actor observations have no `command` component")


def _height_stance(base: Any, height_index: int) -> HeightStance:
    if base._config.get("real_pose_ref", False):
        heights = np.asarray(base._anchor_heights)
        dsecond = np.asarray(base._anchor_dsecond)
    else:
        heights = np.asarray(wojtek_joystick.HEIGHT_TABLE)
        dsecond = np.asarray(wojtek_joystick.DSECOND_TABLE)
    crange = np.asarray(base._ctrlrange, dtype=np.float64)
    return HeightStance(
        height_index=height_index,
        knots_height=tuple(float(x) for x in heights),
        knots_dsecond=tuple(float(x) for x in dsecond),
        offsets=LEG_OFFSETS * (base.action_size // 3),
        home=tuple(float(x) for x in np.asarray(base._home_ctrl)),
        stance_min=tuple(float(x) for x in crange[:, 0]),
        stance_max=tuple(float(x) for x in crange[:, 1]),
    )


def _gait_clock(base: Any) -> GaitClock:
    cfg = base._config
    vx = cfg.command.vx
    return GaitClock(
        freq=(float(cfg.gait.freq[0]), float(cfg.gait.freq[1])),
        trot_band=(
            float(cfg.gait.trot_band[0]),
            float(cfg.gait.trot_band[1]),
        ),
        walk_phase=tuple(float(p) for p in wojtek_joystick.WALK_PHASE),
        trot_phase=tuple(float(p) for p in wojtek_joystick.TROT_PHASE),
        max_speed=float(max(abs(vx[0]), abs(vx[1]))),
        dt=float(base.dt),
    )


def _wojtek_command(cfg: Any) -> Command:
    c = cfg.command
    bias = c.get("terrain_bias", None)
    biased = bias is not None and bool(bias.enable)

    def prob(name: str) -> float:
        if biased and name in bias:
            return float(bias[name])
        return float(c.get(name, 0.0))

    draws = [
        ("pure yaw (vx = vy = 0)", prob("pure_wz_prob"), None),
        ("pure strafe (vx = wz = 0)", prob("pure_vy_prob"), None),
        ("forward arc (vy = 0)", prob("arc_prob"), ("vx", c.arc_vx)),
        (
            "slow straight (vy = wz = 0)",
            prob("pure_slow_prob"),
            ("vx", c.slow_vx),
        ),
        (
            "fast straight (vy = wz = 0)",
            prob("pure_fast_prob"),
            ("vx", c.fast_vx),
        ),
        ("backward (vy = wz = 0)", prob("pure_back_prob"), ("vx", c.back_vx)),
        ("stand (vx = vy = wz = 0)", prob("zero_prob"), None),
    ]
    notes = [
        (
            f"held for {int(c.resample_steps)} control steps "
            f"({c.resample_steps * float(cfg.ctrl_dt):g} s), then redrawn"
        ),
        (
            "the training draw is the uniform box above, then these "
            "overwrite it in order (probability, later ones win):"
        ),
    ]
    for label, p, rng in draws:
        if not p:
            continue
        span = f", {rng[0]} in [{rng[1][0]:g}, {rng[1][1]:g}]" if rng else ""
        notes.append(f"  - {label}: p = {p:g}{span}")
    notes.append(
        "`height` is always live, also for a stand command: the robot "
        "stands at that trunk height"
    )
    return Command(
        names=("vx", "vy", "wz", "height"),
        units=("m/s", "m/s", "rad/s", "m"),
        low=(
            float(c.vx[0]),
            float(c.vy[0]),
            float(c.wz[0]),
            float(c.height[0]),
        ),
        high=(
            float(c.vx[1]),
            float(c.vy[1]),
            float(c.wz[1]),
            float(c.height[1]),
        ),
        notes=tuple(notes),
    )


def _wojtek_conditions(cfg: Any, sim_dt: float) -> tuple[str, ...]:
    out = [
        "actor observation noise (uniform +-): "
        + ", ".join(f"{k} {v:g}" for k, v in cfg.obs_noise.items() if v)
    ]
    if cfg.action_filter:
        out.append(
            f"action low-pass: `filt = {cfg.action_filter:g} * filt_prev + "
            f"{1 - cfg.action_filter:g} * action` before the joint targets"
        )
    if cfg.action_delay:
        out.append(f"action delay: {cfg.action_delay} control step(s)")
    if cfg.latency.enable:
        out.append(
            f"control latency: {cfg.latency.min_substeps}-"
            f"{cfg.latency.max_substeps} physics substeps of {sim_dt * 1e3:g} "
            "ms before a new target takes effect"
        )
    if cfg.encoder.enable:
        out.append(
            "encoder offset: constant per-joint error "
            f"U(+-{cfg.encoder.range:g}) rad on the observed joint angle"
        )
    if cfg.push.enable:
        out.append(
            f"random horizontal push of {cfg.push.vel:g} m/s every "
            f"{cfg.push.interval_steps} control steps"
        )
    if cfg.symmetry.enable:
        out.append(
            "left/right mirror augmentation (training only; the robot "
            "always sees real observations)"
        )
    return tuple(out)


# --- Playground robots ------------------------------------------------------

# robots whose step clips the targets to explicit bounds on the env
EXPLICIT_CLIP = {Robot.barkour, Robot.spot}

GRAVITY_BODY = (
    "gravity direction in the IMU frame: rotate (0, 0, -1) by the inverse "
    "of the IMU orientation; upright = (0, 0, -1)"
)
UP_WORLD = (
    "the IMU frame's z axis in the world frame (third column of the IMU "
    "rotation matrix); upright = (0, 0, 1)"
)


def _go1_obs(cfg: Any) -> tuple[ObsComponent, ...]:
    noise = cfg.noise_config
    level = float(noise.level)

    def n(name: str) -> float | None:
        return _noise(level * float(noise.scales[name]))

    return (
        ObsComponent(
            "linvel",
            3,
            "m/s",
            "base linear velocity in the IMU frame; not an IMU output: "
            "needs a state estimator (leg odometry / Kalman filter)",
            n("linvel"),
        ),
        ObsComponent("gyro", 3, "rad/s", "IMU angular velocity", n("gyro")),
        ObsComponent("gravity", 3, "unit vector", GRAVITY_BODY, n("gravity")),
        ObsComponent(
            "joint_pos",
            12,
            "rad",
            "joint encoder angle minus `default` of the joint table",
            n("joint_pos"),
        ),
        ObsComponent(
            "joint_vel", 12, "rad/s", "joint encoder velocity", n("joint_vel")
        ),
        ObsComponent(
            "last_act", 12, "[-1, 1]", "previous `action` (zeros at start)"
        ),
        ObsComponent(
            "command",
            3,
            "m/s, m/s, rad/s",
            "operator setpoint [vx, vy, wz] in the base frame",
        ),
    )


def _spot_obs(cfg: Any) -> tuple[ObsComponent, ...]:
    scales = cfg.obs_noise.scales
    history = int(cfg.history_len)
    return (
        ObsComponent(
            "gyro", 3, "rad/s", "IMU angular velocity", _noise(scales.gyro)
        ),
        ObsComponent(
            "gravity", 3, "unit vector", UP_WORLD, _noise(scales.gravity)
        ),
        ObsComponent(
            "joint_pos",
            12,
            "rad",
            "joint encoder angle minus `default` of the joint table",
            _noise(scales.joint_pos),
        ),
        ObsComponent(
            "qpos_error_history",
            12 * history,
            "rad",
            f"last {history} control steps of (encoder angle - the "
            "`joint_targets` sent at that step), newest first, 12 per step",
            _noise(scales.joint_pos),
        ),
        ObsComponent(
            "feet_pos",
            12,
            "m",
            "foot positions (x, y, z per foot) relative to the trunk, from "
            "forward kinematics",
            _noise(max(scales.feet_pos)),
        ),
        ObsComponent(
            "last_act", 12, "[-1, 1]", "previous `action` (zeros at start)"
        ),
        ObsComponent(
            "command",
            3,
            "m/s, m/s, rad/s",
            "operator setpoint [vx, vy, wz] in the base frame",
        ),
    )


def _barkour_obs(cfg: Any) -> tuple[ObsComponent, ...]:
    noise = _noise(cfg.obs_noise)
    return (
        ObsComponent(
            "gyro_z",
            1,
            "0.25 rad/s",
            "IMU yaw rate times 0.25",
            noise,
        ),
        ObsComponent("gravity", 3, "unit vector", UP_WORLD, noise),
        ObsComponent(
            "command",
            3,
            "2 m/s, 2 m/s, 0.25 rad/s",
            "operator setpoint [vx, vy, wz] scaled by [2, 2, 0.25]",
            noise,
        ),
        ObsComponent(
            "joint_pos",
            12,
            "rad",
            "joint encoder angle minus `default` of the joint table",
            noise,
        ),
        ObsComponent(
            "last_act",
            12,
            "[-1, 1]",
            "previous `action` (zeros at start)",
            noise,
        ),
    )


def _playground_obs(
    robot: Robot, cfg: Any
) -> tuple[tuple[ObsComponent, ...], int]:
    if robot in (Robot.go1, Robot.go2):
        return _go1_obs(cfg), 1
    if robot == Robot.spot:
        return _spot_obs(cfg), 1
    if robot == Robot.barkour:
        return _barkour_obs(cfg), BARKOUR_HISTORY
    raise ValueError(f"no export description for robot {robot!r}")


# barkour/joystick.py keeps 15 frames of its 31-value observation
BARKOUR_HISTORY = 15


def _playground_contract(robot: Robot, env: FunctionalEnv) -> Contract:
    base = env.base
    m = base.mj_model
    cfg = base._config
    obs, frames = _playground_obs(robot, cfg)
    default = np.asarray(base._default_pose, dtype=np.float64)
    lo, hi = _target_bounds(robot, base)
    scale = _scale_vector(cfg.action_scale, m.nu)
    low, high = np.asarray(base._cmd_low), np.asarray(base._cmd_high)
    command = Command(
        names=("vx", "vy", "wz"),
        units=("m/s", "m/s", "rad/s"),
        low=tuple(float(x) for x in low),
        high=tuple(float(x) for x in high),
        notes=(
            "drawn uniformly from the box above, redrawn during the episode",
            (
                "the forward half of the robot's native command range "
                "(vx >= 0) is what was trained on terrain"
            ),
        ),
    )
    return Contract(
        robot=robot.value,
        obs=obs,
        frames=frames,
        joints=_joints(m, default, lo, hi, scale),
        ctrl_dt=float(base.dt),
        command=command,
        conditions=(
            "observation noise (uniform +-) as in the Noise column",
            "no action filter, no extra latency",
        ),
    )


def _target_bounds(robot: Robot, base: Any) -> tuple[np.ndarray, np.ndarray]:
    m = base.mj_model
    lo = np.full(m.nu, -math.inf)
    hi = np.full(m.nu, math.inf)
    if robot in EXPLICIT_CLIP:
        lo = np.asarray(base._lowers, dtype=np.float64)
        hi = np.asarray(base._uppers, dtype=np.float64)
    limited = np.asarray(m.actuator_ctrllimited, dtype=bool)
    crange = np.asarray(m.actuator_ctrlrange, dtype=np.float64)
    lo = np.where(limited, np.maximum(lo, crange[:, 0]), lo)
    hi = np.where(limited, np.minimum(hi, crange[:, 1]), hi)
    return lo, hi
