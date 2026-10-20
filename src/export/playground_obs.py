"""Actor observation layout of a Playground robot, measured on its env.

Each input signal the env feeds its observation (a sensor getter, the joint
angles, an `info` entry) is shifted on its own; the observation entries that
move are that component's slice, and by how much its scale. The contract thus
follows the env: a swap or resize of components changes it, and a layout the
channels below do not tile fails instead of passing on the total size."""

import contextlib
import dataclasses
import functools
from collections.abc import Iterator, Sequence
from typing import Any

import jax
import numpy as np

from config.schema import Robot

PROBE_STEP = 1.0
MOVED = 1e-2  # an entry that moves by less is noise-free numerical residue

GRAVITY_BODY = (
    "gravity direction in the IMU frame: rotate (0, 0, -1) by the inverse "
    "of the IMU orientation; upright = (0, 0, -1)"
)
UP_WORLD = (
    "the IMU frame's z axis in the world frame (third column of the IMU "
    "rotation matrix); upright = (0, 0, 1)"
)
JOINT_POS = "joint encoder angle minus `default` of the joint table"
LAST_ACT = "previous `action` (zeros at start)"
COMMAND = "operator setpoint [vx, vy, wz] in the base frame"


@dataclasses.dataclass(frozen=True)
class Signal:
    """What a probe shifts: env getters (by method name), the joint angles
    and velocities, entries of `state.info`."""

    getters: tuple[str, ...] = ()
    info: tuple[str, ...] = ()
    qpos: bool = False
    qvel: bool = False


@dataclasses.dataclass(frozen=True)
class Channel:
    name: str
    units: tuple[str, ...]
    source: str
    signal: Signal
    noise: float = 0.0


@dataclasses.dataclass(frozen=True)
class Placement:
    channel: Channel
    start: int
    stop: int
    gains: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class Layout:
    placements: tuple[Placement, ...]
    frames: int


def _shifted_call(getter: Any, *args: Any) -> Any:
    return getter(*args) + PROBE_STEP


@contextlib.contextmanager
def _shifted_getters(base: Any, names: Sequence[str]) -> Iterator[None]:
    try:
        for name in names:
            if not hasattr(base, name):
                raise AttributeError(
                    f"{type(base).__name__} has no getter {name!r}; update "
                    "the channels in export/playground_obs.py"
                )
            setattr(
                base,
                name,
                functools.partial(_shifted_call, getattr(base, name)),
            )
        yield
    finally:
        for name in names:
            vars(base).pop(name, None)


def _shifted_state(state: Any, signal: Signal) -> Any:
    data, info = state.data, dict(state.info)
    if signal.qpos:
        data = data.replace(qpos=data.qpos.at[7:].add(PROBE_STEP))
    if signal.qvel:
        data = data.replace(qvel=data.qvel.at[6:].add(PROBE_STEP))
    for key in signal.info:
        info[key] = info[key] + PROBE_STEP
    return state.replace(data=data, info=info)


def _moved(base: Any, state: Any, reference: np.ndarray, ch: Channel) -> Any:
    with _shifted_getters(base, ch.signal.getters):
        shifted = np.asarray(base.actor_obs(_shifted_state(state, ch.signal)))
    delta = shifted - reference
    index = np.flatnonzero(np.abs(delta) > MOVED)
    if index.size == 0:
        raise ValueError(
            f"{type(base).__name__}: shifting {ch.name!r} ({ch.signal}) moves "
            "no actor observation; update the channels in "
            "export/playground_obs.py"
        )
    if np.any(np.diff(index) != 1):
        raise ValueError(
            f"{type(base).__name__}: {ch.name!r} moves the non-contiguous "
            f"observation entries {index.tolist()}; split it into channels"
        )
    return index, tuple(
        round(float(g), 3) for g in np.abs(delta[index]) / PROBE_STEP
    )


def _carried(base: Any, state: Any, reference: np.ndarray) -> np.ndarray:
    """Entries that follow the previous observation: older frames."""
    previous = {**state.obs, "state": state.obs["state"] + PROBE_STEP}
    shifted = np.asarray(base.actor_obs(state.replace(obs=previous)))
    return np.flatnonzero(np.abs(shifted - reference) > MOVED)


def measure(base: Any, channels: Sequence[Channel]) -> Layout:
    """Where each channel sits in the newest frame of `base.actor_obs`, and
    how many frames the observation stacks."""
    name = type(base).__name__
    state = base.reset(jax.random.key(0))
    reference = np.asarray(base.actor_obs(state))
    placements = []
    for ch in channels:
        index, gains = _moved(base, state, reference, ch)
        placements.append(
            Placement(ch, int(index[0]), int(index[-1]) + 1, gains)
        )
    placements.sort(key=lambda p: p.start)
    edge = 0
    for p in placements:
        if p.start != edge:
            raise ValueError(
                f"{name}: observation entries [{min(edge, p.start)}:"
                f"{max(edge, p.start)}] before {p.channel.name!r} belong to "
                "no channel or to two; update the channels in "
                "export/playground_obs.py"
            )
        edge = p.stop
    older = _carried(base, state, reference)
    if not np.array_equal(older, np.arange(edge, reference.size)):
        raise ValueError(
            f"{name}: the channels cover the entries [0:{edge}] of "
            f"{reference.size}, but the entries that carry older frames are "
            f"{older.tolist()}; update the channels in "
            "export/playground_obs.py"
        )
    if reference.size % edge:
        raise ValueError(
            f"{name}: {reference.size} observations are not whole frames of "
            f"{edge}"
        )
    return Layout(tuple(placements), reference.size // edge)


def scaled_units(units: Sequence[str], gains: Sequence[float]) -> str:
    """`units` of the physical signal as the observation carries it: an entry
    scaled by `gain` reads `gain units`."""
    parts = list(units)
    if len(parts) == 1:
        parts *= len(gains)
    if len(parts) != len(gains):
        raise ValueError(
            f"units {tuple(units)} name {len(parts)} values for "
            f"{len(gains)} entries"
        )
    scaled = [u if g == 1 else f"{g:g} {u}" for u, g in zip(parts, gains)]
    if len(set(scaled)) == 1:
        return scaled[0]
    return ", ".join(scaled)


def channels(robot: Robot, cfg: Any) -> tuple[Channel, ...]:
    if robot in (Robot.go1, Robot.go2):
        return _go1_channels(cfg)
    if robot == Robot.spot:
        return _spot_channels(cfg)
    if robot == Robot.barkour:
        return _barkour_channels(cfg)
    raise ValueError(f"no export description for robot {robot!r}")


def _go1_channels(cfg: Any) -> tuple[Channel, ...]:
    noise = cfg.noise_config
    level = float(noise.level)

    def n(name: str) -> float:
        return level * float(noise.scales[name])

    return (
        Channel(
            "linvel",
            ("m/s",),
            "base linear velocity in the IMU frame; not an IMU output: "
            "needs a state estimator (leg odometry / Kalman filter)",
            Signal(getters=("get_local_linvel",)),
            n("linvel"),
        ),
        Channel(
            "gyro",
            ("rad/s",),
            "IMU angular velocity",
            Signal(getters=("get_gyro",)),
            n("gyro"),
        ),
        Channel(
            "gravity",
            ("unit vector",),
            GRAVITY_BODY,
            Signal(getters=("get_gravity",)),
            n("gravity"),
        ),
        Channel(
            "joint_pos", ("rad",), JOINT_POS, Signal(qpos=True), n("joint_pos")
        ),
        Channel(
            "joint_vel",
            ("rad/s",),
            "joint encoder velocity",
            Signal(qvel=True),
            n("joint_vel"),
        ),
        Channel("last_act", ("[-1, 1]",), LAST_ACT, Signal(info=("last_act",))),
        Channel(
            "command",
            ("m/s", "m/s", "rad/s"),
            COMMAND,
            Signal(info=("command",)),
        ),
    )


def _spot_channels(cfg: Any) -> tuple[Channel, ...]:
    scales = cfg.obs_noise.scales
    return (
        Channel(
            "gyro",
            ("rad/s",),
            "IMU angular velocity",
            Signal(getters=("get_gyro",)),
            float(scales.gyro),
        ),
        Channel(
            "gravity",
            ("unit vector",),
            UP_WORLD,
            Signal(getters=("get_gravity",)),
            float(scales.gravity),
        ),
        # the history entry is encoder - target: shifting both leaves it
        Channel(
            "joint_pos",
            ("rad",),
            JOINT_POS,
            Signal(qpos=True, info=("motor_targets",)),
            float(scales.joint_pos),
        ),
        Channel(
            "qpos_error_history",
            ("rad",),
            "per control step: encoder angle - the `joint_targets` sent at "
            "that step, newest first, 12 per step",
            Signal(info=("qpos_error_history", "motor_targets")),
            float(scales.joint_pos),
        ),
        Channel(
            "feet_pos",
            ("m",),
            "foot positions (x, y, z per foot) relative to the trunk, from "
            "forward kinematics",
            Signal(getters=("get_feet_pos",)),
            float(max(scales.feet_pos)),
        ),
        Channel("last_act", ("[-1, 1]",), LAST_ACT, Signal(info=("last_act",))),
        Channel(
            "command",
            ("m/s", "m/s", "rad/s"),
            COMMAND,
            Signal(info=("command",)),
        ),
    )


def _barkour_channels(cfg: Any) -> tuple[Channel, ...]:
    noise = float(cfg.obs_noise)
    return (
        Channel(
            "gyro_z",
            ("rad/s",),
            "IMU yaw rate times the factor in units",
            Signal(getters=("_get_localrpyrate",)),
            noise,
        ),
        Channel(
            "gravity",
            ("unit vector",),
            UP_WORLD,
            Signal(getters=("_get_gravity",)),
            noise,
        ),
        Channel(
            "command",
            ("m/s", "m/s", "rad/s"),
            "operator setpoint [vx, vy, wz] times the factors in units",
            Signal(info=("command",)),
            noise,
        ),
        Channel("joint_pos", ("rad",), JOINT_POS, Signal(qpos=True), noise),
        Channel(
            "last_act",
            ("[-1, 1]",),
            LAST_ACT,
            Signal(info=("last_act",)),
            noise,
        ),
    )
