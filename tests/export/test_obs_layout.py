import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from config.schema import Robot
from export.contract import Contract, build_contract, index_ranges
from export.playground_obs import Channel, Signal, measure, scaled_units
from export.run import make_export_env
from wojtek_env import paths

Compose = Callable[..., Any]
Reader = Callable[[Any, Any], Any]

SMALL_ARENA = ("arena.tiles=2", "arena.reach=2.0", "arena.patch_radius=1.5")
STEPS = 4


def built(
    compose_cfg: Compose, robot: str, *overrides: str
) -> tuple[Any, Contract]:
    cfg = compose_cfg(
        "train",
        "mode=dr",
        "name=t",
        f"robot={robot}",
        "backend=jax",
        *overrides,
    )
    env, _, _ = make_export_env(cfg)
    return env.base, build_contract(Robot(robot), env)


def after_steps(base: Any) -> Any:
    """A state off the reset pose: every signal holds a distinct value."""
    state = base.reset(jax.random.key(0))
    step = jax.jit(base.step)
    for i in range(STEPS):
        action = jax.random.uniform(
            jax.random.key(i + 1), (base.action_size,), minval=-1.0, maxval=1.0
        )
        state = step(state, action)
    return state


def assert_slices_hold_the_env_signals(
    base: Any, contract: Contract, readers: dict[str, Reader]
) -> None:
    state = after_steps(base)
    obs = np.asarray(base.actor_obs(state))
    assert [c.name for c, _, _ in index_ranges(contract)] == list(readers)
    for component, start, stop in index_ranges(contract):
        expected = np.asarray(readers[component.name](base, state))
        assert expected.size == component.size, component.name
        tol = (component.noise or 0.0) + 1e-4
        np.testing.assert_allclose(
            obs[start:stop], expected.ravel(), atol=tol, err_msg=component.name
        )


def go1_readers() -> dict[str, Reader]:
    return {
        "linvel": lambda b, s: b.get_local_linvel(s.data),
        "gyro": lambda b, s: b.get_gyro(s.data),
        "gravity": lambda b, s: b.get_gravity(s.data),
        "joint_pos": lambda b, s: s.data.qpos[7:] - b._default_pose,
        "joint_vel": lambda b, s: s.data.qvel[6:],
        "last_act": lambda b, s: s.info["last_act"],
        "command": lambda b, s: s.info["command"],
    }


@pytest.mark.parametrize("robot", ["go1", "go2"])
def test_go1_family_slices_hold_their_signals(
    compose_cfg: Compose, robot: str
) -> None:
    base, contract = built(compose_cfg, robot, *SMALL_ARENA)
    assert contract.frames == 1
    assert [(c.name, c.size) for c in contract.obs] == [
        ("linvel", 3),
        ("gyro", 3),
        ("gravity", 3),
        ("joint_pos", 12),
        ("joint_vel", 12),
        ("last_act", 12),
        ("command", 3),
    ]
    assert_slices_hold_the_env_signals(base, contract, go1_readers())


def test_spot_slices_hold_their_signals(compose_cfg: Compose) -> None:
    base, contract = built(compose_cfg, "spot", *SMALL_ARENA)

    def history(b: Any, s: Any) -> Any:
        # `_get_obs` shifts the stored history by one step
        newest = s.data.qpos[7:] - s.info["motor_targets"]
        return jnp.roll(s.info["qpos_error_history"], 12).at[:12].set(newest)

    readers: dict[str, Reader] = {
        "gyro": lambda b, s: b.get_gyro(s.data),
        "gravity": lambda b, s: b.get_gravity(s.data),
        "joint_pos": lambda b, s: s.data.qpos[7:] - b._default_pose,
        "qpos_error_history": history,
        "feet_pos": lambda b, s: b.get_feet_pos(s.data),
        "last_act": lambda b, s: s.info["last_act"],
        "command": lambda b, s: s.info["command"],
    }
    assert contract.frames == 1
    sizes = {c.name: c.size for c in contract.obs}
    assert sizes["qpos_error_history"] == 12 * base._config.history_len
    assert_slices_hold_the_env_signals(base, contract, readers)


def test_barkour_slices_hold_their_signals(compose_cfg: Compose) -> None:
    base, contract = built(compose_cfg, "barkour", *SMALL_ARENA)
    scale = jnp.array([2.0, 2.0, 0.25])
    readers: dict[str, Reader] = {
        "gyro_z": lambda b, s: 0.25 * b._get_localrpyrate(s.data)[-1],
        "gravity": lambda b, s: b._get_gravity(s.data),
        "command": lambda b, s: s.info["command"] * scale,
        "joint_pos": lambda b, s: s.data.qpos[7:] - b._default_pose,
        "last_act": lambda b, s: s.info["last_act"],
    }
    assert contract.frames == 15
    assert contract.frame_dim == 31
    units = {c.name: c.units for c in contract.obs}
    assert units["gyro_z"] == "0.25 rad/s"
    assert units["command"] == "2 m/s, 2 m/s, 0.25 rad/s"
    assert_slices_hold_the_env_signals(base, contract, readers)


@pytest.mark.slow
@pytest.mark.skipif(
    not paths.terrain_paths("accel")["spec"].exists(),
    reason="accel arena not built; run `just build-accel-arena`",
)
def test_wojtek_slices_hold_the_env_catalog(compose_cfg: Compose) -> None:
    base, contract = built(compose_cfg, "wojtek")
    state = after_steps(base)
    # the mirror augmentation permutes the observation in training only
    info = {**state.info, "mirror": jnp.zeros_like(state.info["mirror"])}
    obs = np.asarray(base._get_obs(state.data, info)["state"])
    catalog = base._obs_catalog(state.data, info)
    assert [c.name for c in contract.obs] == base.actor_obs_names
    for component, start, stop in index_ranges(contract):
        np.testing.assert_allclose(
            obs[start:stop],
            np.asarray(catalog[component.name]),
            atol=1e-5,
            err_msg=component.name,
        )
    assert contract.frame_dim == obs.size


# --- the probe itself, on a stand-in env ------------------------------------


@dataclasses.dataclass(frozen=True)
class FakeData:
    qpos: jax.Array
    qvel: jax.Array

    def replace(self, **kwargs: Any) -> "FakeData":
        return dataclasses.replace(self, **kwargs)


@dataclasses.dataclass(frozen=True)
class FakeState:
    data: FakeData
    info: dict[str, jax.Array]
    obs: dict[str, jax.Array]

    def replace(self, **kwargs: Any) -> "FakeState":
        return dataclasses.replace(self, **kwargs)


class FakeRobot:
    """Observation [joint_pos (2), command (3)], or the swapped order."""

    def __init__(self, swapped: bool = False, extra: int = 0) -> None:
        self.swapped, self.extra = swapped, extra

    def reset(self, rng: jax.Array) -> FakeState:
        del rng
        data = FakeData(qpos=jnp.arange(9.0), qvel=jnp.zeros(8))
        info = {"command": jnp.array([0.5, 0.0, 0.1])}
        return FakeState(data, info, {"state": jnp.zeros(5 + self.extra)})

    def get_gyro(self, data: FakeData) -> jax.Array:
        return data.qpos[:3]

    def actor_obs(self, state: FakeState) -> jax.Array:
        joint_pos = state.data.qpos[7:]
        command = state.info["command"]
        parts = [command, joint_pos] if self.swapped else [joint_pos, command]
        return jnp.concatenate([*parts, jnp.zeros(self.extra)])


CHANNELS = (
    Channel("joint_pos", ("rad",), "angles", Signal(qpos=True)),
    Channel(
        "command",
        ("m/s", "m/s", "rad/s"),
        "setpoint",
        Signal(info=("command",)),
    ),
)


def test_measure_places_channels_where_the_env_puts_them() -> None:
    layout = measure(FakeRobot(), CHANNELS)
    placed = [(p.channel.name, p.start, p.stop) for p in layout.placements]
    assert placed == [("joint_pos", 0, 2), ("command", 2, 5)]
    assert layout.frames == 1


def test_measure_follows_a_swap_of_components() -> None:
    layout = measure(FakeRobot(swapped=True), CHANNELS)
    placed = [(p.channel.name, p.start, p.stop) for p in layout.placements]
    assert placed == [("command", 0, 3), ("joint_pos", 3, 5)]


def test_measure_refuses_entries_no_channel_explains() -> None:
    with pytest.raises(ValueError, match="carry older frames"):
        measure(FakeRobot(extra=2), CHANNELS)


def test_measure_refuses_a_channel_that_moves_nothing() -> None:
    silent = Channel("gyro", ("rad/s",), "gyro", Signal(getters=("get_gyro",)))
    with pytest.raises(ValueError, match="moves no actor observation"):
        measure(FakeRobot(), (*CHANNELS, silent))


def test_measure_names_a_missing_getter() -> None:
    missing = Channel("gyro", ("rad/s",), "gyro", Signal(getters=("nope",)))
    with pytest.raises(AttributeError, match="no getter 'nope'"):
        measure(FakeRobot(), (*CHANNELS, missing))


def test_measure_refuses_overlapping_channels() -> None:
    twice = Channel("again", ("rad",), "angles", Signal(qpos=True))
    with pytest.raises(ValueError, match="belong to no channel or to two"):
        measure(FakeRobot(), (*CHANNELS, twice))


@pytest.mark.parametrize(
    ("units", "gains", "expected"),
    [
        (("rad",), (1.0, 1.0), "rad"),
        (("rad/s",), (0.25,), "0.25 rad/s"),
        (("m/s", "m/s", "rad/s"), (1.0, 1.0, 1.0), "m/s, m/s, rad/s"),
        (("m/s", "m/s", "rad/s"), (2.0, 2.0, 0.25), "2 m/s, 2 m/s, 0.25 rad/s"),
    ],
)
def test_scaled_units_carry_the_measured_gain(
    units: tuple[str, ...], gains: tuple[float, ...], expected: str
) -> None:
    assert scaled_units(units, gains) == expected


def test_scaled_units_refuse_a_mismatched_count() -> None:
    with pytest.raises(ValueError, match="name 2 values for 3 entries"):
        scaled_units(("m/s", "rad/s"), (1.0, 1.0, 1.0))
