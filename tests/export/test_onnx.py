from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import pytest
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks

from config.schema import Activation, NetworkConfig
from export.contract import (
    Command,
    Contract,
    HeightStance,
    Joint,
    ObsComponent,
)
from export.onnx_graph import OPSET, build_model
from export.verify import TOLERANCE, session
from training.ppo import make_network

OBS_DIM = 10
ACTIONS = 6
HIDDEN = [16, 8]


@dataclass
class FakeObsSize:
    actor: int = OBS_DIM
    critic: int = OBS_DIM + 4


@dataclass
class FakeEnv:
    observation_size: FakeObsSize
    action_size: int = ACTIONS


def network_config(activation: Activation) -> NetworkConfig:
    return NetworkConfig(
        policy_hidden=HIDDEN, value_hidden=HIDDEN, activation=activation
    )


def make_params(
    config: NetworkConfig, obs_dim: int = OBS_DIM
) -> tuple[Any, Callable[[jax.Array], jax.Array]]:
    """Random policy weights, a normalizer fitted to non-unit data, and the
    JAX deterministic policy."""
    env = FakeEnv(FakeObsSize(actor=obs_dim, critic=obs_dim + 4))
    network = make_network(env, config)  # ty: ignore[invalid-argument-type]
    policy_params = network.policy_network.init(jax.random.key(0))
    value_params = network.value_network.init(jax.random.key(1))
    float32 = jnp.dtype("float32")
    normalizer = running_statistics.init_state(
        {
            "state": specs.Array((obs_dim,), float32),
            "privileged_state": specs.Array((obs_dim + 4,), float32),
        }
    )
    rng = np.random.default_rng(0)
    batch = {
        "state": jnp.asarray(
            3.0 * rng.standard_normal((512, obs_dim)) + 1.5, jnp.float32
        ),
        "privileged_state": jnp.zeros((512, obs_dim + 4), jnp.float32),
    }
    normalizer = running_statistics.update(normalizer, batch)
    params = (normalizer, policy_params, value_params)
    make_policy = ppo_networks.make_inference_fn(network)
    deterministic = make_policy(params, deterministic=True)

    def policy(obs: jax.Array) -> jax.Array:
        return deterministic({"state": obs}, jax.random.key(0))[0]

    return params, policy


def contract(
    obs_dim: int = OBS_DIM, stance: HeightStance | None = None
) -> Contract:
    lo = np.linspace(-1.0, -0.5, ACTIONS)
    hi = np.linspace(0.5, 1.0, ACTIONS)
    joints = tuple(
        Joint(
            name=f"j{i}",
            default=0.1 * i,
            target_min=float(lo[i]),
            target_max=float(hi[i]),
            kp=20.0,
            kd=1.0,
            torque_limit=None,
            action_scale=0.4 + 0.1 * i,
        )
        for i in range(ACTIONS)
    )
    return Contract(
        robot="synthetic",
        obs=(ObsComponent("x", obs_dim, "-", "-"),),
        frames=1,
        joints=joints,
        ctrl_dt=0.02,
        command=Command(("vx",), ("m/s",), (0.0,), (1.0,), ()),
        conditions=(),
        stance=stance,
    )


def stance() -> HeightStance:
    home = tuple(0.05 * i for i in range(ACTIONS))
    return HeightStance(
        height_index=3,
        knots_height=(0.08, 0.1, 0.13, 0.17),
        knots_dsecond=(-0.4, -0.1, 0.2, 0.5),
        offsets=(0.0, 1.0, 2.0) * (ACTIONS // 3),
        home=home,
        stance_min=(-0.3,) * ACTIONS,
        stance_max=(0.6,) * ACTIONS,
    )


def reference_targets(
    contract: Contract, obs: jax.Array, action: jax.Array
) -> jax.Array:
    joints = contract.joints
    scale = jnp.asarray([j.action_scale for j in joints])
    lo = jnp.asarray([j.target_min for j in joints])
    hi = jnp.asarray([j.target_max for j in joints])
    if contract.stance is None:
        origin = jnp.asarray([j.default for j in joints])
    else:
        s = contract.stance
        dsecond = jnp.interp(
            obs[:, s.height_index],
            jnp.asarray(s.knots_height),
            jnp.asarray(s.knots_dsecond),
        )
        offset = dsecond[:, None] * jnp.asarray(s.offsets)
        origin = jnp.clip(
            jnp.asarray(s.home) + offset,
            jnp.asarray(s.stance_min),
            jnp.asarray(s.stance_max),
        )
    return jnp.clip(origin + action * scale, lo, hi)


def observations(params: Any) -> np.ndarray:
    rng = np.random.default_rng(1)
    mean = np.asarray(params[0].mean["state"])
    std = np.asarray(params[0].std["state"])
    obs = mean + 2.0 * std * rng.standard_normal((300, OBS_DIM))
    # command height from below the first knot to above the last
    obs[:, 3] = np.linspace(0.05, 0.22, len(obs))
    return obs.astype(np.float32)


def check(
    model: onnx.ModelProto, params: Any, policy: Any, c: Contract
) -> None:
    obs = observations(params)
    action, targets = session(model).run(None, {"obs": obs})
    expected = policy(jnp.asarray(obs))
    expected_targets = reference_targets(c, jnp.asarray(obs), expected)
    assert np.abs(action - np.asarray(expected)).max() <= TOLERANCE
    assert np.abs(targets - np.asarray(expected_targets)).max() <= TOLERANCE


@pytest.mark.parametrize("kind", list(Activation))
def test_affine_targets_match_jax(kind: Activation) -> None:
    config = network_config(kind)
    params, policy = make_params(config)
    c = contract()
    check(build_model(params, config, c), params, policy, c)


def test_height_stance_targets_match_jax() -> None:
    config = network_config(Activation.elu)
    params, policy = make_params(config)
    c = contract(stance=stance())
    check(build_model(params, config, c), params, policy, c)


def test_the_stance_follows_the_command_height() -> None:
    config = network_config(Activation.elu)
    params, _ = make_params(config)
    model = build_model(params, config, contract(stance=stance()))
    obs = np.zeros((2, OBS_DIM), np.float32)
    obs[:, 3] = [0.09, 0.16]
    _, targets = session(model).run(None, {"obs": obs})
    targets = np.asarray(targets)
    assert np.abs(targets[0] - targets[1]).max() > 0.05


def test_model_declares_inputs_outputs_and_pinned_opset() -> None:
    config = network_config(Activation.elu)
    params, _ = make_params(config)
    model = build_model(params, config, contract())
    assert [i.name for i in model.graph.input] == ["obs"]
    assert [o.name for o in model.graph.output] == ["action", "joint_targets"]
    assert [(o.domain, o.version) for o in model.opset_import] == [("", OPSET)]
    shape = model.graph.input[0].type.tensor_type.shape
    assert shape.dim[0].dim_param == "batch"
    assert shape.dim[1].dim_value == OBS_DIM


def test_refuses_a_checkpoint_with_another_obs_size() -> None:
    config = network_config(Activation.elu)
    params, _ = make_params(config, obs_dim=OBS_DIM + 2)
    with pytest.raises(ValueError, match=r"takes 12 actor observations.*10"):
        build_model(params, config, contract())


def test_refuses_a_checkpoint_with_another_network() -> None:
    params, _ = make_params(network_config(Activation.elu))
    other = NetworkConfig(
        policy_hidden=[16], value_hidden=[16], activation=Activation.elu
    )
    with pytest.raises(ValueError, match="policy_hidden"):
        build_model(params, other, contract())
