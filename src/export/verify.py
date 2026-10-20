"""Export-time check: onnxruntime against the JAX policy and the env's own
action-to-target mapping."""

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime

from export.contract import Contract
from export.onnx_graph import INPUT, OUTPUTS

TOLERANCE = 1e-5
RANDOM_BATCH = 256

Targets = Callable[[jax.Array, jax.Array], jax.Array]


@dataclasses.dataclass(frozen=True)
class Verification:
    random_samples: int
    random_error: float
    env_samples: int
    env_error: float
    tolerance: float = TOLERANCE


def session(model: onnx.ModelProto) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def random_observations(
    params: Any, contract: Contract, seed: int = 0
) -> np.ndarray:
    """Standard-normal draws around the normalizer's mean / std, plus a
    unit-scale block, so both the typical and the extreme inputs run."""
    rng = np.random.default_rng(seed)
    mean = np.asarray(params[0].mean["state"], np.float32)
    std = np.asarray(params[0].std["state"], np.float32)
    shape = (RANDOM_BATCH // 2, contract.obs_dim)
    around = mean + std * rng.standard_normal(shape).astype(np.float32)
    unit = rng.uniform(-1.0, 1.0, shape).astype(np.float32)
    return np.concatenate([around, unit])


def max_error(
    sess: onnxruntime.InferenceSession,
    obs: np.ndarray,
    policy: Callable[[jax.Array], jax.Array],
    targets: Targets,
) -> float:
    """Largest |onnx - jax| over both outputs."""
    action, joint_targets = sess.run(list(OUTPUTS), {INPUT: obs})
    with jax.default_matmul_precision("highest"):
        expected_action = policy(jnp.asarray(obs))
        expected_targets = targets(jnp.asarray(obs), expected_action)
    return max(
        float(np.max(np.abs(action - np.asarray(expected_action)))),
        float(np.max(np.abs(joint_targets - np.asarray(expected_targets)))),
    )


def verify(
    model: onnx.ModelProto,
    params: Any,
    contract: Contract,
    policy: Callable[[jax.Array], jax.Array],
    targets: Targets,
    env_obs: np.ndarray,
) -> Verification:
    sess = session(model)
    random = random_observations(params, contract)
    result = Verification(
        random_samples=len(random),
        random_error=max_error(sess, random, policy, targets),
        env_samples=len(env_obs),
        env_error=max_error(sess, env_obs, policy, targets),
    )
    worst = max(result.random_error, result.env_error)
    if not worst <= TOLERANCE:
        raise ValueError(
            f"onnxruntime disagrees with the JAX policy: max abs error "
            f"{worst:.2e} > {TOLERANCE:.0e} (random {result.random_error:.2e}, "
            f"env observations {result.env_error:.2e}); nothing written"
        )
    return result
