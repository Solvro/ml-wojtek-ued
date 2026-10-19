from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks

from config.schema import BC, ModeConfig

# a BC teacher as one student's (running obs normalizer, policy params)
Teacher = tuple[Any, Any]


def paired_regret(
    prot_returns: np.ndarray, antag_returns: np.ndarray
) -> np.ndarray:
    """Per level (rows: levels, columns: episodes): the antagonist's best
    return minus the protagonist's mean return, clipped at 0."""
    return np.maximum(
        antag_returns.max(axis=1) - prot_returns.mean(axis=1), 0.0
    )


def gaussian_kl(
    p_loc: jax.Array, p_scale: jax.Array, q_loc: jax.Array, q_scale: jax.Array
) -> jax.Array:
    """KL(p || q) of diagonal Gaussians, summed over the last axis."""
    var_ratio = jnp.square(p_scale / q_scale)
    mahalanobis = jnp.square((p_loc - q_loc) / q_scale)
    return 0.5 * jnp.sum(
        var_ratio + mahalanobis - 1.0 - jnp.log(var_ratio), axis=-1
    )


def policy_kl(
    network: ppo_networks.PPONetworks,
    normalizer: Any,
    policy_params: Any,
    teacher: Teacher,
    observation: Any,
) -> jax.Array:
    """KL(pi_teacher || pi_student) per observation, each policy through its
    own normalizer, no gradient into the teacher. KL is invariant under the
    shared tanh squash, so the pre-tanh Gaussian KL is exact."""
    apply = network.policy_network.apply
    dist = network.parametric_action_distribution
    student = dist.create_dist(apply(normalizer, policy_params, observation))
    teacher_normalizer, teacher_policy = jax.lax.stop_gradient(teacher)
    ref = dist.create_dist(
        apply(teacher_normalizer, teacher_policy, observation)
    )
    return gaussian_kl(ref.loc, ref.scale, student.loc, student.scale)


def bc_coefs(mode: ModeConfig, it: int) -> tuple[float | None, float | None]:
    """(protagonist, antagonist) BC weight this iteration, None for a
    student that never distils: UniBC distils the protagonist from the
    antagonist, BiBC also the other way round."""
    weight = 0.0 if it % mode.bc_every else mode.bc_coef
    prot = None if mode.bc == BC.none else weight
    return prot, weight if mode.bc == BC.bi else None
