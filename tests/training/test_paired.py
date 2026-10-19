from typing import Any

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import networks as ppo_networks

from config.schema import BC, ModeConfig
from training.paired import bc_coefs, gaussian_kl, paired_regret, policy_kl

OBS, ACTIONS = 5, 3


def make_network() -> ppo_networks.PPONetworks:
    return ppo_networks.make_ppo_networks(
        {"state": (OBS,), "privileged_state": (OBS + 2,)},
        ACTIONS,
        preprocess_observations_fn=running_statistics.normalize,  # ty: ignore[invalid-argument-type]
        policy_hidden_layer_sizes=(16,),
        value_hidden_layer_sizes=(16,),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )


def observations(seed: int, n: int = 8) -> dict[str, jax.Array]:
    k1, k2 = jax.random.split(jax.random.key(seed))
    return {
        "state": jax.random.normal(k1, (n, OBS)),
        "privileged_state": jax.random.normal(k2, (n, OBS + 2)),
    }


def normalizer(obs: dict[str, jax.Array] | None = None) -> Any:
    state = running_statistics.init_state(
        {
            "state": specs.Array((OBS,), jnp.dtype("float32")),
            "privileged_state": specs.Array((OBS + 2,), jnp.dtype("float32")),
        }
    )
    return state if obs is None else running_statistics.update(state, obs)


def base_gaussian(logits: jax.Array) -> distrax.MultivariateNormalDiag:
    # brax NormalTanhDistribution: scale = softplus(raw) + min_std 0.001
    loc, raw = jnp.split(logits, 2, axis=-1)
    return distrax.MultivariateNormalDiag(loc, jax.nn.softplus(raw) + 0.001)


def test_gaussian_kl_matches_distrax() -> None:
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    p_loc, q_loc = jax.random.normal(k1, (4, 6)), jax.random.normal(k2, (4, 6))
    p_scale = jnp.exp(jax.random.normal(k3, (4, 6)))
    q_scale = jnp.exp(jax.random.normal(k4, (4, 6)))
    expected = distrax.MultivariateNormalDiag(p_loc, p_scale).kl_divergence(
        distrax.MultivariateNormalDiag(q_loc, q_scale)
    )
    np.testing.assert_allclose(
        gaussian_kl(p_loc, p_scale, q_loc, q_scale), expected, rtol=1e-5
    )


def test_policy_kl_is_zero_for_identical_policies() -> None:
    network = make_network()
    params = network.policy_network.init(jax.random.key(1))
    obs = observations(2)
    norm = normalizer(obs)
    kl = policy_kl(network, norm, params, (norm, params), obs)
    np.testing.assert_allclose(kl, 0.0, atol=1e-6)


def test_policy_kl_matches_closed_form_with_each_normalizer() -> None:
    network = make_network()
    student = network.policy_network.init(jax.random.key(1))
    teacher = network.policy_network.init(jax.random.key(2))
    obs = observations(3)
    student_norm = normalizer(observations(4, 64))
    teacher_norm = normalizer(observations(5, 64))
    kl = policy_kl(network, student_norm, student, (teacher_norm, teacher), obs)
    apply = network.policy_network.apply
    expected = base_gaussian(apply(teacher_norm, teacher, obs)).kl_divergence(
        base_gaussian(apply(student_norm, student, obs))
    )
    assert kl.shape == (8,)
    assert bool(jnp.all(kl > 0.0))
    np.testing.assert_allclose(kl, expected, rtol=1e-4)


def test_policy_kl_gradient_reaches_only_the_student() -> None:
    network = make_network()
    student = network.policy_network.init(jax.random.key(1))
    teacher = network.policy_network.init(jax.random.key(2))
    obs = observations(3)
    norm = normalizer(obs)

    def mean_kl(student: Any, teacher: Any) -> jax.Array:
        return policy_kl(network, norm, student, (norm, teacher), obs).mean()

    student_grad, teacher_grad = jax.grad(mean_kl, argnums=(0, 1))(student, teacher)
    student_norm = sum(jnp.abs(g).sum() for g in jax.tree.leaves(student_grad))
    teacher_norm = sum(jnp.abs(g).sum() for g in jax.tree.leaves(teacher_grad))
    assert float(student_norm) > 0.0
    assert float(teacher_norm) == 0.0


def test_paired_regret_per_level() -> None:
    prot = np.array([[1.0, 3.0], [5.0, 5.0], [0.0, 2.0]])
    antag = np.array([[4.0, 0.0], [2.0, 3.0], [1.0, 1.0]])
    # antag max - prot mean: 4 - 2, 3 - 5 (clipped), 1 - 1
    np.testing.assert_array_equal(paired_regret(prot, antag), [2.0, 0.0, 0.0])


def test_paired_regret_uses_the_antagonist_best_episode() -> None:
    prot = np.zeros((1, 4))
    antag = np.array([[-5.0, -5.0, -5.0, 0.5]])
    assert paired_regret(prot, antag) == pytest.approx([0.5])


def mode(bc: BC, bc_every: int = 1) -> ModeConfig:
    return ModeConfig(bc=bc, bc_coef=0.5, bc_every=bc_every)


@pytest.mark.parametrize(
    ("bc", "it", "expected"),
    [
        (BC.none, 0, (None, None)),
        (BC.uni, 0, (0.5, None)),
        (BC.bi, 0, (0.5, 0.5)),
        (BC.bi, 3, (0.5, 0.5)),
    ],
)
def test_bc_coefs_per_variant(bc: BC, it: int, expected: tuple) -> None:
    assert bc_coefs(mode(bc), it) == expected


def test_bc_coefs_zero_between_distillation_iterations() -> None:
    assert bc_coefs(mode(BC.bi, bc_every=3), 4) == (0.0, 0.0)
    assert bc_coefs(mode(BC.uni, bc_every=3), 6) == (0.5, None)
