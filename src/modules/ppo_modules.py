from collections.abc import Callable
from typing import NamedTuple

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp


class Obs(NamedTuple):
    actor: jnp.ndarray
    critic: jnp.ndarray


class Episode(NamedTuple):
    obs: Obs
    next_obs: Obs
    action: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    entropy: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    metrics: dict[str, jnp.ndarray]


class A2C(eqx.Module):
    activation_func: Callable = eqx.field(static=True)
    actor_mean: eqx.nn.Sequential
    actor_log_std: jnp.ndarray
    critic: eqx.nn.Sequential
    action_high: tuple = eqx.field(static=True)

    def __init__(
        self,
        *,
        dense_dim: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        action_high: float,
        key: jax.Array,
        activation: str = "relu",
    ):
        self.action_high = tuple(float(x) for x in (action_high,))

        ka1, ka2, ka3, kc1, kc2, kc3 = jax.random.split(key, 6)

        if activation == "relu":
            self.activation_func = jax.nn.relu
        else:
            self.activation_func = jax.nn.tanh

        self.actor_mean = eqx.nn.Sequential(
            [
                eqx.nn.Linear(actor_obs_dim, dense_dim, key=ka1),
                eqx.nn.LayerNorm(dense_dim),
                eqx.nn.Lambda(self.activation_func),
                eqx.nn.Linear(dense_dim, dense_dim, key=ka2),
                eqx.nn.LayerNorm(dense_dim),
                eqx.nn.Lambda(self.activation_func),
                eqx.nn.Linear(dense_dim, action_dim, key=ka3),
            ]
        )
        self.actor_log_std = jnp.zeros((action_dim,))

        self.critic = eqx.nn.Sequential(
            [
                eqx.nn.Linear(critic_obs_dim, dense_dim, key=kc1),
                eqx.nn.LayerNorm(dense_dim),
                eqx.nn.Lambda(self.activation_func),
                eqx.nn.Linear(dense_dim, dense_dim, key=kc2),
                eqx.nn.LayerNorm(dense_dim),
                eqx.nn.Lambda(self.activation_func),
                eqx.nn.Linear(dense_dim, 1, key=kc3),
            ]
        )

    def __call__(
        self,
        obs: Obs,
        key: jax.Array,
        action: jax.Array | None = None,
    ):
        actor_key, critic_key, action_key = jax.random.split(key, 3)
        actor_mean = self.actor_mean(obs.actor, key=actor_key)
        base = distrax.MultivariateNormalDiag(
            actor_mean,
            jnp.exp(self.actor_log_std),
        )
        bijector = distrax.Block(
            distrax.Chain(
                [
                    distrax.ScalarAffine(
                        shift=0.0,
                        scale=jnp.array(self.action_high),
                    ),
                    distrax.Tanh(),
                ]
            ),
            ndims=1,
        )
        pi = distrax.Transformed(distribution=base, bijector=bijector)

        critic_value = self.critic(obs.critic, key=critic_key)
        if action is None:
            action = pi.sample(seed=action_key)
        log_prob = pi.log_prob(action)
        entropy = -1 * log_prob
        return action, log_prob, critic_value, entropy

    def deterministic_action(self, obs: Obs) -> jax.Array:
        actor_mean = self.actor_mean(obs.actor)
        return jnp.array(self.action_high) * jnp.tanh(actor_mean)
