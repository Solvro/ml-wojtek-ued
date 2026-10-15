import itertools
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

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


ACTIVATIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "relu": jax.nn.relu,
    "tanh": jax.nn.tanh,
    "elu": jax.nn.elu,
}


def mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: Callable[[jax.Array], jax.Array],
    key: jax.Array,
) -> eqx.nn.Sequential:
    dims = [in_dim, *hidden_dims, out_dim]
    keys = jax.random.split(key, len(dims) - 1)
    layers: list[Callable[..., Any]] = []
    for i, (d_in, d_out) in enumerate(itertools.pairwise(dims)):
        layers.append(eqx.nn.Linear(d_in, d_out, key=keys[i]))
        if i < len(dims) - 2:
            layers.append(eqx.nn.Lambda(activation))
    return eqx.nn.Sequential(layers)


class RunningNorm(eqx.Module):
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    def __init__(self, dim: int):
        self.mean = jnp.zeros((dim,))
        self.var = jnp.ones((dim,))
        self.count = jnp.array(1e-4)

    def __call__(self, x: jax.Array) -> jax.Array:
        # statistics are fitted to rollouts, never trained by the PPO loss
        mean = jax.lax.stop_gradient(self.mean)
        var = jax.lax.stop_gradient(self.var)
        return jnp.clip((x - mean) / jnp.sqrt(var + 1e-8), -5.0, 5.0)

    def update(self, batch: jax.Array) -> "RunningNorm":
        batch = batch.reshape(-1, batch.shape[-1])
        n = batch.shape[0]
        batch_mean = batch.mean(axis=0)
        total = self.count + n
        delta = batch_mean - self.mean
        m2 = (
            self.var * self.count
            + batch.var(axis=0) * n
            + delta**2 * self.count * n / total
        )
        return eqx.tree_at(
            lambda r: (r.mean, r.var, r.count),
            self,
            (self.mean + delta * n / total, m2 / total, total),
        )


class A2C(eqx.Module):
    actor_norm: RunningNorm
    critic_norm: RunningNorm
    actor_mean: eqx.nn.Sequential
    actor_log_std: jnp.ndarray
    critic: eqx.nn.Sequential
    action_high: tuple = eqx.field(static=True)

    def __init__(
        self,
        *,
        hidden_dims: Sequence[int],
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        action_high: float,
        key: jax.Array,
        activation: str = "elu",
    ):
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r}, have {sorted(ACTIVATIONS)}"
            )
        self.action_high = tuple(float(x) for x in (action_high,))
        actor_key, critic_key = jax.random.split(key)
        act = ACTIVATIONS[activation]
        self.actor_norm = RunningNorm(actor_obs_dim)
        self.critic_norm = RunningNorm(critic_obs_dim)
        self.actor_mean = mlp(actor_obs_dim, hidden_dims, action_dim, act, actor_key)
        self.actor_log_std = jnp.zeros((action_dim,))
        self.critic = mlp(critic_obs_dim, hidden_dims, 1, act, critic_key)

    def update_obs_norm(self, obs: Obs) -> "A2C":
        return eqx.tree_at(
            lambda m: (m.actor_norm, m.critic_norm),
            self,
            (self.actor_norm.update(obs.actor), self.critic_norm.update(obs.critic)),
        )

    def value(self, critic_obs: jax.Array) -> jax.Array:
        return self.critic(self.critic_norm(critic_obs))

    def __call__(
        self,
        obs: Obs,
        key: jax.Array,
        action: jax.Array | None = None,
    ):
        actor_key, critic_key, action_key = jax.random.split(key, 3)
        actor_mean = self.actor_mean(self.actor_norm(obs.actor), key=actor_key)
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

        critic_value = self.critic(self.critic_norm(obs.critic), key=critic_key)
        if action is None:
            action = pi.sample(seed=action_key)
        log_prob = pi.log_prob(action)
        entropy = -1 * log_prob
        return action, log_prob, critic_value, entropy

    def deterministic_action(self, obs: Obs) -> jax.Array:
        actor_mean = self.actor_mean(self.actor_norm(obs.actor))
        return jnp.array(self.action_high) * jnp.tanh(actor_mean)
