import math
from collections.abc import Callable
from typing import Any

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp


class LevelGenerator(eqx.Module):
    """PAIRED's adversary policy: a diagonal Gaussian over the level genome
    (`shape`), squashed into [0, 1] by a sigmoid."""

    activation_func: Callable = eqx.field(static=True)
    trunk: eqx.nn.Sequential
    log_std: jnp.ndarray
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        latent_dim: int,
        dense_dim: int,
        shape: tuple[int, ...],
        key: jax.Array,
        activation: str = "tanh",
    ):
        self.shape = tuple(shape)
        out_dim = math.prod(self.shape)

        k1, k2, k3 = jax.random.split(key, 3)
        relu = activation == "relu"
        self.activation_func = jax.nn.relu if relu else jax.nn.tanh

        self.trunk = eqx.nn.Sequential(
            [
                eqx.nn.Linear(latent_dim, dense_dim, key=k1),
                eqx.nn.LayerNorm(dense_dim),
                eqx.nn.Lambda(self.activation_func),
                eqx.nn.Linear(dense_dim, dense_dim, key=k2),
                eqx.nn.LayerNorm(dense_dim),
                eqx.nn.Lambda(self.activation_func),
                eqx.nn.Linear(dense_dim, out_dim, key=k3),
            ]
        )
        self.log_std = jnp.zeros((out_dim,))

    def distribution(self, z: jax.Array) -> distrax.Transformed:
        mean = self.trunk(z)
        base = distrax.MultivariateNormalDiag(mean, jnp.exp(self.log_std))
        bijector = distrax.Block(distrax.Sigmoid(), ndims=1)
        return distrax.Transformed(distribution=base, bijector=bijector)

    def __call__(
        self, z: jax.Array, key: jax.Array
    ) -> tuple[Any, Any, Any]:
        """A sampled level: (theta, log_prob, entropy estimate)."""
        pi = self.distribution(z)
        flat = pi.sample(seed=key)
        log_prob = pi.log_prob(flat)
        theta = flat.reshape(self.shape)
        return theta, log_prob, -1 * log_prob

    def evaluate(self, z: jax.Array, theta: jax.Array) -> tuple[Any, Any]:
        """(log_prob, entropy estimate) of a sampled `theta` now."""
        pi = self.distribution(z)
        log_prob = pi.log_prob(theta.reshape(-1))
        return log_prob, -1 * log_prob
