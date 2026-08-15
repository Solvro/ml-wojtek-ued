import jax
import jax.numpy as jnp


def positive_value_loss(advantages: jax.Array) -> jax.Array:
    return jnp.mean(jnp.maximum(advantages, 0.0))


def positive_value_loss_per_lane(advantages: jax.Array) -> jax.Array:
    return jnp.mean(jnp.maximum(advantages, 0.0), axis=1)
