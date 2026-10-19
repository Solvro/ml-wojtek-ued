from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from config.schema import ModeConfig
from training.generator import LevelGenerator

ADVERSARY_EPOCHS = 5


class AdversaryBatch(NamedTuple):
    z: jax.Array
    theta: jax.Array
    log_prob: jax.Array
    advantages: jax.Array


class Adversary:
    """PAIRED's learned level designer, trained by single-step PPO on regret."""

    def __init__(
        self, mode: ModeConfig, shape: tuple[int, ...], key: jax.Array
    ) -> None:
        self.latent_dim = mode.latent_dim
        self.generator = LevelGenerator(
            latent_dim=mode.latent_dim, dense_dim=128, shape=shape, key=key
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.5), optax.adam(mode.gen_learning_rate)
        )
        params = eqx.filter(self.generator, eqx.is_array)
        self.opt_state = self.optimizer.init(params)
        self.entropy_coef = mode.gen_entropy_coef
        self.sample = eqx.filter_jit(lambda g, z, keys: jax.vmap(g)(z, keys)[:2])
        self.step = eqx.filter_jit(self._step)

    def _loss(self, generator: LevelGenerator, batch: AdversaryBatch) -> jax.Array:
        new_log_prob, entropy = jax.vmap(generator.evaluate)(batch.z, batch.theta)
        log_ratio = jnp.clip(new_log_prob - batch.log_prob, -20.0, 20.0)
        ratio = jnp.exp(log_ratio)
        adv = batch.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        clipped = jnp.clip(ratio, 0.8, 1.2) * adv
        policy_loss = -jnp.mean(jnp.minimum(ratio * adv, clipped))
        return policy_loss - self.entropy_coef * jnp.mean(entropy)

    def _step(
        self,
        generator: LevelGenerator,
        opt_state: optax.OptState,
        batch: AdversaryBatch,
    ) -> tuple[LevelGenerator, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(self._loss)(generator, batch)
        updates, opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(generator, eqx.is_array)
        )
        generator = eqx.apply_updates(generator, updates)
        log_std = jnp.clip(generator.log_std, -5.0, 2.0)
        generator = eqx.tree_at(lambda g: g.log_std, generator, log_std)
        return generator, opt_state, loss

    def update(
        self, z: jax.Array, theta: jax.Array, log_prob: Any, regret: jax.Array
    ) -> float:
        batch = AdversaryBatch(z, theta, log_prob, regret - regret.mean())
        loss = jnp.zeros(())
        for _ in range(ADVERSARY_EPOCHS):
            self.generator, self.opt_state, loss = self.step(
                self.generator, self.opt_state, batch
            )
        return float(loss)
