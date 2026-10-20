from typing import Any

import jax
import jax.numpy as jnp
import optax
from brax.training import types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks

from config.schema import NetworkConfig, PPOConfig
from env.functional import FunctionalEnv
from training.paired import Teacher, policy_kl
from training.ppo import Normalizer, make_network

Params = ppo_losses.PPONetworkParams


class Student:
    """brax PPO networks, optimizer and jitted update / regret / BC KL."""

    def __init__(
        self,
        env: FunctionalEnv,
        key: jax.Array,
        entropy_coef: float,
        ppo: PPOConfig,
        network: NetworkConfig,
    ) -> None:
        obs_size = env.observation_size
        self.entropy_coef = entropy_coef
        self.ppo = ppo
        self.network = make_network(env, network)
        self.make_policy = ppo_networks.make_inference_fn(self.network)
        policy_key, value_key = jax.random.split(key)
        self.params = Params(
            policy=self.network.policy_network.init(policy_key),
            value=self.network.value_network.init(value_key),
        )
        float32 = jnp.dtype("float32")
        self.normalizer = running_statistics.init_state(
            {
                "state": specs.Array((obs_size.actor,), float32),
                "privileged_state": specs.Array((obs_size.critic,), float32),
            }
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(ppo.max_grad_norm),
            optax.adam(ppo.learning_rate),
        )
        self.opt_state = self.optimizer.init(self.params)
        self.update = jax.jit(self._update)
        self.regret = jax.jit(self._regret)
        self.bc_kl = jax.jit(self._bc_kl)

    def inference_params(self) -> tuple[Any, Any, Any]:
        return (self.normalizer, self.params.policy, self.params.value)

    def as_teacher(self) -> Teacher:
        return (self.normalizer, self.params.policy)

    def train_step(
        self,
        data: types.Transition,
        key: jax.Array,
        teacher: Teacher | None,
        bc_coef: float | None,
    ) -> None:
        self.params, self.opt_state, self.normalizer, _ = self.update(
            self.params,
            self.opt_state,
            self.normalizer,
            data,
            key,
            teacher=teacher,
            bc_coef=jnp.float32(bc_coef or 0.0),
        )

    def _bc_kl(
        self,
        policy_params: Any,
        normalizer: Normalizer,
        teacher: Teacher,
        observation: Any,
    ) -> jax.Array:
        kl = policy_kl(
            self.network, normalizer, policy_params, teacher, observation
        )
        return kl.mean()

    def _loss(
        self,
        params: Params,
        normalizer: Normalizer,
        data: types.Transition,
        key: jax.Array,
        *,
        teacher: Teacher | None,
        bc_coef: jax.Array,
    ) -> tuple[jax.Array, types.Metrics]:
        loss, metrics = ppo_losses.compute_ppo_loss(
            params,
            normalizer,
            data,
            key,
            self.network,
            entropy_cost=self.entropy_coef,
            discounting=self.ppo.discounting,
            gae_lambda=self.ppo.gae_lambda,
            clipping_epsilon=self.ppo.clipping_epsilon,
        )
        if teacher is None:
            return loss, metrics
        kl = self._bc_kl(params.policy, normalizer, teacher, data.observation)
        return loss + bc_coef * kl, metrics

    def _update(
        self,
        params: Params,
        opt_state: optax.OptState,
        normalizer: Normalizer,
        data: types.Transition,
        key: jax.Array,
        *,
        teacher: Teacher | None,
        bc_coef: jax.Array,
    ) -> tuple[Params, optax.OptState, Normalizer, types.Metrics]:
        """One PPO update; a `teacher` adds `bc_coef * KL(teacher || self)`
        on the batch observations."""
        normalizer = running_statistics.update(normalizer, data.observation)
        grad_fn = jax.value_and_grad(self._loss, has_aux=True)

        def minibatch(carry: Any, batch: Any) -> tuple[Any, types.Metrics]:
            params, opt_state, key = carry
            key, loss_key = jax.random.split(key)
            (_, metrics), grads = grad_fn(
                params,
                normalizer,
                batch,
                loss_key,
                teacher=teacher,
                bc_coef=bc_coef,
            )
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state, key), metrics

        def epoch(carry: Any, _: None) -> tuple[Any, types.Metrics]:
            params, opt_state, key = carry
            key, perm_key = jax.random.split(key)

            def shuffle(x: jax.Array) -> jax.Array:
                x = jax.random.permutation(perm_key, x)
                return x.reshape((self.ppo.num_minibatches, -1, *x.shape[1:]))

            batches = jax.tree.map(shuffle, data)
            return jax.lax.scan(minibatch, (params, opt_state, key), batches)

        (params, opt_state, _), metrics = jax.lax.scan(
            epoch,
            (params, opt_state, key),
            None,
            self.ppo.num_updates_per_batch,
        )
        return params, opt_state, normalizer, jax.tree.map(jnp.mean, metrics)

    def _regret(
        self, params: Params, normalizer: Normalizer, data: types.Transition
    ) -> jax.Array:
        """Per env: mean positive GAE advantage (positive value loss)."""
        value_apply = self.network.value_network.apply
        data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), data)
        values = value_apply(normalizer, params.value, data.observation)
        last_obs = jax.tree.map(lambda x: x[-1], data.next_observation)
        bootstrap = value_apply(normalizer, params.value, last_obs)
        truncation = data.extras["state_extras"]["truncation"]
        _, advantages = ppo_losses.compute_gae(
            truncation=truncation,
            termination=(1 - data.discount) * (1 - truncation),
            rewards=data.reward,
            values=values,
            bootstrap_value=bootstrap,
            lambda_=self.ppo.gae_lambda,
            discount=self.ppo.discounting,
        )
        return jnp.maximum(advantages, 0.0).mean(axis=0)
