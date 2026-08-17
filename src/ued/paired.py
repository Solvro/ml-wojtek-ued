from collections.abc import Callable
from typing import Any, NamedTuple

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hfield.manager import HFieldManager
from modules.generator import LevelGenerator
from modules.ppo_modules import A2C, Episode
from ued.levels import (
    level_to_hfield_data,
    level_to_hfield_data_batch,
    sample_from_buffer,
)
from ued.types import Curator
from utils.ppo_helpers import clip_log_std, is_finite_array, ppo_loss


class PairedTrainer(NamedTuple):
    sample_levels: Callable  # (generator, z, keys) -> (thetas, log_probs)
    rollout: Callable  # (key, model, hfield_data) -> episode  (lanes_rollout)
    advantage: Callable  # (model, episode) -> (advantages, returns)
    student_step: Callable  # jitted student_make_step
    gen_step: Callable  # jitted generator_make_step
    hfield_manager: HFieldManager
    num_lanes: int
    latent_dim: int
    student_epochs: int
    student_minibatches: int
    gen_epochs: int
    bc_mode: str  # "none" | "uni" | "bi"
    bc_coef: float
    kl_update_step: int


class PairedEvoTrainer(NamedTuple):
    rollout: Callable  # (key, model, hfield_data=...) -> episode  (batch_rollout)
    advantage: Callable  # (model, episode) -> (advantages, returns)
    student_step: Callable  # jitted student_make_step (BC-aware)
    epochs: int
    minibatches: int
    bc_mode: str  # "none" | "uni" | "bi"
    bc_coef: float
    kl_update_step: int


def flatten_lanes(episode: Episode) -> Episode:
    return jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), episode)


def sample_levels(
    generator: LevelGenerator, z: jax.Array, keys: jax.Array
) -> tuple[jax.Array, jax.Array]:
    thetas, log_probs, _ = jax.vmap(generator)(z, keys)
    return thetas, log_probs


def paired_regret(
    prot_episode: Episode, antag_episode: Episode
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    regret = max(antagonist_max_return - protagonist_mean_return, 0) over
    the ``envs_per_lane`` episodes rolled out on each level
    """
    prot_return = prot_episode.reward.sum(axis=2)  # (lanes, envs)
    antag_return = antag_episode.reward.sum(axis=2)
    prot_mean = prot_return.mean(axis=1)
    antag_max = antag_return.max(axis=1)
    regret = jnp.maximum(antag_max - prot_mean, 0.0)
    return regret, prot_mean, antag_return.mean(axis=1)


def paired_regret_single(
    prot_episode: Episode, antag_episode: Episode
) -> tuple[jax.Array, jax.Array]:
    """PAIRED regret for a single level rolled out over ``envs_num`` episodes
    (the ``batch_rollout`` shape ``(envs, steps, ...)``). Returns
    ``(regret, prot_mean)`` scalars. Used by the Evo variant."""
    prot_return = prot_episode.reward.sum(axis=1)  # (envs,)
    antag_return = antag_episode.reward.sum(axis=1)
    prot_mean = prot_return.mean()
    antag_max = antag_return.max()
    return jnp.maximum(antag_max - prot_mean, 0.0), prot_mean


def _policy_kl(model: A2C, ref_model: A2C, obs) -> jax.Array:
    """KL(pi_ref || pi_model) per state."""
    model_mean = jax.vmap(model.actor_mean)(obs.actor)
    ref_mean = jax.lax.stop_gradient(jax.vmap(ref_model.actor_mean)(obs.actor))
    model_std = jnp.broadcast_to(jnp.exp(model.actor_log_std), model_mean.shape)
    ref_std = jnp.broadcast_to(
        jnp.exp(jax.lax.stop_gradient(ref_model.actor_log_std)), ref_mean.shape
    )
    ref_dist = distrax.MultivariateNormalDiag(ref_mean, ref_std)
    model_dist = distrax.MultivariateNormalDiag(model_mean, model_std)
    return jnp.asarray(ref_dist.kl_divergence(model_dist))


def student_ppo_loss(
    model,
    ref_model,
    obs,
    actions,
    old_log_prob,
    advantages,
    returns,
    keys,
    bc_coef,
    *,
    epsilon=0.1,
    value_coef=0.5,
    entropy_coef=0.01,
):
    base = ppo_loss(
        model,
        obs,
        actions,
        old_log_prob,
        advantages,
        returns,
        keys,
        epsilon=epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
    )
    bc_loss = jnp.mean(_policy_kl(model, ref_model, obs))
    return base + bc_coef * bc_loss


def student_make_step(
    optimizer,
    model,
    ref_model,
    opt_state,
    obs,
    actions,
    old_log_prob,
    advantages,
    returns,
    keys,
    bc_coef,
    *,
    epsilon=0.1,
    value_coef=0.5,
    entropy_coef=0.01,
):
    loss_value, grads = eqx.filter_value_and_grad(student_ppo_loss)(
        model,
        ref_model,
        obs,
        actions,
        old_log_prob,
        advantages,
        returns,
        keys,
        bc_coef,
        epsilon=epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
    )
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    new_model = clip_log_std(eqx.apply_updates(model, updates))
    finite = is_finite_array(new_model)
    new_arr, _ = eqx.partition(new_model, eqx.is_array)
    old_arr, static = eqx.partition(model, eqx.is_array)
    model = eqx.combine(
        jax.tree.map(lambda n, o: jnp.where(finite, n, o), new_arr, old_arr),
        static,
    )
    opt_state = jax.tree.map(
        lambda n, o: jnp.where(finite, n, o), new_opt_state, opt_state
    )
    return model, opt_state, loss_value


def run_student_epochs(
    jit_student_step,
    model: A2C,
    ref_model: A2C,
    opt_state,
    episode: Episode,
    advantages: jax.Array,
    returns: jax.Array,
    key: jax.Array,
    bc_coef: jax.Array,
    epochs: int,
    minibatches: int,
):
    batch_size = episode.action.shape[0] * episode.action.shape[1]
    minibatch_size = batch_size // minibatches

    obs = jax.tree.map(lambda x: x.reshape(batch_size, -1), episode.obs)
    actions = episode.action.reshape(batch_size, -1)
    old_log_prob = episode.log_prob.reshape(batch_size)
    advantages_flat = advantages.reshape(batch_size)
    returns_flat = returns.reshape(batch_size)

    loss = jnp.zeros(())
    for _ in range(epochs):
        key, perm_key, batch_key = jax.random.split(key, 3)
        perm = jax.random.permutation(perm_key, batch_size)
        minibatch_keys = jax.random.split(batch_key, batch_size)

        for start in range(0, batch_size, minibatch_size):
            idx = perm[start : start + minibatch_size]
            model, opt_state, loss = jit_student_step(
                model,
                ref_model,
                opt_state,
                jax.tree.map(lambda x, idx=idx: x[idx], obs),
                actions[idx],
                old_log_prob[idx],
                advantages_flat[idx],
                returns_flat[idx],
                minibatch_keys[idx],
                bc_coef,
            )

    return model, opt_state, loss


def _clip_gen_log_std(generator: LevelGenerator, low=-5.0, high=2.0):
    clipped = jnp.clip(generator.log_std, low, high)
    return eqx.tree_at(lambda g: g.log_std, generator, clipped)


def generator_loss(
    generator: LevelGenerator,
    z: jax.Array,
    theta: jax.Array,
    old_log_prob: jax.Array,
    advantages: jax.Array,
    *,
    epsilon=0.2,
    entropy_coef=0.1,
):
    new_log_prob, entropy = jax.vmap(generator.evaluate)(z, theta)
    log_ratio = jnp.clip(new_log_prob - old_log_prob, -20.0, 20.0)
    ratio = jnp.exp(log_ratio)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    clipped = jnp.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped))
    entropy_loss = -jnp.mean(entropy)
    return policy_loss + entropy_coef * entropy_loss


def generator_make_step(
    optimizer,
    generator: LevelGenerator,
    gen_opt_state,
    z: jax.Array,
    theta: jax.Array,
    old_log_prob: jax.Array,
    advantages: jax.Array,
    *,
    epsilon=0.2,
    entropy_coef=0.1,
):
    loss_value, grads = eqx.filter_value_and_grad(generator_loss)(
        generator,
        z,
        theta,
        old_log_prob,
        advantages,
        epsilon=epsilon,
        entropy_coef=entropy_coef,
    )
    updates, new_opt_state = optimizer.update(
        grads, gen_opt_state, eqx.filter(generator, eqx.is_array)
    )
    new_generator = _clip_gen_log_std(eqx.apply_updates(generator, updates))
    finite = is_finite_array(new_generator)
    new_arr, _ = eqx.partition(new_generator, eqx.is_array)
    old_arr, static = eqx.partition(generator, eqx.is_array)
    generator = eqx.combine(
        jax.tree.map(lambda n, o: jnp.where(finite, n, o), new_arr, old_arr),
        static,
    )
    gen_opt_state = jax.tree.map(
        lambda n, o: jnp.where(finite, n, o), new_opt_state, gen_opt_state
    )
    return generator, gen_opt_state, loss_value


def run_generator_epochs(
    jit_gen_step,
    generator: LevelGenerator,
    gen_opt_state,
    z: jax.Array,
    theta: jax.Array,
    old_log_prob: jax.Array,
    regret: jax.Array,
    epochs: int,
):
    advantages = regret - regret.mean()
    loss = jnp.zeros(())
    for _ in range(epochs):
        generator, gen_opt_state, loss = jit_gen_step(
            generator, gen_opt_state, z, theta, old_log_prob, advantages
        )
    return generator, gen_opt_state, loss


def _bc_coef_for(
    trainer: PairedTrainer | PairedEvoTrainer, iteration: int, enabled: bool
) -> jax.Array:
    active = (
        enabled
        and trainer.bc_mode != "none"
        and (iteration % trainer.kl_update_step == 0)
    )
    return jnp.asarray(trainer.bc_coef if active else 0.0, dtype=jnp.float32)


def paired_iteration(
    key: jax.Array,
    prot: A2C,
    prot_opt: Any,
    antag: A2C,
    antag_opt: Any,
    generator: LevelGenerator,
    gen_opt: Any,
    trainer: PairedTrainer,
    iteration: int,
) -> tuple[jax.Array, A2C, Any, A2C, Any, LevelGenerator, Any, dict[str, Any]]:
    key, z_key, sample_key, prot_key, antag_key, prot_up, antag_up = jax.random.split(
        key, 7
    )

    z = jax.random.normal(z_key, (trainer.num_lanes, trainer.latent_dim))
    sample_keys = jax.random.split(sample_key, trainer.num_lanes)
    thetas, gen_log_prob = trainer.sample_levels(generator, z, sample_keys)
    hfield_data = level_to_hfield_data_batch(trainer.hfield_manager, np.asarray(thetas))

    prot_episode = trainer.rollout(prot_key, prot, hfield_data=hfield_data)
    antag_episode = trainer.rollout(antag_key, antag, hfield_data=hfield_data)

    regret, prot_mean, antag_mean = paired_regret(prot_episode, antag_episode)

    prot_flat = flatten_lanes(prot_episode)
    prot_adv, prot_ret = trainer.advantage(prot, prot_flat)
    prot_bc = _bc_coef_for(trainer, iteration, enabled=trainer.bc_mode in ("uni", "bi"))
    prot, prot_opt, prot_loss = run_student_epochs(
        trainer.student_step,
        prot,
        antag,  # protagonist distils from the antagonist
        prot_opt,
        prot_flat,
        prot_adv,
        prot_ret,
        prot_up,
        prot_bc,
        trainer.student_epochs,
        trainer.student_minibatches,
    )

    antag_flat = flatten_lanes(antag_episode)
    antag_adv, antag_ret = trainer.advantage(antag, antag_flat)
    antag_bc = _bc_coef_for(trainer, iteration, enabled=trainer.bc_mode == "bi")
    antag, antag_opt, antag_loss = run_student_epochs(
        trainer.student_step,
        antag,
        prot,  # antagonist distils from the updated protagonist under BiBC
        antag_opt,
        antag_flat,
        antag_adv,
        antag_ret,
        antag_up,
        antag_bc,
        trainer.student_epochs,
        trainer.student_minibatches,
    )

    generator, gen_opt, gen_loss = run_generator_epochs(
        trainer.gen_step,
        generator,
        gen_opt,
        z,
        thetas,
        gen_log_prob,
        regret,
        trainer.gen_epochs,
    )

    log = {
        "regret_score": float(regret.mean()),
        "max_regret_score": float(regret.max()),
        "prot_return": float(prot_mean.mean()),
        "antag_return": float(antag_mean.mean()),
        "mean_reward": float(prot_mean.mean()),
        "prot_loss": float(prot_loss),
        "antag_loss": float(antag_loss),
        "gen_loss": float(gen_loss),
        "bc_coef": float(prot_bc),
    }
    return key, prot, prot_opt, antag, antag_opt, generator, gen_opt, log


def score_regret_evo(
    key: jax.Array,
    prot: A2C,
    antag: A2C,
    level: np.ndarray,
    curator: Curator,
    trainer: PairedEvoTrainer,
) -> tuple[float, float]:
    hfield_data = level_to_hfield_data(curator.hfield_manager, level)
    prot_key, antag_key = jax.random.split(key)
    prot_episode = trainer.rollout(prot_key, prot, hfield_data=hfield_data)
    antag_episode = trainer.rollout(antag_key, antag, hfield_data=hfield_data)
    regret, prot_mean = paired_regret_single(prot_episode, antag_episode)
    regret = float(regret)
    curator.buffer.add(level, regret)
    return regret, float(prot_mean)


def paired_evo_generate_branch(
    key: jax.Array,
    prot: A2C,
    antag: A2C,
    curator: Curator,
    trainer: PairedEvoTrainer,
) -> tuple[jax.Array, dict[str, Any]]:
    key, rollout_key = jax.random.split(key)
    level = curator.hfield_manager.generate(curator.rng)
    regret, prot_mean = score_regret_evo(
        rollout_key, prot, antag, level, curator, trainer
    )
    log = {"branch": 0, "regret_score": regret, "mean_reward": prot_mean}
    return key, log


def _update_student(
    key: jax.Array,
    student: A2C,
    ref: A2C,
    opt_state: Any,
    episode: Episode,
    trainer: PairedEvoTrainer,
    bc_coef: jax.Array,
) -> tuple[A2C, Any, jax.Array]:
    advantages, returns = trainer.advantage(student, episode)
    return run_student_epochs(
        trainer.student_step,
        student,
        ref,
        opt_state,
        episode,
        advantages,
        returns,
        key,
        bc_coef,
        trainer.epochs,
        trainer.minibatches,
    )


def paired_evo_replay_branch(
    key: jax.Array,
    prot: A2C,
    prot_opt: Any,
    antag: A2C,
    antag_opt: Any,
    curator: Curator,
    trainer: PairedEvoTrainer,
    iteration: int,
) -> tuple[jax.Array, A2C, Any, A2C, Any, dict[str, Any]]:
    key, train_key, prot_up, antag_up, edit_key = jax.random.split(key, 5)
    theta = sample_from_buffer(curator.buffer, curator.rng)
    hfield_data = level_to_hfield_data(curator.hfield_manager, theta)

    prot_rk, antag_rk = jax.random.split(train_key)
    prot_episode = trainer.rollout(prot_rk, prot, hfield_data=hfield_data)
    antag_episode = trainer.rollout(antag_rk, antag, hfield_data=hfield_data)

    prot_bc = _bc_coef_for(trainer, iteration, enabled=trainer.bc_mode in ("uni", "bi"))
    prot, prot_opt, prot_loss = _update_student(
        prot_up, prot, antag, prot_opt, prot_episode, trainer, prot_bc
    )
    antag_bc = _bc_coef_for(trainer, iteration, enabled=trainer.bc_mode == "bi")
    antag, antag_opt, antag_loss = _update_student(
        antag_up, antag, prot, antag_opt, antag_episode, trainer, antag_bc
    )

    theta_prime = curator.hfield_manager.mutate(theta, curator.rng)
    regret, prot_mean = score_regret_evo(
        edit_key, prot, antag, theta_prime, curator, trainer
    )

    log = {
        "branch": 1,
        "regret_score": regret,
        "mean_reward": prot_mean,
        "prot_loss": float(prot_loss),
        "antag_loss": float(antag_loss),
        "bc_coef": float(prot_bc),
    }
    return key, prot, prot_opt, antag, antag_opt, log
