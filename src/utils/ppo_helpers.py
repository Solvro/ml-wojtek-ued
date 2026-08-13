import functools

import equinox as eqx
import jax
import jax.numpy as jnp

from modules.ppo_modules import A2C, Episode
from utils.env import FunctionalEnv


def clip_log_std(model, low=-5.0, high=0.5):
    clipped = jnp.clip(model.actor_log_std, low, high)
    return eqx.tree_at(lambda m: m.actor_log_std, model, clipped)


def is_finite_array(model):
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    return jnp.all(jnp.array([jnp.all(jnp.isfinite(leave)) for leave in leaves]))


def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    def gae_step(carry, transition):
        next_advantage, next_value = carry
        reward, value, done = transition
        delta = reward + gamma * (1 - done) * next_value - value
        advantage = delta + (1 - done) * gamma * lam * next_advantage
        return (advantage, value), advantage

    _, advantages = jax.lax.scan(
        gae_step,
        (jnp.zeros_like(next_value), next_value),
        (rewards, values, dones),
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns


def policy_step(
    env: FunctionalEnv,
    model: A2C,
    state_input: list,
    tmp: tuple[()],
) -> tuple[list, Episode]:
    obs, state, key = state_input
    key, key_step, key_act = jax.random.split(key, 3)
    action, log_prob, value, entropy = model(obs, key_act)
    next_obs, next_state, reward, done, _ = env.step(key_step, state, action)
    carry = [next_obs, next_state, key]
    step_out = Episode(
        obs=obs,
        action=action,
        log_prob=log_prob,
        value=value,
        entropy=entropy,
        reward=reward,
        next_obs=next_obs,
        done=done,
    )
    return carry, step_out


def batch_rollout(
    env: FunctionalEnv,
    key_input: jax.Array,
    model: A2C,
    envs_num: int,
    steps_in_episode: int,
) -> Episode:
    key_reset, key_episode = jax.random.split(key_input)
    reset_keys = jax.random.split(key_reset, envs_num)
    episode_keys = jax.random.split(key_episode, envs_num)

    obs, state = jax.vmap(env.reset)(reset_keys)
    step_fn = jax.vmap(functools.partial(policy_step, env, model))
    _, episode = jax.lax.scan(
        step_fn,
        [obs, state, episode_keys],
        (),
        steps_in_episode,
    )
    # scan stacks steps on axis 0 and vmap adds envs on axis 1, giving
    # (steps, envs, ...). Callers (calculate_episode_advantage's GAE scan,
    # the reshape in ppo.py) expect envs-major (envs, steps, ...), so swap
    # back right here rather than pushing the time-major convention
    # through every downstream consumer.
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), episode)


def calculate_episode_advantage(model, episode, gamma=0.99, lam=0.95):
    next_values = jax.vmap(model.critic)(
        episode.next_obs.critic[:, -1, :],
    ).squeeze(-1)
    advantages, returns = jax.vmap(
        compute_gae,
        in_axes=(0, 0, 0, 0, None, None),
    )(
        episode.reward,
        episode.value.squeeze(-1),
        episode.done,
        next_values,
        gamma,
        lam,
    )
    return advantages, returns


def ppo_loss(
    model,
    obs,
    actions,
    old_log_prob,
    advantages,
    returns,
    keys,
    epsilon=0.1,
    value_coef=0.5,
    entropy_coef=0.01,
):
    _, new_log_prob, value, entropy = jax.vmap(model, in_axes=(0, 0, 0))(
        obs, keys, actions
    )
    log_ratio = jnp.clip(new_log_prob - old_log_prob, -20.0, 20.0)
    ratio = jnp.exp(log_ratio)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    clipped = jnp.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages

    policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped))
    value_loss = jnp.mean((value.squeeze(-1) - returns) ** 2)
    entropy_loss = -jnp.mean(entropy)

    return policy_loss + value_coef * value_loss + entropy_coef * entropy_loss


def make_step(
    optimizer,
    model,
    opt_state,
    obs,
    actions,
    old_log_prob,
    advantages,
    returns,
    keys,
):
    loss_value, grads = eqx.filter_value_and_grad(ppo_loss)(
        model, obs, actions, old_log_prob, advantages, returns, keys
    )
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    new_model = clip_log_std(eqx.apply_updates(model, updates))
    # A single saturated tanh sample can turn a step's loss (and thus its
    # gradients and the Adam moments) into NaN, which never recovers on its
    # own. Select the old model/opt_state for such a step instead of applying
    # it, inside jit so the finite check never forces a device->host sync in
    # the training loop.
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


def eval_step(
    env: FunctionalEnv,
    model: A2C,
    state_input: list,
    tmp: tuple[()],
) -> tuple[list, jax.Array]:
    obs, state, key = state_input
    key, key_step = jax.random.split(key)
    action = model.deterministic_action(obs)
    next_obs, next_state, reward, _, _ = env.step(key_step, state, action)
    carry = [next_obs, next_state, key]
    return carry, reward


def eval(
    env: FunctionalEnv,
    model: A2C,
    key: jax.Array,
    envs_num: int,
    steps: int,
) -> jax.Array:
    key_reset, key_episode = jax.random.split(key)
    reset_keys = jax.random.split(key_reset, envs_num)
    episode_keys = jax.random.split(key_episode, envs_num)

    obs, state = jax.vmap(env.reset)(reset_keys)
    step_fn = jax.vmap(functools.partial(eval_step, env, model))
    _, rewards = jax.lax.scan(step_fn, [obs, state, episode_keys], (), steps)
    # scan(vmap(step)) stacks as (steps, envs) -- sum over steps (axis 0)
    # for per-env return, then mean over envs.
    return rewards.sum(axis=0).mean()
