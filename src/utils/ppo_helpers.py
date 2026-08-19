import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from mujoco import mjx

from modules.ppo_modules import A2C, Episode
from utils.checkpoint import (
    Algo,
    load_meta,
    open_checkpoint_manager,
    restore_checkpoint,
    run_dir,
)
from utils.env import FunctionalEnv


def clip_log_std(model, low=-5.0, high=0.5):
    clipped = jnp.clip(model.actor_log_std, low, high)
    return eqx.tree_at(lambda m: m.actor_log_std, model, clipped)


def warm_start_actor(artifacts_dir: str, algo: Algo, run: str, target: A2C) -> A2C:
    directory = run_dir(artifacts_dir, algo.value, run)
    meta = load_meta(directory)
    target_actor_dim = target.actor_mean.layers[0].in_features
    if meta["actor_obs_dim"] != target_actor_dim:
        raise ValueError(
            f"warm start actor_obs_dim mismatch: donor {run!r} ({algo.value}) "
            f"has {meta['actor_obs_dim']}, target has {target_actor_dim} — "
            "obs.state must match between the two runs"
        )
    donor_template = A2C(
        dense_dim=meta["dense_dim"],
        actor_obs_dim=meta["actor_obs_dim"],
        critic_obs_dim=meta["critic_obs_dim"],
        action_dim=meta["action_dim"],
        action_high=meta["action_high"],
        key=jax.random.key(0),
        activation=meta["activation"],
    )
    donor, _ = restore_checkpoint(
        open_checkpoint_manager(directory), None, donor_template
    )
    return eqx.tree_at(
        lambda m: (m.actor_mean, m.actor_log_std),
        target,
        (donor.actor_mean, donor.actor_log_std),
    )


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
    hfield_data: jax.Array | None = None,
    dr_model: mjx.Model | None = None,
) -> tuple[list, Episode]:
    obs, state, key = state_input
    key, key_step, key_act = jax.random.split(key, 3)
    action, log_prob, value, entropy = model(obs, key_act)
    next_obs, next_state, reward, done, _ = env.step(
        key_step, state, action, hfield_data, dr_model
    )
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
    hfield_data: jax.Array | None = None,
    dr_model: mjx.Model | None = None,
    dr_in_axes: mjx.Model | None = None,
) -> Episode:
    key_reset, key_episode = jax.random.split(key_input)
    reset_keys = jax.random.split(key_reset, envs_num)
    episode_keys = jax.random.split(key_episode, envs_num)

    obs, state = jax.vmap(
        lambda k, dr: env.reset(k, hfield_data, dr),
        in_axes=(0, dr_in_axes),
    )(reset_keys, dr_model)
    step_fn = jax.vmap(
        lambda state_input, tmp, dr: policy_step(
            env, model, state_input, tmp, hfield_data=hfield_data, dr_model=dr
        ),
        in_axes=(0, None, dr_in_axes),
    )
    _, episode = jax.lax.scan(
        lambda carry, x: step_fn(carry, x, dr_model),
        [obs, state, episode_keys],
        (),
        steps_in_episode,
    )
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), episode)


def lanes_rollout(
    env: FunctionalEnv,
    key_input: jax.Array,
    model: A2C,
    num_lanes: int,
    envs_per_lane: int,
    steps_in_episode: int,
    hfield_data: jax.Array,
) -> Episode:
    keys = jax.random.split(key_input, num_lanes)

    def rollout_fn(key: jax.Array, model: A2C, hfield_data: jax.Array) -> Episode:
        return batch_rollout(
            env, key, model, envs_per_lane, steps_in_episode, hfield_data=hfield_data
        )

    return jax.vmap(rollout_fn, in_axes=(0, None, 0))(keys, model, hfield_data)


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


def run_ppo_epochs(
    jit_make_step,
    model: A2C,
    opt_state,
    episode: Episode,
    advantages: jax.Array,
    returns: jax.Array,
    key: jax.Array,
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
            model, opt_state, loss = jit_make_step(
                model,
                opt_state,
                jax.tree.map(lambda x, idx=idx: x[idx], obs),
                actions[idx],
                old_log_prob[idx],
                advantages_flat[idx],
                returns_flat[idx],
                minibatch_keys[idx],
            )

    return model, opt_state, loss


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
    return rewards.sum(axis=0).mean()


FALL_CAUSES = ("min_height", "tilt", "toggle", "base_contact")


def level_eval_step(
    env: FunctionalEnv,
    model: A2C,
    state_input: list,
    tmp: tuple[()],
    hfield_data: jax.Array,
) -> tuple[list, tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]]:
    obs, state, key = state_input
    key, key_step = jax.random.split(key)
    action = model.deterministic_action(obs)
    next_obs, next_state, reward, done, info = env.step(
        key_step, state, action, hfield_data
    )
    carry = [next_obs, next_state, key]
    alive = 1.0 - info["truncation"]
    fell = done * alive
    cause_at_done = {
        cause: next_state.metrics[f"fall_{cause}_at_done"] * alive
        for cause in FALL_CAUSES
    }
    base_contact_alive = next_state.metrics["base_contact_alive_per_step"]
    return carry, (reward, fell, base_contact_alive, cause_at_done)


def level_eval_rollout(
    env: FunctionalEnv,
    key_input: jax.Array,
    model: A2C,
    steps: int,
    hfield_data: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    key_reset, key_episode = jax.random.split(key_input)
    obs, state = env.reset(key_reset, hfield_data)
    step_fn = functools.partial(level_eval_step, env, model, hfield_data=hfield_data)
    _, (rewards, fell, base_contact_alive, cause_at_done) = jax.lax.scan(
        step_fn, [obs, state, key_episode], (), steps
    )
    causes = {cause: jnp.any(cause_at_done[cause] > 0) for cause in FALL_CAUSES}
    return rewards.sum(), jnp.any(fell > 0), base_contact_alive.mean(), causes


def eval_levels(
    env: FunctionalEnv,
    key_input: jax.Array,
    model: A2C,
    hfield_data: jax.Array,
    episodes_per_level: int,
    steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    num_levels = hfield_data.shape[0]
    level_keys = jax.random.split(key_input, num_levels)

    def per_level(level_key: jax.Array, level_hfield: jax.Array):
        episode_keys = jax.random.split(level_key, episodes_per_level)
        rollout_fn = functools.partial(
            level_eval_rollout,
            env,
            model=model,
            steps=steps,
            hfield_data=level_hfield,
        )
        return jax.vmap(rollout_fn)(episode_keys)

    return jax.vmap(per_level)(level_keys, hfield_data)
