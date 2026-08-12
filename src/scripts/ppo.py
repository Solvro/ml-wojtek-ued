import functools
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import hydra
import jax
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf

from config.ppo_config import Config
from modules.ppo_modules import A2C
from utils.checkpoint import save_checkpoint
from utils.env import FunctionalEnv
from utils.logging import Logger, quiet_dependencies
from utils.ppo_helpers import (
    batch_rollout,
    calculate_episode_advantage,
    clip_log_std,
    eval,
    is_finite,
    make_step,
)


@hydra.main(version_base=None, config_name="config")
def train(cfg: Config) -> None:
    jax.config.update("jax_platforms", "cuda,cpu")
    quiet_dependencies()

    run = (
        wandb.init(
            entity=cfg.wandb_params.entity,
            project=cfg.wandb_params.project,
            tags=cfg.wandb_params.tags,
            name=cfg.wandb_params.name,
            config=cast(
                dict[str, Any],
                OmegaConf.to_container(
                    cfg,
                    resolve=True,
                ),
            ),
        )
        if cfg.wandb_params.enable
        else Logger(name=cfg.wandb_params.name)
    )

    env = FunctionalEnv(
        task=cfg.environment_params.name,
        seed=cfg.environment_params.seed,
    )

    action_dim = env.action_size
    obs_size = env.observation_size

    action_high = 1

    key = jax.random.key(cfg.environment_params.seed)
    model_key, key = jax.random.split(key)

    options = ocp.CheckpointManagerOptions(
        max_to_keep=cfg.training_params.max_to_keep,
        save_interval_steps=cfg.training_params.save_every,
    )
    directory = Path(cfg.training_params.artifacts_dir).resolve()
    ckptr = ocp.CheckpointManager(
        directory, item_names=("model", "opt_state"), options=options
    )

    a2c = A2C(
        dense_dim=cfg.model_params.dense_dim,
        actor_obs_dim=obs_size.actor,
        critic_obs_dim=obs_size.critic,
        action_dim=action_dim,
        action_high=action_high,
        key=model_key,
        activation=cfg.model_params.activation,
    )

    linear_scheduler = optax.linear_schedule(
        init_value=cfg.training_params.learning_rate,
        end_value=cfg.training_params.end_learning_rate,
        transition_steps=cfg.training_params.iterations
        * cfg.training_params.epochs
        * cfg.training_params.minibatches,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training_params.max_grad_norm),
        optax.adam(
            learning_rate=linear_scheduler,
            eps=cfg.training_params.adam_eps,
        ),
    )
    opt_state = optimizer.init(eqx.filter(a2c, eqx.is_array))

    jit_batch_rollout = eqx.filter_jit(
        functools.partial(
            batch_rollout,
            env,
            envs_num=cfg.environment_params.envs_num,
            steps_in_episode=cfg.environment_params.steps_in_episode,
        )
    )
    jit_advantage = eqx.filter_jit(
        functools.partial(
            calculate_episode_advantage,
            gamma=cfg.training_params.gamma,
            lam=cfg.training_params.lam,
        )
    )
    jit_make_step = eqx.filter_jit(functools.partial(make_step, optimizer))

    jit_eval = eqx.filter_jit(
        functools.partial(
            eval,
            env,
            envs_num=cfg.evaluation_params.envs_num,
            steps=cfg.evaluation_params.steps,
        )
    )

    batch_size = (
        cfg.environment_params.envs_num * cfg.environment_params.steps_in_episode
    )
    minibatch_size = batch_size // cfg.training_params.minibatches

    loss = 0

    for iteration in range(cfg.training_params.iterations):
        key, rollout_key, update_key = jax.random.split(key, 3)

        episode = jit_batch_rollout(rollout_key, a2c)
        advantages, returns = jit_advantage(a2c, episode)

        obs = jax.tree.map(lambda x: x.reshape(batch_size, -1), episode.obs)
        actions = episode.action.reshape(batch_size, -1)
        old_log_prob = episode.log_prob.reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)
        returns_flat = returns.reshape(batch_size)

        for _ in range(cfg.training_params.epochs):
            update_key, perm_key, batch_key = jax.random.split(update_key, 3)
            perm = jax.random.permutation(perm_key, batch_size)
            minibatch_keys = jax.random.split(batch_key, batch_size)

            for start in range(0, batch_size, minibatch_size):
                idx = perm[start : start + minibatch_size]
                new_a2c, new_opt_state, loss = jit_make_step(
                    a2c,
                    opt_state,
                    jax.tree.map(lambda x, idx=idx: x[idx], obs),
                    actions[idx],
                    old_log_prob[idx],
                    advantages_flat[idx],
                    returns_flat[idx],
                    minibatch_keys[idx],
                )
                # A single saturated tanh sample can turn a step's loss (and
                # thus its gradients and the Adam moments) into NaN, which
                # never recovers on its own. Drop the step instead of
                # applying it, and keep actor_log_std out of the regime that
                # produces those saturated samples in the first place.
                if is_finite(new_a2c):
                    a2c = clip_log_std(new_a2c)
                    opt_state = new_opt_state

        mean_reward = episode.reward.sum(axis=1).mean()

        real_step = (
            iteration * cfg.training_params.epochs * cfg.training_params.minibatches
        )
        lr = linear_scheduler(real_step)

        if iteration % cfg.training_params.eval_every == 0:
            key, eval_key = jax.random.split(key)
            mean_return = jit_eval(a2c, eval_key)
            run.log({"mean_return": mean_return})

        run.log(
            {
                "iteration": iteration,
                "loss": loss,
                "mean_reward": mean_reward,
                "learning_rate": lr,
            }
        )
        save_checkpoint(ckptr, iteration, a2c, opt_state)

    ckptr.wait_until_finished()

    run.finish()


if __name__ == "__main__":
    train()
