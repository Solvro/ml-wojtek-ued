import functools
import time
from typing import Any, cast

import equinox as eqx
import hydra
import jax
import optax
import wandb
from omegaconf import OmegaConf

from config.ppo_config import Config
from modules.ppo_modules import A2C
from utils.checkpoint import (
    make_checkpoint_manager,
    resolve_run_name,
    run_dir,
    save_checkpoint,
    save_meta,
)
from utils.env import FunctionalEnv
from utils.logging import Logger, quiet_dependencies
from utils.ppo_helpers import (
    batch_rollout,
    calculate_episode_advantage,
    eval,
    make_step,
    run_ppo_epochs,
)

ALGO = "ppo"


@hydra.main(version_base=None, config_name="config")
def train(cfg: Config) -> None:
    jax.config.update("jax_platforms", "cuda,cpu")
    quiet_dependencies()

    name = resolve_run_name(cfg.wandb_params.name)

    run = (
        wandb.init(
            entity=cfg.wandb_params.entity,
            project=cfg.wandb_params.project,
            tags=cfg.wandb_params.tags,
            name=name,
            config=cast(
                dict[str, Any],
                OmegaConf.to_container(
                    cfg,
                    resolve=True,
                ),
            ),
        )
        if cfg.wandb_params.enable
        else Logger(name=name)
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

    directory = run_dir(cfg.training_params.artifacts_dir, ALGO, name)
    ckptr = make_checkpoint_manager(
        directory,
        max_to_keep=cfg.training_params.max_to_keep,
        save_interval_steps=cfg.training_params.save_every,
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

    save_meta(
        directory,
        {
            "algo": ALGO,
            "created_at": time.time(),
            "env_task": cfg.environment_params.name,
            "env_overrides": None,
            "env_seed": cfg.environment_params.seed,
            "dense_dim": cfg.model_params.dense_dim,
            "activation": cfg.model_params.activation,
            "action_dim": action_dim,
            "actor_obs_dim": obs_size.actor,
            "critic_obs_dim": obs_size.critic,
            "action_high": action_high,
        },
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

    loss = 0

    for iteration in range(cfg.training_params.iterations):
        key, rollout_key, update_key = jax.random.split(key, 3)

        episode = jit_batch_rollout(rollout_key, a2c)
        advantages, returns = jit_advantage(a2c, episode)

        a2c, opt_state, loss = run_ppo_epochs(
            jit_make_step,
            a2c,
            opt_state,
            episode,
            advantages,
            returns,
            update_key,
            epochs=cfg.training_params.epochs,
            minibatches=cfg.training_params.minibatches,
        )

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
