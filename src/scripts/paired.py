import functools
import time
from typing import Any, cast

import equinox as eqx
import hydra
import jax
import optax
import wandb
from omegaconf import OmegaConf

from config.hfield_config import HFieldConfig
from config.paired_config import PairedConfig
from hfield.manager import HFieldManager
from modules.generator import LevelGenerator
from modules.ppo_modules import A2C
from ued.paired import (
    PairedTrainer,
    generator_make_step,
    paired_iteration,
    sample_levels,
    student_make_step,
)
from ued.stopper import PlateauStopper
from utils.checkpoint import (
    make_checkpoint_manager,
    resolve_run_name,
    run_dir,
    save_checkpoint,
    save_meta,
)
from utils.env import FunctionalEnv
from utils.env_overrides import merge_overrides, movement_overrides, terrain_overrides
from utils.logging import Logger, quiet_dependencies
from utils.ppo_helpers import calculate_episode_advantage, eval, lanes_rollout

ALGO = "paired"


@hydra.main(version_base=None, config_name="paired_config")
def train(cfg: PairedConfig) -> None:
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
                OmegaConf.to_container(cfg, resolve=True),
            ),
        )
        if cfg.wandb_params.enable
        else Logger(name=name)
    )

    num_lanes = cfg.paired_params.num_lanes
    envs_per_lane = cfg.paired_params.envs_per_lane
    rollout_envs = num_lanes * envs_per_lane

    env_overrides = merge_overrides(
        {
            "terrain": {
                "enable": True,
                "arena": "accel",
                "pad_radius": cfg.hfield_params.pad_radius,
            }
        },
        movement_overrides(),
        terrain_overrides(),
    )
    env = FunctionalEnv(
        task=cfg.environment_params.name,
        env_overrides={**env_overrides, "sim": {"num_envs": rollout_envs}},
        seed=cfg.environment_params.seed,
    )
    eval_env = FunctionalEnv(
        task=cfg.environment_params.name,
        env_overrides={
            **env_overrides,
            "sim": {"num_envs": cfg.evaluation_params.envs_num},
        },
        seed=cfg.environment_params.seed,
    )

    action_dim = env.action_size
    obs_size = env.observation_size
    action_high = 1

    key = jax.random.key(cfg.environment_params.seed)
    prot_key, antag_key, gen_key, key = jax.random.split(key, 4)

    directory = run_dir(cfg.training_params.artifacts_dir, ALGO, name)
    ckptr = make_checkpoint_manager(
        directory,
        max_to_keep=cfg.training_params.max_to_keep,
        save_interval_steps=cfg.training_params.save_every,
    )

    def make_student(model_key: jax.Array) -> A2C:
        return A2C(
            dense_dim=cfg.model_params.dense_dim,
            actor_obs_dim=obs_size.actor,
            critic_obs_dim=obs_size.critic,
            action_dim=action_dim,
            action_high=action_high,
            key=model_key,
            activation=cfg.model_params.activation,
        )

    prot = make_student(prot_key)
    antag = make_student(antag_key)
    hfield_config = cast(HFieldConfig, OmegaConf.to_object(cfg.hfield_params))
    generator = LevelGenerator(
        latent_dim=cfg.paired_params.latent_dim,
        dense_dim=cfg.paired_params.gen_dense_dim,
        coarse_n=hfield_config.coarse_n,
        key=gen_key,
        activation=cfg.paired_params.gen_activation,
    )

    save_meta(
        directory,
        {
            "algo": ALGO,
            "created_at": time.time(),
            "env_task": cfg.environment_params.name,
            "env_overrides": env_overrides,
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
        transition_steps=cfg.paired_params.max_iterations
        * cfg.training_params.epochs
        * cfg.training_params.minibatches,
    )
    student_optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training_params.max_grad_norm),
        optax.adam(learning_rate=linear_scheduler, eps=cfg.training_params.adam_eps),
    )
    prot_opt = student_optimizer.init(eqx.filter(prot, eqx.is_array))
    antag_opt = student_optimizer.init(eqx.filter(antag, eqx.is_array))

    gen_optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.paired_params.gen_max_grad_norm),
        optax.adam(
            learning_rate=cfg.paired_params.gen_learning_rate,
            eps=cfg.training_params.adam_eps,
        ),
    )
    gen_opt = gen_optimizer.init(eqx.filter(generator, eqx.is_array))

    jit_sample_levels = eqx.filter_jit(sample_levels)
    jit_lanes_rollout = eqx.filter_jit(
        functools.partial(
            lanes_rollout,
            env,
            num_lanes=num_lanes,
            envs_per_lane=envs_per_lane,
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
    jit_student_step = eqx.filter_jit(
        functools.partial(
            student_make_step,
            student_optimizer,
            epsilon=cfg.training_params.epsilon,
            value_coef=cfg.training_params.value_coef,
            entropy_coef=cfg.paired_params.student_entropy_coef,
        )
    )
    jit_gen_step = eqx.filter_jit(
        functools.partial(
            generator_make_step,
            gen_optimizer,
            epsilon=cfg.paired_params.gen_epsilon,
            entropy_coef=cfg.paired_params.gen_entropy_coef,
        )
    )
    jit_eval = eqx.filter_jit(
        functools.partial(
            eval,
            eval_env,
            envs_num=cfg.evaluation_params.envs_num,
            steps=cfg.evaluation_params.steps,
        )
    )

    trainer = PairedTrainer(
        sample_levels=jit_sample_levels,
        rollout=jit_lanes_rollout,
        advantage=jit_advantage,
        student_step=jit_student_step,
        gen_step=jit_gen_step,
        hfield_manager=HFieldManager(hfield_config),
        num_lanes=num_lanes,
        latent_dim=cfg.paired_params.latent_dim,
        student_epochs=cfg.training_params.epochs,
        student_minibatches=cfg.training_params.minibatches,
        gen_epochs=cfg.paired_params.gen_epochs,
        bc_mode=cfg.paired_params.bc_mode,
        bc_coef=cfg.paired_params.bc_coef,
        kl_update_step=cfg.paired_params.kl_update_step,
    )

    stopper = PlateauStopper(
        window=cfg.paired_params.plateau_window,
        patience=cfg.paired_params.plateau_patience,
        rel_tol=cfg.paired_params.plateau_rel_tol,
        min_iterations=cfg.paired_params.min_iterations,
    )

    iteration = 0
    converged = False

    while not converged:
        (
            key,
            prot,
            prot_opt,
            antag,
            antag_opt,
            generator,
            gen_opt,
            step_log,
        ) = paired_iteration(
            key,
            prot,
            prot_opt,
            antag,
            antag_opt,
            generator,
            gen_opt,
            trainer,
            iteration,
        )

        plateaued = stopper.update(
            step_log["regret_score"], step_log["mean_reward"], iteration
        )

        real_step = (
            iteration * cfg.training_params.epochs * cfg.training_params.minibatches
        )
        lr = linear_scheduler(real_step)

        if iteration % cfg.training_params.eval_every == 0:
            key, eval_key = jax.random.split(key)
            mean_return = jit_eval(prot, eval_key)
            run.log({"mean_return": mean_return})

        run.log(
            {
                "iteration": iteration,
                "learning_rate": lr,
                "smoothed_regret": stopper.smoothed_regret,
                "smoothed_reward": stopper.smoothed_reward,
                **step_log,
            }
        )
        save_checkpoint(ckptr, iteration, prot, prot_opt)

        iteration += 1
        converged = plateaued or iteration >= cfg.paired_params.max_iterations

    ckptr.wait_until_finished()
    run.finish()


if __name__ == "__main__":
    train()
