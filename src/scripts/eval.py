from typing import Any, cast

import equinox as eqx
import hydra
import jax
import wandb
from omegaconf import OmegaConf

from config.eval_config import EvalConfig
from modules.ppo_modules import A2C
from utils.checkpoint import (
    latest_run_dir,
    load_meta,
    open_checkpoint_manager,
    restore_checkpoint,
    run_dir,
)
from utils.env import FunctionalEnv
from utils.logging import Logger, quiet_dependencies
from utils.ppo_helpers import eval as eval_agent


@hydra.main(version_base=None, config_name="eval_config")
def main(cfg: EvalConfig) -> None:
    jax.config.update("jax_platforms", "cuda,cpu")
    quiet_dependencies()

    algo = cfg.algo.value
    name = cfg.wandb_params.name or f"{algo}_eval_run"

    run = (
        wandb.init(
            entity=cfg.wandb_params.entity,
            project=cfg.wandb_params.project,
            tags=cfg.wandb_params.tags,
            name=name,
            config=cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True)),
        )
        if cfg.wandb_params.enable
        else Logger(name=name)
    )

    directory = (
        run_dir(cfg.artifacts_dir, algo, cfg.run)
        if cfg.run
        else latest_run_dir(cfg.artifacts_dir, algo)
    )
    meta = load_meta(directory)

    env = FunctionalEnv(
        task=meta["env_task"],
        env_overrides=meta["env_overrides"],
        seed=meta["env_seed"],
    )

    template_key = jax.random.key(cfg.seed)
    template = A2C(
        dense_dim=meta["dense_dim"],
        actor_obs_dim=meta["actor_obs_dim"],
        critic_obs_dim=meta["critic_obs_dim"],
        action_dim=meta["action_dim"],
        action_high=meta["action_high"],
        key=template_key,
        activation=meta["activation"],
    )

    ckptr = open_checkpoint_manager(directory)
    model, _ = restore_checkpoint(ckptr, cfg.step, template)
    step = cfg.step if cfg.step is not None else ckptr.latest_step()

    jit_eval = eqx.filter_jit(eval_agent)
    mean_return = jit_eval(
        env, model, jax.random.key(cfg.seed), envs_num=cfg.envs_num, steps=cfg.steps
    )

    run.log(
        {
            "algo": meta["algo"],
            "run": directory.name,
            "step": step,
            "mean_return": float(mean_return),
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
