import functools
from typing import Any, cast

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from omegaconf import OmegaConf

from config.eval_config import EvalLevelsConfig
from config.hfield_config import HFieldConfig
from hfield.manager import HFieldManager
from modules.ppo_modules import A2C
from ued.levels import generate_batch, level_to_hfield_data_batch
from utils.checkpoint import (
    latest_run_dir,
    load_meta,
    open_checkpoint_manager,
    restore_checkpoint,
    run_dir,
)
from utils.env import FunctionalEnv
from utils.logging import Logger, quiet_dependencies
from utils.ppo_helpers import eval_levels


def rollout_levels(
    jit_eval_levels,
    model: A2C,
    hfield_manager: HFieldManager,
    levels: np.ndarray,
    chunk_size: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    n = levels.shape[0]
    num_chunks = -(-n // chunk_size)
    padded = np.resize(levels, (num_chunks * chunk_size, *levels.shape[1:]))

    rewards = []
    terminated = []
    for start in range(0, padded.shape[0], chunk_size):
        hfield_data = level_to_hfield_data_batch(
            hfield_manager, padded[start : start + chunk_size]
        )
        key, chunk_key = jax.random.split(key)
        reward, fell = jit_eval_levels(chunk_key, model, hfield_data)
        rewards.append(reward)
        terminated.append(fell)

    reward = jnp.concatenate(rewards)[:n]
    fell = jnp.concatenate(terminated)[:n]
    return key, reward, fell


@hydra.main(version_base=None, config_name="eval_levels_config")
def main(cfg: EvalLevelsConfig) -> None:
    jax.config.update("jax_platforms", "cuda,cpu")
    quiet_dependencies()

    algo = cfg.algo.value
    name = cfg.wandb_params.name or f"{algo}_eval_levels_run"

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

    hfield_config = HFieldConfig()
    hfield_manager = HFieldManager(hfield_config)
    env_overrides = {
        "terrain": {
            "enable": True,
            "arena": "accel",
            "pad_radius": hfield_config.pad_radius,
        }
    }
    env = FunctionalEnv(
        task=meta["env_task"],
        env_overrides={
            **env_overrides,
            "sim": {"num_envs": cfg.chunk_size * cfg.episodes_per_level},
        },
        seed=cfg.seed,
    )

    template = A2C(
        dense_dim=meta["dense_dim"],
        actor_obs_dim=meta["actor_obs_dim"],
        critic_obs_dim=meta["critic_obs_dim"],
        action_dim=meta["action_dim"],
        action_high=meta["action_high"],
        key=jax.random.key(cfg.seed),
        activation=meta["activation"],
    )
    ckptr = open_checkpoint_manager(directory)
    model, _ = restore_checkpoint(ckptr, cfg.step, template)
    step = cfg.step if cfg.step is not None else ckptr.latest_step()

    jit_eval_levels = eqx.filter_jit(
        functools.partial(
            eval_levels,
            env,
            episodes_per_level=cfg.episodes_per_level,
            steps=cfg.steps,
        )
    )

    rng = np.random.default_rng(cfg.seed)
    key = jax.random.key(cfg.seed)

    levels = generate_batch(hfield_manager, rng, cfg.num_levels)
    key, reward, terminated = rollout_levels(
        jit_eval_levels, model, hfield_manager, levels, cfg.chunk_size, key
    )

    mean_reward = float(reward.mean())
    success_rate = float((1.0 - terminated).mean())

    success_per_level = np.asarray((1.0 - terminated).mean(axis=1))
    num_worst = max(1, int(np.ceil(cfg.cvar_alpha * cfg.num_levels)))
    worst_idx = np.argsort(success_per_level)[:num_worst]
    key, cvar_reward, cvar_terminated = rollout_levels(
        jit_eval_levels, model, hfield_manager, levels[worst_idx], cfg.chunk_size, key
    )
    cvar_mean_reward = float(cvar_reward.mean())
    cvar_success_rate = float((1.0 - cvar_terminated).mean())

    run.log(
        {
            "algo": meta["algo"],
            "run": directory.name,
            "step": step,
            "num_levels": cfg.num_levels,
            "episodes_per_level": cfg.episodes_per_level,
            "mean_reward": mean_reward,
            "success_rate": success_rate,
            "cvar_alpha": cfg.cvar_alpha,
            "num_worst_levels": num_worst,
            "cvar_mean_reward": cvar_mean_reward,
            "cvar_success_rate": cvar_success_rate,
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
