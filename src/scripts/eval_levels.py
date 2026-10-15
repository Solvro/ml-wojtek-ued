import functools
from typing import Any, cast

import equinox as eqx
import hydra
import jax
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
from utils.env_overrides import accel_arena_overrides, merge_overrides
from utils.logging import Logger, quiet_dependencies
from utils.ppo_helpers import FALL_CAUSES, LevelEpisode, eval_levels


def rollout_levels(
    jit_eval_levels,
    model: A2C,
    hfield_manager: HFieldManager,
    levels: np.ndarray,
    chunk_size: int,
    key: jax.Array,
) -> tuple[jax.Array, LevelEpisode]:
    n = levels.shape[0]
    num_chunks = -(-n // chunk_size)
    padded = np.resize(levels, (num_chunks * chunk_size, *levels.shape[1:]))

    chunks = []
    for start in range(0, padded.shape[0], chunk_size):
        hfield_data = level_to_hfield_data_batch(
            hfield_manager, padded[start : start + chunk_size]
        )
        key, chunk_key = jax.random.split(key)
        chunks.append(jit_eval_levels(chunk_key, model, hfield_data))

    episodes = jax.tree.map(lambda *xs: np.concatenate(xs)[:n], *chunks)
    return key, episodes


def success(episodes: LevelEpisode, min_travel: float) -> np.ndarray:
    return (np.asarray(episodes.fell) == 0) & (
        np.asarray(episodes.travel) >= min_travel
    )


def summarize(episodes: LevelEpisode, min_travel: float) -> dict[str, float]:
    fell = np.asarray(episodes.fell)
    num_fell = float(fell.sum())
    cause_rates = {
        f"fall_{cause}_rate": (
            float(np.asarray(episodes.causes[cause]).sum() / num_fell)
            if num_fell > 0
            else float("nan")
        )
        for cause in FALL_CAUSES
    }
    return {
        "mean_reward": float(np.mean(episodes.reward)),
        "success_rate": float(success(episodes, min_travel).mean()),
        "fall_rate": float(fell.mean()),
        "mean_travel": float(np.mean(episodes.travel)),
        "mean_cmd_vx": float(np.mean(episodes.cmd_vx)),
        "mean_forward_vel": float(np.mean(episodes.forward_vel)),
        "base_contact_alive_frac": float(np.mean(episodes.base_contact_alive)),
        **cause_rates,
    }


def level_success_corr(first: np.ndarray, second: np.ndarray) -> float:
    if first.std() == 0 or second.std() == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


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

    hfield_config = cast(HFieldConfig, OmegaConf.to_object(cfg.hfield_params))
    hfield_manager = HFieldManager(hfield_config)
    meta_overrides = meta["env_overrides"] or {}
    pad_radius = meta_overrides.get("terrain", {}).get(
        "pad_radius", hfield_config.pad_radius
    )
    env_overrides = merge_overrides(
        meta_overrides,
        accel_arena_overrides(pad_radius),
        # success is judged on falls and travel; the stochastic cut would only
        # add noise and end episodes that have not failed
        {"no_progress": {"enable": False}},
    )
    env = FunctionalEnv(
        task=meta["env_task"],
        env_overrides=merge_overrides(
            env_overrides,
            {"sim": {"num_envs": cfg.chunk_size * cfg.episodes_per_level}},
        ),
        seed=cfg.seed,
    )

    template = A2C(
        hidden_dims=meta["hidden_dims"],
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
    key, first = rollout_levels(
        jit_eval_levels, model, hfield_manager, levels, cfg.chunk_size, key
    )
    # an independent second pass: ranks levels on the first and scores the
    # worst on the second, so CVaR is not inflated by selection noise, and
    # their per-level agreement says whether levels differ in difficulty
    key, second = rollout_levels(
        jit_eval_levels, model, hfield_manager, levels, cfg.chunk_size, key
    )
    first_per_level = success(first, cfg.min_travel).mean(axis=1)
    second_per_level = success(second, cfg.min_travel).mean(axis=1)
    num_worst = max(1, int(np.ceil(cfg.cvar_alpha * cfg.num_levels)))
    worst_idx = np.argsort(first_per_level)[:num_worst]
    worst = jax.tree.map(lambda x: x[worst_idx], second)

    run.log(
        {
            "algo": meta["algo"],
            "run": directory.name,
            "step": step,
            "num_levels": cfg.num_levels,
            "episodes_per_level": cfg.episodes_per_level,
            "difficulty": hfield_config.difficulty,
            "min_travel": cfg.min_travel,
            **summarize(first, cfg.min_travel),
            "cvar_alpha": cfg.cvar_alpha,
            "num_worst_levels": num_worst,
            **{f"cvar_{k}": v for k, v in summarize(worst, cfg.min_travel).items()},
            "level_success_corr": level_success_corr(first_per_level, second_per_level),
            "level_success_std": float(first_per_level.std()),
        }
    )
    run.finish()


if __name__ == "__main__":
    main()
