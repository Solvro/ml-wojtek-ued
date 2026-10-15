from typing import Any

import jax
import numpy as np

from modules.ppo_modules import A2C, Episode
from ued.levels import (
    fill_buffer,
    generate_batch,
    level_to_hfield_data,
    level_to_hfield_data_batch,
    mutate_batch,
    sample_batch_from_buffer,
    sample_from_buffer,
)
from ued.regret import positive_value_loss, positive_value_loss_per_lane
from ued.types import BatchCurator, Curator, PPOTrainer
from utils.ppo_helpers import rollout_metrics, run_ppo_epochs


def mean_episode_reward(episode: Episode) -> float:
    return float(episode.reward.sum(axis=1).mean())


def decide_replay(curator: Curator | BatchCurator) -> bool:
    return bool(curator.buffer.get()) and curator.rng.random() < curator.replay_prob


def terrain_curriculum_step(
    iteration: int,
    curator: BatchCurator,
    *,
    coarse_n_start: int,
    coarse_n_end: int,
    coarse_n_stages: int,
    ramp_iterations: int,
    fill_ratio: float,
) -> dict[str, Any]:
    progress = min(1.0, iteration / max(1, ramp_iterations))
    config = curator.hfield_manager.config
    config.difficulty = progress

    stage = min(coarse_n_stages - 1, int(progress * coarse_n_stages))
    stage_frac = stage / max(1, coarse_n_stages - 1)
    coarse_n = round(coarse_n_start + stage_frac * (coarse_n_end - coarse_n_start))

    if coarse_n != config.coarse_n:
        config.coarse_n = coarse_n
        curator.buffer.clear()
        fill_buffer(curator.buffer, curator.hfield_manager, curator.rng, fill_ratio)

    return {"terrain_difficulty": progress, "terrain_coarse_n": config.coarse_n}


# --- single-level variant (scripts/accel.py) ---------------------------


def score_and_curate(
    key: jax.Array,
    a2c: A2C,
    level: np.ndarray,
    curator: Curator,
    trainer: PPOTrainer,
) -> tuple[float, Episode]:
    hfield_data = level_to_hfield_data(curator.hfield_manager, level)
    episode = trainer.rollout(key, a2c, hfield_data=hfield_data)
    advantages, _ = trainer.advantage(a2c, episode)
    score = float(positive_value_loss(advantages))
    curator.buffer.add(level, score)
    return score, episode


def generate_branch(
    key: jax.Array,
    a2c: A2C,
    curator: Curator,
    trainer: PPOTrainer,
) -> tuple[jax.Array, dict[str, Any]]:
    key, rollout_key = jax.random.split(key)
    level = curator.hfield_manager.generate(curator.rng)
    score, episode = score_and_curate(rollout_key, a2c, level, curator, trainer)

    log = {
        "branch": 0,
        "regret_score": score,
        "mean_reward": mean_episode_reward(episode),
        **rollout_metrics(episode),
    }
    return key, log


def replay_branch(
    key: jax.Array,
    a2c: A2C,
    opt_state: Any,
    curator: Curator,
    trainer: PPOTrainer,
) -> tuple[jax.Array, A2C, Any, dict[str, Any]]:
    key, train_key, update_key, eval_key = jax.random.split(key, 4)
    theta = sample_from_buffer(curator.buffer, curator.rng)
    theta_hfield_data = level_to_hfield_data(curator.hfield_manager, theta)

    episode = trainer.rollout(train_key, a2c, hfield_data=theta_hfield_data)
    advantages, returns = trainer.advantage(a2c, episode)
    a2c, opt_state, loss = run_ppo_epochs(
        trainer.make_step,
        a2c,
        opt_state,
        episode,
        advantages,
        returns,
        update_key,
        epochs=trainer.epochs,
        minibatches=trainer.minibatches,
    )

    theta_prime = curator.hfield_manager.mutate(theta, curator.rng)
    score, _ = score_and_curate(eval_key, a2c, theta_prime, curator, trainer)

    log = {
        "branch": 1,
        "regret_score": score,
        "loss": float(loss),
        "mean_reward": mean_episode_reward(episode),
        **rollout_metrics(episode),
    }
    return key, a2c, opt_state, log


# --- batched-lanes / top-k variant (scripts/accel_top_k.py) ------------


def flatten_lanes(episode: Episode) -> Episode:
    return jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), episode)


def score_and_curate_top_k(
    key: jax.Array,
    a2c: A2C,
    levels: np.ndarray,
    curator: BatchCurator,
    trainer: PPOTrainer,
) -> tuple[np.ndarray, Episode]:
    hfield_data = level_to_hfield_data_batch(curator.hfield_manager, levels)
    episode = flatten_lanes(trainer.rollout(key, a2c, hfield_data=hfield_data))
    advantages, _ = trainer.advantage(a2c, episode)
    scores = np.asarray(
        positive_value_loss_per_lane(advantages.reshape(curator.num_lanes, -1))
    )
    top_idx = np.argsort(scores)[-curator.top_k :]
    for i in top_idx:
        curator.buffer.add(levels[i])
    return scores, episode


def generate_branch_top_k(
    key: jax.Array,
    a2c: A2C,
    curator: BatchCurator,
    trainer: PPOTrainer,
) -> tuple[jax.Array, dict[str, Any]]:
    key, rollout_key = jax.random.split(key)
    levels = generate_batch(curator.hfield_manager, curator.rng, curator.num_lanes)
    scores, episode = score_and_curate_top_k(rollout_key, a2c, levels, curator, trainer)

    log = {
        "branch": 0,
        "regret_score": float(scores.mean()),
        "max_regret_score": float(scores.max()),
        "mean_reward": mean_episode_reward(episode),
        **rollout_metrics(episode),
    }
    return key, log


def replay_branch_top_k(
    key: jax.Array,
    a2c: A2C,
    opt_state: Any,
    curator: BatchCurator,
    trainer: PPOTrainer,
) -> tuple[jax.Array, A2C, Any, dict[str, Any]]:
    key, train_key, update_key, eval_key = jax.random.split(key, 4)
    thetas = sample_batch_from_buffer(curator.buffer, curator.rng, curator.num_lanes)
    hfield_data = level_to_hfield_data_batch(curator.hfield_manager, thetas)

    episode = flatten_lanes(trainer.rollout(train_key, a2c, hfield_data=hfield_data))
    advantages, returns = trainer.advantage(a2c, episode)
    a2c, opt_state, loss = run_ppo_epochs(
        trainer.make_step,
        a2c,
        opt_state,
        episode,
        advantages,
        returns,
        update_key,
        epochs=trainer.epochs,
        minibatches=trainer.minibatches,
    )

    theta_primes = mutate_batch(curator.hfield_manager, thetas, curator.rng)
    scores, _ = score_and_curate_top_k(eval_key, a2c, theta_primes, curator, trainer)

    log = {
        "branch": 1,
        "regret_score": float(scores.mean()),
        "max_regret_score": float(scores.max()),
        "loss": float(loss),
        "mean_reward": mean_episode_reward(episode),
        **rollout_metrics(episode),
    }
    return key, a2c, opt_state, log
