"""Batched Gymnasium surface over a wojtek_env task.

Vmapped through the same brax wrapper stack the original PPO trainer used
(curriculum auto-reset when the task has terrain on, brax's stock wrapper
otherwise). metadata["autoreset_mode"] = SAME_STEP: the observation
returned on a done step is already the next episode's first observation
(brax's AutoResetWrapper / this repo's TerrainAutoResetWrapper substitute it
within the same step() call) -- the true terminal observation is not
separately exposed.

Observation is the actor's flat "state" vector; the critic-only
"privileged_state" rides along under info["privileged_obs"].
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from gymnasium.vector.vector_env import AutoresetMode

from wojtek_env import registry


def _wrap(env, episode_length: int, randomization_fn):
    """The same wrapper stack train.py picked: curriculum auto-reset when
    the env has terrain on, brax's stock wrapper otherwise."""
    if getattr(env, "_terrain_enabled", False):
        from wojtek_env.terrain_wrapper import wrap_for_terrain_brax_training

        return wrap_for_terrain_brax_training(
            env, episode_length=episode_length, randomization_fn=randomization_fn
        )
    from mujoco_playground import wrapper

    return wrapper.wrap_for_brax_training(
        env, episode_length=episode_length, randomization_fn=randomization_fn
    )


class WojtekVectorGymEnv(VectorEnv):
    """Batched Gymnasium VectorEnv over a wojtek_env task, vmapped through
    the same brax wrapper stack the PPO trainer used."""

    metadata: dict[str, Any] = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def __init__(
        self,
        num_envs: int,
        task: str = "joystick",
        env_overrides: dict | None = None,
        episode_length: int | None = None,
        seed: int = 0,
        domain_randomize: bool = False,
    ):
        env_overrides = dict(env_overrides or {})
        env_overrides.setdefault("sim", {})["num_envs"] = num_envs
        self._env = registry.make_env(task, env_overrides)
        length = episode_length or self._env._config.episode_length

        self._key = jax.random.PRNGKey(seed)
        randomization_fn = None
        if domain_randomize:
            from wojtek_env.randomize import make_domain_randomize

            self._key, dr_key = jax.random.split(self._key)
            # One randomized model per env lane, fixed for the process
            # lifetime -- the same convention brax.training.agents.ppo.train
            # uses (functools.partial(randomization_fn, rng=...) before the
            # vmap wrapper sees it).
            randomization_fn = functools.partial(
                make_domain_randomize(self._env.mj_model),
                rng=jax.random.split(dr_key, num_envs),
            )
        wrapped = _wrap(self._env, length, randomization_fn)
        self._reset_fn = jax.jit(wrapped.reset)
        self._step_fn = jax.jit(wrapped.step)

        self.num_envs = num_envs
        probe = self._reset_fn(jax.random.split(self._key, num_envs))
        obs_dim = int(probe.obs["state"].shape[-1])
        act_dim = self._env.action_size

        self.single_observation_space = spaces.Box(
            -np.inf, np.inf, (obs_dim,), np.float32
        )
        self.single_action_space = spaces.Box(-1.0, 1.0, (act_dim,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self._state = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, sub = jax.random.split(self._key)
        self._state = self._reset_fn(jax.random.split(sub, self.num_envs))
        return self._obs(), self._info()

    def step(self, actions):
        self._state = self._step_fn(
            self._state, jp.asarray(np.asarray(actions, dtype=np.float32))
        )
        truncations = np.asarray(self._state.info["truncation"] > 0.5)
        terminations = np.asarray(self._state.done > 0.5) & ~truncations
        rewards = np.asarray(self._state.reward, dtype=np.float32)
        return self._obs(), rewards, terminations, truncations, self._info()

    def _obs(self) -> np.ndarray:
        return np.asarray(self._state.obs["state"], dtype=np.float32)

    def _info(self) -> dict:
        return {
            "privileged_obs": np.asarray(
                self._state.obs["privileged_state"], dtype=np.float32
            )
        }
