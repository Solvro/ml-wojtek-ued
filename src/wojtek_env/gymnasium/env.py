from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jp
import numpy as np
from brax.envs.wrappers import training as brax_training
from gymnasium import spaces

from wojtek_env.envs import registry


class WojtekGymEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        task: str = "joystick",
        env_overrides: dict | None = None,
        episode_length: int | None = None,
        seed: int = 0,
    ):
        self._env = registry.make_env(task, env_overrides)
        length = episode_length or self._env._config.episode_length
        wrapped = brax_training.EpisodeWrapper(
            self._env, length, action_repeat=1)
        self._reset_fn = jax.jit(wrapped.reset)
        self._step_fn = jax.jit(wrapped.step)

        self._key = jax.random.PRNGKey(seed)
        probe = self._reset_fn(self._key)
        obs_dim = int(probe.obs["state"].shape[-1])
        act_dim = self._env.action_size

        self.observation_space = spaces.Box(-np.inf,
                                            np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (act_dim,), np.float32)
        self._state = None
        self._needs_reset = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._key = (
            jax.random.PRNGKey(seed)
            if seed is not None
            else (jax.random.split(self._key)[0])
        )
        self._state = self._reset_fn(self._key)
        self._needs_reset = False
        return self._obs(), self._info()

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError(
                "step() called after termination/truncation; call reset() first"
            )
        self._state = self._step_fn(
            self._state, jp.asarray(np.asarray(action, dtype=np.float32))
        )
        truncated = bool(self._state.info["truncation"] > 0.5)
        terminated = bool(self._state.done > 0.5) and not truncated
        self._needs_reset = terminated or truncated
        return (
            self._obs(),
            float(self._state.reward),
            terminated,
            truncated,
            self._info(),
        )

    def _obs(self) -> np.ndarray:
        return np.asarray(self._state.obs["state"], dtype=np.float32)

    def _info(self) -> dict:
        return {
            "privileged_obs": np.asarray(
                self._state.obs["privileged_state"], dtype=np.float32
            )
        }
