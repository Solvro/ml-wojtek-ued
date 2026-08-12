from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
from brax.envs.wrappers import training as brax_training

from modules.ppo_modules import Obs
from wojtek_env.envs import registry


class ObsSize(NamedTuple):
    actor: int
    critic: int


class FunctionalEnv:
    def __init__(
        self,
        task: str = "joystick",
        env_overrides: dict | None = None,
        episode_length: int | None = None,
        seed: int = 0,
    ):
        base = registry.make_env(task, env_overrides)
        length = episode_length or base._config.episode_length

        self._env = brax_training.EpisodeWrapper(base, length, action_repeat=1)
        self.action_size = base.action_size

        probe = self._env.reset(jax.random.key(seed))
        self.observation_size = ObsSize(
            actor=int(probe.obs["state"].shape[-1]),
            critic=int(probe.obs["privileged_state"].shape[-1]),
        )

    def reset(self, key: jax.Array) -> tuple[Obs, Any]:
        state: Any = self._env.reset(key)
        state.info["first_data"] = state.data
        state.info["first_obs"] = state.obs
        return self._obs(state), state

    def step(
        self, key: jax.Array, state: Any, action: jax.Array
    ) -> tuple[Obs, Any, jax.Array, jax.Array, dict]:
        del key
        steps = state.info["steps"]
        state.info["steps"] = jnp.where(state.done, jnp.zeros_like(steps), steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = cast(Any, self._env.step(state, action))

        done = state.done
        data = jax.tree.map(
            lambda first, cur: jnp.where(done, first, cur),
            state.info["first_data"],
            state.data,
        )
        obs = jax.tree.map(
            lambda first, cur: jnp.where(done, first, cur),
            state.info["first_obs"],
            state.obs,
        )
        state = state.replace(data=data, obs=obs)
        return self._obs(state), state, state.reward, state.done, state.info

    @staticmethod
    def _obs(state: Any) -> Obs:
        return Obs(actor=state.obs["state"], critic=state.obs["privileged_state"])
