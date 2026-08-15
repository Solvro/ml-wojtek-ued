import contextlib
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
from brax.envs.wrappers import training as brax_training
from mujoco import mjx

from modules.ppo_modules import Obs
from wojtek_env.envs import randomize, registry


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
        domain_randomize: bool = False,
    ):
        base = registry.make_env(task, env_overrides)
        length = episode_length or base._config.episode_length

        self._env = brax_training.EpisodeWrapper(base, length, action_repeat=1)
        self.action_size = base.action_size
        self._domain_randomize = (
            randomize.make_domain_randomize(base.mj_model) if domain_randomize else None
        )

        probe = self._env.reset(jax.random.key(seed))
        self.observation_size = ObsSize(
            actor=int(probe.obs["state"].shape[-1]),
            critic=int(probe.obs["privileged_state"].shape[-1]),
        )

    def sample_dr(
        self, rng: jax.Array, envs_num: int
    ) -> tuple[mjx.Model, mjx.Model] | tuple[None, None]:
        """One randomized physics draw per env, `wojtek_env`'s baseline DR
        (floor friction, base/link mass, gain/kd scale). Returns
        `(dr_model, dr_in_axes)`: `dr_model` is a per-call traced argument for
        `batch_rollout`, `dr_in_axes` is the matching (call-independent)
        `jax.vmap` axis spec -- bind it once via `functools.partial`, not per
        iteration. `(None, None)` when the env was built without DR."""
        if self._domain_randomize is None:
            return None, None
        base = cast(Any, self._env.unwrapped)
        dr_rng = jax.random.split(rng, envs_num)
        return self._domain_randomize(base._mjx_model, dr_rng)

    @contextlib.contextmanager
    def _swapped_model(self, hfield_data: jax.Array | None, dr_model: mjx.Model | None):
        if hfield_data is None and dr_model is None:
            yield
            return
        base = cast(Any, self._env.unwrapped)
        old_model = base._mjx_model
        new_model = dr_model if dr_model is not None else old_model
        if hfield_data is not None:
            new_model = new_model.tree_replace({"hfield_data": hfield_data})
        base._mjx_model = new_model
        try:
            yield
        finally:
            base._mjx_model = old_model

    def reset(
        self,
        key: jax.Array,
        hfield_data: jax.Array | None = None,
        dr_model: mjx.Model | None = None,
    ) -> tuple[Obs, Any]:
        with self._swapped_model(hfield_data, dr_model):
            state: Any = self._env.reset(key)
        state.info["first_data"] = state.data
        state.info["first_obs"] = state.obs
        return self._obs(state), state

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: jax.Array,
        hfield_data: jax.Array | None = None,
        dr_model: mjx.Model | None = None,
    ) -> tuple[Obs, Any, jax.Array, jax.Array, dict]:
        del key
        steps = state.info["steps"]
        state.info["steps"] = jnp.where(state.done, jnp.zeros_like(steps), steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        with self._swapped_model(hfield_data, dr_model):
            state = cast(Any, self._env.step(state, action))

        done = state.done

        def where_done(first, cur):
            d = done
            if d.shape and d.shape[0] != first.shape[0]:
                return cur
            if d.shape:
                d = jnp.reshape(d, [first.shape[0]] + [1] * (len(first.shape) - 1))
            return jnp.where(d, first, cur)

        def reset_leaf(path, first, cur):
            if any(getattr(p, "name", None) == "_impl" for p in path):
                return cur
            return where_done(first, cur)

        data = jax.tree.map_with_path(reset_leaf, state.info["first_data"], state.data)
        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
        state = state.replace(data=data, obs=obs)
        return self._obs(state), state, state.reward, state.done, state.info

    @staticmethod
    def _obs(state: Any) -> Obs:
        return Obs(actor=state.obs["state"], critic=state.obs["privileged_state"])
