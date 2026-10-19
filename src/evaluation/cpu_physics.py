import contextlib
import copy
import functools
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from wojtek_env.envs.base import WojtekEnv

# MjData fields written back into mjx.Data after each physics call: the
# integrated state plus every derived field the Wojtek / Go1 envs read
OUT_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "time",
    "ctrl",
    "qacc",
    "actuator_force",
    "qfrc_actuator",
    "xpos",
    "xquat",
    "xmat",
    "xanchor",
    "xaxis",
    "site_xpos",
    "site_xmat",
    "subtree_com",
    "cvel",
    "sensordata",
)
# computed from body frames with the training model's geom offsets, so a
# variant may add or drop geoms (Go1 full collision)
GEOM_FIELDS = ("geom_xpos", "geom_xmat")
SAME_SIZES = ("nq", "nv", "nu", "na", "nbody", "njnt", "nsite")
# base height (free joint qpos[2]) given to an env whose solver diverged
DIVERGED_Z = -100.0
UNSTABLE = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
)


def sensor_index(train: mujoco.MjModel, m: mujoco.MjModel) -> np.ndarray:
    """Variant sensordata indices that rebuild the training layout."""
    index = []
    for i in range(train.nsensor):
        name = train.sensor(i).name
        try:
            j = m.sensor(name).id
        except KeyError as e:
            raise ValueError(
                f"variant model lacks training sensor {name!r}; the env "
                "reads sensordata by the training model's addresses"
            ) from e
        if train.sensor_dim[i] != m.sensor_dim[j]:
            raise ValueError(f"sensor {name!r} changed dimension")
        adr = m.sensor_adr[j]
        index.extend(range(adr, adr + m.sensor_dim[j]))
    return np.asarray(index, dtype=int)


def geom_rotations(m: mujoco.MjModel) -> np.ndarray:
    mats = np.zeros((m.ngeom, 9))
    for g in range(m.ngeom):
        mujoco.mju_quat2Mat(mats[g], m.geom_quat[g])
    return mats.reshape(m.ngeom, 3, 3)


def unstable_warnings(d: mujoco.MjData) -> int:
    return sum(d.warning[w].number for w in UNSTABLE)


def worker_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def check_transferable(train: mujoco.MjModel, model: mujoco.MjModel) -> int:
    """CPU substeps per training substep; fails if state cannot transfer."""
    for size in SAME_SIZES:
        if getattr(model, size) != getattr(train, size):
            raise ValueError(
                f"variant model changes {size} "
                f"({getattr(train, size)} -> "
                f"{getattr(model, size)}); state cannot transfer"
            )
    ratio = train.opt.timestep / model.opt.timestep
    if not np.isclose(ratio, round(ratio)):
        raise ValueError(
            f"variant timestep {model.opt.timestep} must divide the "
            f"training timestep {train.opt.timestep}"
        )
    return round(ratio)


class CpuPhysics:
    """Steps a batch of envs on CPU MuJoCo, one model and data per thread;
    `set_arena` takes the next unroll's shared arena `hfield_data`."""

    def __init__(self, train_model: mujoco.MjModel, threads: int):
        self.train_model = train_model
        self.threads = threads
        self.pool = ThreadPoolExecutor(threads)
        self.geom_body = train_model.geom_bodyid.copy()
        self.geom_pos = train_model.geom_pos.copy()
        self.geom_rot = geom_rotations(train_model)
        self.arena_id = 0
        self.use(train_model)
        self.set_arena(np.zeros(train_model.nhfielddata, np.float32))

    def use(self, model: mujoco.MjModel) -> None:
        self.substeps = check_transferable(self.train_model, model)
        self.models = [copy.deepcopy(model) for _ in range(self.threads)]
        self.datas = [mujoco.MjData(m) for m in self.models]
        self.sensors = sensor_index(self.train_model, model)
        self.diverged = [0] * self.threads
        self.loaded = [-1] * self.threads

    def set_arena(self, hfield: np.ndarray) -> None:
        self.hfield = np.asarray(hfield, np.float32)
        if self.hfield.shape != (self.train_model.nhfielddata,):
            raise ValueError(
                f"arena hfield_data of shape {self.hfield.shape}, the model "
                f"holds {self.train_model.nhfielddata} nodes; pass the "
                "TileLayout.assemble output of this model's arena"
            )
        self.arena_id += 1

    def step_env(
        self,
        m: mujoco.MjModel,
        d: mujoco.MjData,
        n_substeps: int,
        inputs: list[np.ndarray],
    ) -> bool:
        """One env's control period; False if the solver diverged."""
        qpos, qvel, act, warm, t, qfrc, xfrc, prev, new, delay = inputs
        d.qpos[:], d.qvel[:], d.act[:] = qpos, qvel, act
        d.qacc_warmstart[:], d.time = warm, float(t)
        d.qfrc_applied[:], d.xfrc_applied[:] = qfrc, xfrc
        warnings = unstable_warnings(d)
        held = int(delay) * self.substeps
        rest = n_substeps * self.substeps - held
        if held:
            d.ctrl[:] = prev
            mujoco.mj_step(m, d, nstep=held)
        if rest:
            d.ctrl[:] = new
            mujoco.mj_step(m, d, nstep=rest)
        finite = np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
        if unstable_warnings(d) == warnings and finite:
            return True
        # MuJoCo resets (Wojtek) or NaNs out (Go1) a diverged env; drop the
        # base far below the terrain instead so both envs score a fall
        d.qpos[:], d.qvel[:] = qpos, 0.0
        d.qpos[2] = DIVERGED_Z
        mujoco.mj_forward(m, d)
        return False

    def take_diverged(self) -> int:
        count = sum(self.diverged)
        self.diverged = [0] * self.threads
        return count

    def step(
        self,
        n_substeps: int,
        specs: dict[str, jax.ShapeDtypeStruct],
        *inputs: np.ndarray,
    ) -> dict[str, np.ndarray]:
        # vmapped callbacks receive jax Arrays; indexing those per env is a
        # jax dispatch holding the GIL, which serialized the threads
        arrays = [np.asarray(x) for x in inputs]
        n = len(arrays[0])
        out = {f: np.empty((n, *specs[f].shape), np.float64) for f in OUT_FIELDS}
        bounds = np.linspace(0, n, self.threads + 1).astype(int)

        def work(thread: int) -> None:
            m, d = self.models[thread], self.datas[thread]
            if self.loaded[thread] != self.arena_id:
                m.hfield_data[:] = self.hfield
                self.loaded[thread] = self.arena_id
            for i in range(bounds[thread], bounds[thread + 1]):
                env_inputs = [x[i] for x in arrays]
                if not self.step_env(m, d, n_substeps, env_inputs):
                    self.diverged[thread] += 1
                for f in OUT_FIELDS:
                    value = getattr(d, f)
                    if f == "sensordata":
                        value = value[self.sensors]
                    out[f][i] = np.reshape(value, out[f].shape[1:])

        list(self.pool.map(work, range(self.threads)))
        body_rot = out["xmat"][:, self.geom_body]
        out["geom_xpos"] = out["xpos"][:, self.geom_body] + np.einsum(
            "bgij,gj->bgi", body_rot, self.geom_pos
        )
        out["geom_xmat"] = body_rot @ self.geom_rot
        return {f: out[f].astype(specs[f].dtype) for f in specs}


def host_step(
    physics: CpuPhysics,
    data: mjx.Data,
    prev: jax.Array,
    new: jax.Array,
    delay: jax.Array,
    *,
    n_substeps: int,
) -> mjx.Data:
    specs = {
        f: jax.ShapeDtypeStruct(getattr(data, f).shape, getattr(data, f).dtype)
        for f in (*OUT_FIELDS, *GEOM_FIELDS)
    }
    out = jax.pure_callback(
        functools.partial(physics.step, n_substeps, specs),
        specs,
        data.qpos,
        data.qvel,
        data.act,
        data.qacc_warmstart,
        data.time,
        data.qfrc_applied,
        data.xfrc_applied,
        prev,
        new,
        jnp.asarray(delay, jnp.int32),
        vmap_method="broadcast_all",
    )
    return data.replace(**out)


@contextlib.contextmanager
def cpu_physics(physics: CpuPhysics) -> Iterator[None]:
    """Redirect the envs' physics calls to `physics` while tracing."""

    def step(
        model: Any, data: mjx.Data, action: jax.Array, n_substeps: int = 1
    ) -> mjx.Data:
        del model
        no_delay = jnp.zeros(())
        return host_step(physics, data, action, action, no_delay, n_substeps=n_substeps)

    def step_with_latency(
        env: Any, data: mjx.Data, prev: Any, new: Any, d: Any
    ) -> mjx.Data:
        return host_step(physics, data, prev, new, d, n_substeps=env.n_substeps)

    def unpatched(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "an env physics path not redirected by "
            "evaluation.cpu_physics.cpu_physics called mjx.step; add it there"
        )

    modules = cast(Any, (mjx_env, WojtekEnv, mjx))
    saved = (mjx_env.step, WojtekEnv._step_with_latency, mjx.step)
    modules[0].step, modules[1]._step_with_latency, modules[2].step = (
        step,
        step_with_latency,
        unpatched,
    )
    try:
        yield
    finally:
        modules[0].step, modules[1]._step_with_latency, modules[2].step = saved
