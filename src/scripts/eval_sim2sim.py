"""Sim2sim transfer of a `brax_ued.py` checkpoint: MJX-trained policy, physics
stepped by CPU MuJoCo.

Only the simulator changes. The env code (observations, termination,
commands, control latency) is the jax code `eval_brax.py` runs; while the
unroll is traced, `mjx_env.step` and Wojtek's `_step_with_latency` are
swapped for a `jax.pure_callback` that steps one CPU `MjData` per env with
`mj_step` (thread pool, one model copy per thread) and writes back the
integrated state plus the frames, sensors and actuator forces the envs read.
Reset stays on MJX (it only places the robot).

Env, network and level space come from the checkpoint's sidecar
(`<run>.json`, written by `brax_ued.py`); the run must use the `param`
level space.

Physics variants are named transforms of the training `MjModel`
(`VARIANTS`). Every variant runs the same levels with the same reset keys:

- `flat`: `suite_instances` flat levels, as in eval_brax.
- `param/uniform`: the first `num_levels` of eval_brax's 2048 uniform
  levels (same generator and seed, so levels line up with its npz).
- `suite/<terrain>_<k>`: eval_brax's held-out suite at strengths 1-2,
  pooled as `suite`.

Success = no fall and >= `min_travel` x robot scale from spawn after
`level_steps`. Per variant and set: success, fall rate and travel with 95%
bootstrap CIs over levels, and `drop` = `mjx_match` success - variant
success with a paired bootstrap CI.
"""

import argparse
import contextlib
import copy
import functools
import json
import os
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import wandb
from brax.training.agents.ppo import networks as ppo_networks
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base

from config.hfield_config import HFieldConfig
from hfield.level_space import make_level_space
from hfield.tiles import TileLayout
from robots.go1 import Go1Terrain, scene_xml
from scripts.brax_ued import (
    ROBOT_SCALE,
    check_spec_arena,
    load_run,
    make_env,
    make_unroll,
    min_travel,
    spec_network,
)
from scripts.eval_brax import (
    SUITE_STRENGTHS,
    SUITE_TERRAINS,
    bootstrap_ci,
    levels_per_chunk,
    rollout,
    suite_hfield,
)
from wojtek_env.envs.base import WojtekEnv

# eval_brax's default --num_levels; param/uniform is a prefix of its draw
EVAL_BRAX_UNIFORM = 2048
SIM2SIM_STRENGTHS = SUITE_STRENGTHS[:2]

Transform = Callable[[mujoco.MjModel, Any], mujoco.MjModel]


def mjx_match(model: mujoco.MjModel, base: Any) -> mujoco.MjModel:
    return copy.deepcopy(model)


def newton_elliptic(model: mujoco.MjModel, base: Any) -> mujoco.MjModel:
    m = copy.deepcopy(model)
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    # MuJoCo's defaults; training runs 1 (Go1) or 2 (Wojtek) iterations
    m.opt.iterations = 100
    m.opt.ls_iterations = 50
    return m


SOLREF_FIELDS = (
    "geom_solref",
    "eq_solref",
    "jnt_solref",
    "dof_solref",
    "tendon_solref_lim",
    "tendon_solref_fri",
    "pair_solref",
    "pair_solreffriction",
)


def run_timeconst(m: mujoco.MjModel, solref: np.ndarray) -> np.ndarray:
    """Solref time constants as MuJoCo runs them: refsafe raises one below
    2 x timestep to that floor (Wojtek's 0.001 s runs at 0.008 s)."""
    timeconst = solref[..., 0]
    if m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE:
        return timeconst.copy()
    # direct (negative) stiffness/damping is not a time constant
    return np.where(timeconst > 0, np.maximum(timeconst, 2 * m.opt.timestep), timeconst)


def half_dt(model: mujoco.MjModel, base: Any) -> mujoco.MjModel:
    m = copy.deepcopy(model)
    # pin time constants at the training floor so halving dt changes only
    # the integration step
    for name in SOLREF_FIELDS:
        solref = getattr(m, name)
        solref[..., 0] = run_timeconst(m, solref)
    m.opt.timestep /= 2
    return m


# relative to each robot's training contacts: Wojtek 0.001 s (runs at the
# 0.008 s floor) -> 0.016 s, Go1 0.02 s -> 0.04 s
SOFT_TIMECONST_SCALE = 2.0


def soft_contact(model: mujoco.MjModel, base: Any) -> mujoco.MjModel:
    m = copy.deepcopy(model)
    timeconst = run_timeconst(m, m.geom_solref)
    m.geom_solref[:, 0] = np.where(
        timeconst > 0, SOFT_TIMECONST_SCALE * timeconst, timeconst
    )
    return m


# contype/conaffinity bit no training geom uses: restored collision
# geoms pair with the terrain and nothing else
TERRAIN_ONLY_BIT = 16


def wojtek_full_collision_spec(scene: Path) -> mujoco.MjSpec:
    """Wojtek's per-link collision meshes back on, against the terrain.

    `build_model` turns every mesh off and collides through foot spheres,
    base boxes and floor-only leg capsules. Recompiled from the spec so
    the meshes get convex hulls (a runtime contype flip has none).
    """
    spec = mujoco.MjSpec.from_file(str(scene))
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.name.endswith("_collision"):
            geom.contype, geom.conaffinity = TERRAIN_ONLY_BIT, 0
    for geom in spec.worldbody.geoms:
        if geom.contype or geom.conaffinity:
            geom.conaffinity |= TERRAIN_ONLY_BIT
    return spec


def copy_tuning(train: mujoco.MjModel, m: mujoco.MjModel) -> mujoco.MjModel:
    """The env constructors' edits to a freshly loaded model."""
    m.opt.timestep = train.opt.timestep
    m.opt.ccd_iterations = train.opt.ccd_iterations
    for name in (
        "dof_damping",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_ctrlrange",
    ):
        getattr(m, name)[:] = getattr(train, name)
    return m


def full_collision(model: mujoco.MjModel, base: Any) -> mujoco.MjModel:
    if isinstance(base, Go1Terrain):
        # Playground's full-collision Go1 (capsule/cylinder links, trunk)
        xml = scene_xml(base._hfield_config).replace(
            "go1_mjx_feetonly.xml", "go1_mjx_fullcollisions.xml"
        )
        m = mujoco.MjModel.from_xml_string(xml, assets=go1_base.get_assets())
        return copy_tuning(model, m)
    scene = Path(base._terrain.files["scene"])
    return copy_tuning(model, wojtek_full_collision_spec(scene).compile())


# name -> (what it tests, transform of the training MjModel)
VARIANTS: dict[str, tuple[str, Transform]] = {
    "mjx_match": (
        "training model and options; CPU MuJoCo vs MJX implementation only",
        mjx_match,
    ),
    "newton_elliptic": (
        "converged solver: Newton, elliptic cones, 100/50 iterations",
        newton_elliptic,
    ),
    "half_dt": (
        "timestep halved, two CPU steps per training substep",
        half_dt,
    ),
    "full_collision": (
        "link collision geometry the MJX model drops",
        full_collision,
    ),
    "soft_contact": (
        f"contact solref time constant x{SOFT_TIMECONST_SCALE} of training",
        soft_contact,
    ),
}

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


UNSTABLE = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
)


def unstable_warnings(d: mujoco.MjData) -> int:
    return sum(d.warning[w].number for w in UNSTABLE)


def worker_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


class CpuPhysics:
    """Steps a batch of envs on CPU MuJoCo, one model and data per thread.

    Every env stands on its own tile of one shared arena: `set_arena` takes
    the next unroll's arena `hfield_data`, and each thread copies it into
    its model once per arena.
    """

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
        for size in SAME_SIZES:
            if getattr(model, size) != getattr(self.train_model, size):
                raise ValueError(
                    f"variant model changes {size} "
                    f"({getattr(self.train_model, size)} -> "
                    f"{getattr(model, size)}); state cannot transfer"
                )
        ratio = self.train_model.opt.timestep / model.opt.timestep
        if not np.isclose(ratio, round(ratio)):
            raise ValueError(
                f"variant timestep {model.opt.timestep} must divide the "
                f"training timestep {self.train_model.opt.timestep}"
            )
        self.substeps = round(ratio)
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
                if not self.step_env(m, d, n_substeps, [x[i] for x in arrays]):
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
        return host_step(physics, data, action, action, jnp.zeros(()), n_substeps)

    def step_with_latency(
        env: Any, data: mjx.Data, prev: Any, new: Any, d: Any
    ) -> mjx.Data:
        return host_step(physics, data, prev, new, d, env.n_substeps)

    def unpatched(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(
            "an env physics path not redirected by eval_sim2sim.cpu_physics "
            "called mjx.step; add it there"
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


def level_sets(
    args: argparse.Namespace, spec: dict[str, Any], config: HFieldConfig
) -> dict[str, Any]:
    """Rendered hfields per set, the levels eval_brax builds for them."""
    scale = ROBOT_SCALE[spec["robot"]]
    level_space = spec["level_space"]
    space = make_level_space(
        level_space, config, n_edits=1, edit_scale=0.0, robot_scale=scale
    )
    flat = space.scaled(np.random.default_rng(args.seed), args.suite_instances, 0.0)
    uniform = space.uniform(np.random.default_rng(args.seed + 1), EVAL_BRAX_UNIFORM)
    sets = {
        "flat": space.render(flat),
        f"{level_space}/uniform": space.render(uniform[: args.num_levels]),
    }
    # eval_brax's suite stream: only blocks draws, k = 1..4 in order
    rng = np.random.default_rng(args.seed)
    for terrain in SUITE_TERRAINS:
        for k in SIM2SIM_STRENGTHS:
            sets[f"suite/{terrain}_{k}"] = np.stack(
                [
                    suite_hfield(config, terrain, k, rng, scale)
                    for _ in range(args.suite_instances)
                ]
            )
    return sets


def with_ci(
    out: dict[str, float], tag: str, per_level: np.ndarray, rng: np.random.Generator
) -> None:
    out[tag] = float(per_level.mean())
    lo, hi = bootstrap_ci(lambda i: float(per_level[i].mean()), len(per_level), rng)
    out[f"{tag}_ci_low"], out[f"{tag}_ci_high"] = lo, hi


def set_metrics(
    prefix: str,
    fell: np.ndarray,
    travel: np.ndarray,
    threshold: float,
    reference: np.ndarray | None,
) -> dict[str, float]:
    """fell / travel: (levels, episodes). reference: mjx_match success."""
    rng = np.random.default_rng(0)
    success = ((fell == 0) & (travel >= threshold)).mean(axis=1)
    out: dict[str, float] = {}
    with_ci(out, f"{prefix}/success_rate", success, rng)
    with_ci(out, f"{prefix}/fall_rate", fell.mean(axis=1), rng)
    with_ci(out, f"{prefix}/mean_travel", travel.mean(axis=1), rng)
    if reference is not None:
        with_ci(out, f"{prefix}/drop", reference - success, rng)
    return out


Results = dict[str, dict[str, tuple[np.ndarray, np.ndarray, int]]]


def pool_suite(
    sets: dict[str, tuple[np.ndarray, np.ndarray, int]],
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    pooled = {k: v for k, v in sets.items() if not k.startswith("suite/")}
    cells = [v for k, v in sets.items() if k.startswith("suite/")]
    pooled["suite"] = (
        np.concatenate([c[0] for c in cells]),
        np.concatenate([c[1] for c in cells]),
        sum(c[2] for c in cells),
    )
    return pooled


def summarize(results: Results, threshold: float) -> dict[str, float]:
    """Per variant x set metrics; `drop` against `mjx_match` on each set."""
    metrics: dict[str, float] = {}
    reference: dict[str, np.ndarray] = {}
    for variant, sets in results.items():
        for name, (fell, travel, diverged) in pool_suite(sets).items():
            prefix = f"{variant}/{name}"
            success = ((fell == 0) & (travel >= threshold)).mean(axis=1)
            if variant == "mjx_match":
                reference[name] = success
            ref = None if variant == "mjx_match" else reference[name]
            metrics.update(set_metrics(prefix, fell, travel, threshold, ref))
            metrics[f"{prefix}/diverged_per_episode"] = diverged / fell.size
        for name, (fell, travel, _) in sets.items():
            if name.startswith("suite/"):
                success = ((fell == 0) & (travel >= threshold)).mean()
                metrics[f"{variant}/{name}/success_rate"] = float(success)
    return metrics


def evaluate(
    ctx: dict[str, Any], sets: dict[str, np.ndarray]
) -> tuple[dict[str, float], dict[str, Any]]:
    args, base, physics = ctx["args"], ctx["base"], ctx["physics"]
    results: Results = {}
    raw: dict[str, Any] = {}
    elapsed: dict[str, float] = {}
    for variant, (tests, transform) in VARIANTS.items():
        print(f"{variant}: {tests}", flush=True)
        start = time.time()
        physics.use(transform(base.mj_model, base))
        results[variant] = {}
        for i, (name, hfields) in enumerate(sets.items()):
            # same reset keys for every variant: paired comparisons
            key = jax.random.fold_in(jax.random.key(args.seed), i)
            episodes = rollout(ctx, hfields, key)
            fell, travel = episodes["fell"], episodes["travel"]
            results[variant][name] = (fell, travel, physics.take_diverged())
            tag = f"{variant}_{name.replace('/', '_')}"
            raw[f"{tag}_fell"], raw[f"{tag}_travel"] = fell, travel
        elapsed[variant] = time.time() - start
        print(f"{variant}: {elapsed[variant]:.0f} s", flush=True)
    metrics = summarize(results, min_travel(ctx["spec"]["robot"], args.min_travel))
    for variant, seconds in elapsed.items():
        metrics[f"{variant}/elapsed_s"] = seconds
    return metrics, raw


def build(args: argparse.Namespace) -> dict[str, Any]:
    spec, params = load_run(Path(args.checkpoint))
    if spec["level_space"] != "param":
        raise ValueError(
            f"{args.checkpoint} trained on the {spec['level_space']!r} level "
            "space; sim2sim reuses eval_brax's param-space levels only"
        )
    check_spec_arena(spec)
    config = HFieldConfig(**spec["hfield"])
    layout = TileLayout(config)
    env = make_env(spec, levels_per_chunk(args, layout) * args.episodes, train=False)
    base = cast(Any, env)._env.unwrapped
    physics = CpuPhysics(base.mj_model, worker_count())
    make_policy = ppo_networks.make_inference_fn(spec_network(spec))
    unroll = make_unroll(env, make_policy, None)

    def cpu_unroll(
        params: Any,
        state: Any,
        key: jax.Array,
        hfield: jax.Array,
        dr_model: None,
        deterministic: bool,
        length: int,
    ) -> Any:
        physics.set_arena(np.asarray(hfield))
        with cpu_physics(physics):
            return unroll(
                params,
                state,
                key,
                hfield,
                dr_model,
                deterministic=deterministic,
                length=length,
            )

    return {
        "args": args,
        "spec": spec,
        "config": config,
        "layout": layout,
        "assemble": jax.jit(layout.assemble),
        "base": base,
        "physics": physics,
        "params": params,
        "reset": jax.jit(
            jax.vmap(lambda k, h, t: env.reset(k, h, tile_xy=t), in_axes=(0, None, 0))
        ),
        "unroll": cpu_unroll,
        "dt": base.dt,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", required=True, help="brax_ued.py .pkl, sidecar .json beside it"
    )
    p.add_argument("--name", required=True, help="wandb run name")
    p.add_argument("--entity", default="")
    p.add_argument("--project", default="wojtyla")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_levels", type=int, default=256)
    p.add_argument("--episodes", type=int, default=4, help="per level")
    p.add_argument("--suite_instances", type=int, default=64)
    p.add_argument("--chunk_envs", type=int, default=1024)
    p.add_argument("--level_steps", type=int, default=300)
    p.add_argument("--min_travel", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    ctx = build(args)
    spec = ctx["spec"]
    run = wandb.init(
        entity=args.entity or None,
        project=args.project,
        name=args.name,
        tags=["sim2sim", spec["robot"]],
        config={
            **vars(args),
            "robot": spec["robot"],
            "level_space": spec["level_space"],
            "run": spec,
        },
    )
    metrics, raw = evaluate(ctx, level_sets(args, spec, ctx["config"]))
    run.summary.update(metrics)
    run.finish()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / f"{args.name}.npz", **raw)


if __name__ == "__main__":
    main()
