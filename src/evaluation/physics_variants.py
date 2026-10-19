import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from robots.terrain import TerrainRobot

Transform = Callable[[mujoco.MjModel, Any], mujoco.MjModel]

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
# relative to each robot's training contacts: Wojtek 0.001 s (runs at the
# 0.008 s floor) -> 0.016 s, Go1 0.02 s -> 0.04 s
SOFT_TIMECONST_SCALE = 2.0
# contype/conaffinity bit no training geom uses: restored collision
# geoms pair with the terrain and nothing else
TERRAIN_ONLY_BIT = 16


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


def run_timeconst(m: mujoco.MjModel, solref: np.ndarray) -> np.ndarray:
    """Solref time constants as MuJoCo runs them: refsafe raises one below
    2 x timestep to that floor (Wojtek's 0.001 s runs at 0.008 s)."""
    timeconst = solref[..., 0]
    if m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE:
        return timeconst.copy()
    # direct (negative) stiffness/damping is not a time constant
    floor = np.maximum(timeconst, 2 * m.opt.timestep)
    return np.where(timeconst > 0, floor, timeconst)


def half_dt(model: mujoco.MjModel, base: Any) -> mujoco.MjModel:
    m = copy.deepcopy(model)
    # pin time constants at the training floor so halving dt changes only
    # the integration step
    for name in SOLREF_FIELDS:
        solref = getattr(m, name)
        solref[..., 0] = run_timeconst(m, solref)
    m.opt.timestep /= 2
    return m


def soft_contact(model: mujoco.MjModel, base: Any) -> mujoco.MjModel:
    m = copy.deepcopy(model)
    timeconst = run_timeconst(m, m.geom_solref)
    m.geom_solref[:, 0] = np.where(
        timeconst > 0, SOFT_TIMECONST_SCALE * timeconst, timeconst
    )
    return m


def wojtek_full_collision_spec(scene: Path) -> mujoco.MjSpec:
    """Wojtek's per-link collision meshes back on, against the terrain only.
    Recompiled from the spec so the meshes get convex hulls (a runtime
    contype flip has none)."""
    spec = mujoco.MjSpec.from_file(str(scene))
    for geom in spec.geoms:
        mesh = geom.type == mujoco.mjtGeom.mjGEOM_MESH
        if mesh and geom.name.endswith("_collision"):
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
    if not isinstance(base, TerrainRobot):
        scene = Path(base._terrain.files["scene"])
        return copy_tuning(model, wojtek_full_collision_spec(scene).compile())
    if base.FULL_COLLISION is None:
        raise ValueError(
            f"{type(base).__name__} has no full-collision model, so "
            "eval_sim2sim cannot run its full_collision variant; use "
            "robot=wojtek, go1 or spot"
        )
    # Playground's full-collision model (capsule/cylinder links, trunk)
    feet_only, full = base.FULL_COLLISION
    xml = base.scene_xml(base._hfield_config).replace(feet_only, full)
    m = mujoco.MjModel.from_xml_string(xml, assets=base.assets())
    return copy_tuning(model, m)


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
