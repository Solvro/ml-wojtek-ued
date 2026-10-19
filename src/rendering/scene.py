from pathlib import Path
from typing import Any

import jax.numpy as jnp
import mujoco
import numpy as np

from config.hfield_config import HFieldConfig
from hfield.arena import single_tile, single_tile_spec
from hfield.tiles import TileLayout
from robots.terrain import TerrainRobot

# mujoco is a C extension that ty cannot introspect
mj: Any = mujoco


def render_model(base: Any, config: HFieldConfig) -> Any:
    """The env's scene with its arena cut to one tile, the level at the
    origin: the renderer meshes the whole hfield."""
    if isinstance(base, TerrainRobot):
        xml = base.scene_xml(single_tile(config))
        return mj.MjModel.from_xml_string(xml, assets=base.assets())
    scene = Path(base._terrain.files["scene"])
    return single_tile_spec(scene, config).compile()


def level_on_single_tile(
    config: HFieldConfig, hfield: np.ndarray
) -> np.ndarray:
    """A rendered level as the `render_model` arena's hfield_data."""
    layout = TileLayout(single_tile(config))
    return np.asarray(layout.assemble(jnp.asarray(hfield)[None]))


def tracking_camera(model: Any, scale: float) -> Any:
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_TRACKING
    free = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)[0]
    camera.trackbodyid = model.jnt_bodyid[free]
    camera.distance = 1.6 * scale
    camera.azimuth = 135.0
    camera.elevation = -30.0
    return camera
