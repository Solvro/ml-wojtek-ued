"""Rendered example levels for the paper (MuJoCo, offscreen).

`terrains.pdf`: W01-TEK at the home pose on one level per feature at full
strength (every other feature flat) plus uniform samples at d=0.5 and d=1.
`robots.pdf`: both robots on the same uniform level, each in its own arena
scale, to show that the space is defined relative to robot size.
Run through `paper_figures.py --terrains` (needs MUJOCO_GL=cgl on macOS).
"""

import argparse
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco_playground._src.locomotion.go1 import base as go1_base

from config.hfield_config import HFieldConfig
from hfield.arena import single_tile, single_tile_spec
from hfield.level_space import PARAMS, ParamSpace
from hfield.tiles import TileLayout
from robots.go1 import Go1Terrain, scene_xml
from scripts.brax_ued import ROBOT_SCALE, hfield_config, make_env
from scripts.paper_tables import ROBOT_LABEL

# mujoco is a C extension that ty cannot introspect
mj: Any = mujoco

WIDTH, HEIGHT = 640, 480
SEED_SLOT = 0.37
INDEX = {p.name: i for i, p in enumerate(PARAMS)}
# (panel title, {param: genome value}); every unset param stays flat
FEATURES = (
    ("flat", {}),
    ("rough", {"rough_amp_m": 1.0}),
    ("slope up", {"slope_deg": 1.0}),
    ("slope down", {"slope_deg": 0.0}),
    ("stairs up", {"step_height_m": 1.0}),
    ("stairs down", {"step_height_m": 0.0}),
    ("gaps", {"gap_width_m": 1.0}),
    ("blocks", {"block_height_m": 1.0}),
)


def _args(robot: str) -> argparse.Namespace:
    return argparse.Namespace(
        robot=robot,
        amplitude=0.3,
        coarse_n=13,
        pad_radius=0.55,
        pad_taper=0.2,
        backend="jax",
    )


def _model(robot: str) -> tuple[Any, ParamSpace, HFieldConfig]:
    args = _args(robot)
    config = hfield_config(args)
    env = make_env(args, config, num_envs=1, train=False)
    space = ParamSpace(config, robot_scale=ROBOT_SCALE[robot])
    base: Any = env._env.env
    return render_model(base, config), space, config


def render_model(base: Any, config: HFieldConfig) -> Any:
    """The env's scene with its arena cut to one tile, the level at the
    origin (`single_tile`): the CPU renderer meshes the whole hfield."""
    if isinstance(base, Go1Terrain):
        xml = scene_xml(single_tile(config))
        return mj.MjModel.from_xml_string(xml, assets=go1_base.get_assets())
    return single_tile_spec(Path(base._terrain.files["scene"]), config).compile()


def level_on_single_tile(config: HFieldConfig, hfield: np.ndarray) -> np.ndarray:
    """A rendered level as the `render_model` arena's hfield_data."""
    layout = TileLayout(single_tile(config))
    return np.asarray(layout.assemble(jnp.asarray(hfield)[None]))


def _level(space: ParamSpace, genes: dict[str, float]) -> np.ndarray:
    level = np.append(space.flat.copy(), SEED_SLOT)
    for name, value in genes.items():
        level[INDEX[name]] = value
    return level


def _restyle(model: Any) -> None:
    """One flat terrain colour and a brighter headlight for every robot: the
    Go1 scene ships with no ground material or light."""
    hfield = mj.mjtGeom.mjGEOM_HFIELD
    for i in np.flatnonzero(model.geom_type == hfield):
        model.geom_matid[i] = -1
        model.geom_rgba[i] = (0.55, 0.62, 0.70, 1.0)
    model.vis.headlight.ambient[:] = 0.35
    model.vis.headlight.diffuse[:] = 0.55


def _render(
    model: Any,
    config: HFieldConfig,
    hfield: np.ndarray,
    scale: float,
    elevation: float = -35.0,
) -> np.ndarray:
    model.hfield_data[:] = level_on_single_tile(config, hfield)
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, model.key("home").id)
    mj.mj_forward(model, data)
    camera = mj.MjvCamera()
    camera.lookat[:] = (0.6 * scale, 0.0, 0.0)
    camera.distance = 2.8 * scale
    camera.azimuth = 200.0
    camera.elevation = elevation
    # the renderer uploads the hfield at creation, so build it per level
    with mj.Renderer(model, HEIGHT, WIDTH) as renderer:
        renderer.update_scene(data, camera)
        return renderer.render()


def _grid(images: list[tuple[str, np.ndarray]], cols: int, path: Path) -> None:
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.3 * cols, 1.25 * rows))
    for ax, (title, image) in zip(axes.flat, images, strict=False):
        ax.imshow(image)
        ax.set_title(title, fontsize=8, pad=2)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=1.2)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_terrains(out: Path) -> None:
    rng = np.random.default_rng(3)
    model, space, config = _model("wojtek")
    levels = [(title, _level(space, genes)) for title, genes in FEATURES]
    levels += [
        ("uniform, $d=0.5$", space.scaled(rng, 1, 0.5)[0]),
        ("uniform, $d=1$", space.uniform(rng, 1)[0]),
    ]
    rendered = space.render(np.stack([lv for _, lv in levels]))
    images = [
        (title, _render(model, config, hf, 1.0))
        for (title, _), hf in zip(levels, rendered, strict=True)
    ]
    _grid(images, 5, out / "terrains.pdf")

    shared = space.scaled(rng, 1, 0.6)[0]
    pair: list[tuple[str, Any]] = []
    for robot in ROBOT_SCALE:
        model, space, config = _model(robot)
        _restyle(model)
        hfield = space.render(shared[None])[0]
        pair.append(
            (
                ROBOT_LABEL[robot],
                _render(model, config, hfield, ROBOT_SCALE[robot], -50.0),
            )
        )
    _grid(pair, 2, out / "robots.pdf")
