"""Rendered example levels (`terrains.pdf`, `robots.pdf`) for the paper."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from config.hfield_config import HFieldConfig
from config.schema import Backend, Robot
from env.factory import make_env
from hfield.level_space import PARAMS, ParamSpace
from paper.labels import ROBOT_LABEL
from rendering.scene import level_on_single_tile, render_model
from robots.scale import (
    ROBOT_SCALE,
    default_robot,
    hfield_config,
    robot_scale,
)

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


def _model(
    robot: Robot, arena: HFieldConfig
) -> tuple[Any, ParamSpace, HFieldConfig]:
    robot_cfg = default_robot(robot)
    config = hfield_config(robot_cfg, arena)
    env = make_env(robot_cfg, Backend.jax, config, num_envs=1, train=False)
    space = ParamSpace(config, robot_scale=robot_scale(robot_cfg))
    base: Any = env._env.env
    return render_model(base, config), space, config


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


def plot_terrains(out: Path, arena: HFieldConfig) -> None:
    rng = np.random.default_rng(3)
    model, space, config = _model(Robot.wojtek, arena)
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
    for robot in Robot:
        model, space, config = _model(robot, arena)
        _restyle(model)
        hfield = space.render(shared[None])[0]
        pair.append(
            (
                ROBOT_LABEL[robot],
                _render(model, config, hfield, ROBOT_SCALE[robot], -50.0),
            )
        )
    _grid(pair, 2, out / "robots.pdf")
