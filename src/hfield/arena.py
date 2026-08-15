from __future__ import annotations

import json
import os
import re

import numpy as np

from config.hfield_config import HFieldConfig
from wojtek_env import paths
from wojtek_env.terrain import terrain

_APRON_EXTENT = 25.0
_APRON_HALF_Z = 0.5


def _robot_meshdir():
    match = re.search(r'meshdir="([^"]+)"', paths.ROBOT_XML.read_text())
    rel = match.group(1) if match else "."
    return (paths.MUJOCO_DIR / rel).resolve()


def _apron_geoms(radius: float) -> list[str]:
    ox = radius + _APRON_EXTENT
    strips = (
        ("accel_apron_north", 0.0, radius + _APRON_EXTENT / 2, ox, _APRON_EXTENT / 2),
        ("accel_apron_south", 0.0, -radius - _APRON_EXTENT / 2, ox, _APRON_EXTENT / 2),
        (
            "accel_apron_west",
            -radius - _APRON_EXTENT / 2,
            0.0,
            _APRON_EXTENT / 2,
            radius,
        ),
        (
            "accel_apron_east",
            radius + _APRON_EXTENT / 2,
            0.0,
            _APRON_EXTENT / 2,
            radius,
        ),
    )
    return [
        f'    <geom name="{name}" type="box" size="{hx:.4f} {hy:.4f} {_APRON_HALF_Z}" '
        f'pos="{cx:.4f} {cy:.4f} {-_APRON_HALF_Z}" material="groundplane" '
        f'condim="3" conaffinity="15"/>'
        for name, cx, cy, hx, hy in strips
    ]


def build_accel_scene_xml(config: HFieldConfig, hfield_file: str) -> str:
    size = (
        f"{config.patch_radius:.4f} {config.patch_radius:.4f} "
        f"{2 * config.amplitude:.6f} {config.base_z:.4f}"
    )
    apron = "\n".join(_apron_geoms(config.patch_radius))
    return f"""<mujoco model="wojtek_accel_scene">
  <include file="wojtek_mjx.xml"/>
  <statistic center="0 0 0.15" extent="0.8"/>
  <visual>
    <headlight diffuse="0.3 0.3 0.3" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
    <hfield name="terrain" file="{hfield_file}" size="{size}"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" castshadow="true" pos="0 0 20" dir="0.25 0.35 -0.9" diffuse="0.7 0.7 0.7" specular="0 0 0"/>
    <geom name="terrain_hfield" type="hfield" hfield="terrain" material="groundplane" pos="0 0 {-config.amplitude:.6f}" condim="3" conaffinity="15"/>
{apron}
    <camera name="track" mode="trackcom" pos="0.9 -1.3 0.5" xyaxes="0.83 0.55 0 -0.15 0.23 0.96"/>
  </worldbody>
</mujoco>
"""


def build_accel_arena(config: HFieldConfig, seed: int = 0) -> terrain.Arena:
    grid = config.grid_size
    radius = config.patch_radius
    hfield_data = np.full((grid, grid), 0.5, dtype=np.float32)
    lookup = np.zeros((grid, grid), dtype=np.float32)
    tiles = tuple(
        terrain.TileSpec(
            row=0,
            col=j,
            terrain_type=ttype,
            difficulty=0.0,
            origin=(0.0, 0.0, 0.0),
            pad_radius=config.pad_radius,
            pad_height=0.0,
        )
        for j, ttype in enumerate(terrain.TYPES)
    )
    spec = terrain.TerrainSpec(
        seed=seed,
        n_rows=1,
        n_cols=len(terrain.TYPES),
        tile_size=2 * radius,
        border=0.0,
        cell_size=config.cell_size,
        ordered=True,
        pad_radius=config.pad_radius,
        stair_platform_half=terrain.STAIR_PLATFORM_HALF,
        n_steps=terrain.N_STEPS,
        types=terrain.TYPES,
        x_min=-radius,
        x_max=radius,
        y_min=-radius,
        y_max=radius,
        hfield=terrain.HFieldSpec(
            nrow=grid,
            ncol=grid,
            radius_x=radius,
            radius_y=radius,
            elevation_z=2 * config.amplitude,
            base_z=config.base_z,
            pos_z=-config.amplitude,
        ),
        tiles=tiles,
    )
    return terrain.Arena(spec=spec, boxes=(), hfield_data=hfield_data, lookup=lookup)


def write_accel_arena(config: HFieldConfig | None = None, seed: int = 0) -> None:
    config = config or HFieldConfig()
    arena = build_accel_arena(config, seed=seed)
    out = paths.terrain_paths("accel")
    terrain.write_hfield_bin(out["hfield"], arena.hfield_data)
    hfield_file = os.path.relpath(out["hfield"], _robot_meshdir()).replace(os.sep, "/")
    out["scene"].write_text(build_accel_scene_xml(config, hfield_file))
    out["spec"].write_text(json.dumps(terrain.spec_to_dict(arena.spec), indent=2))
    s = arena.spec
    np.savez_compressed(
        out["lookup"],
        lookup=arena.lookup,
        x_min=s.x_min,
        x_max=s.x_max,
        y_min=s.y_min,
        y_max=s.y_max,
        cell_size=s.cell_size,
        nrow=s.hfield.nrow,
        ncol=s.hfield.ncol,
    )
    for p in (out["hfield"], out["scene"], out["spec"], out["lookup"]):
        print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    write_accel_arena()
