from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any

import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig

from config.hfield_config import HFieldConfig
from config.schema import CONF_DIR, ArenaBuildConfig, to_dataclass
from hfield.tiles import TileLayout
from wojtek_env import paths
from wojtek_env.terrain import terrain

# mujoco is a C extension that ty cannot introspect
mj: Any = mujoco

_APRON_EXTENT = 25.0
_APRON_HALF_Z = 0.5


def _robot_meshdir():
    match = re.search(r'meshdir="([^"]+)"', paths.ROBOT_XML.read_text())
    rel = match.group(1) if match else "."
    return (paths.MUJOCO_DIR / rel).resolve()


def _apron_strips(
    radius: float,
) -> list[tuple[str, float, float, float, float]]:
    """(name, centre x, centre y, half x, half y) of the flat boxes around a
    square of half-width `radius`."""
    outer, mid = radius + _APRON_EXTENT, radius + _APRON_EXTENT / 2
    half = _APRON_EXTENT / 2
    return [
        ("accel_apron_north", 0.0, mid, outer, half),
        ("accel_apron_south", 0.0, -mid, outer, half),
        ("accel_apron_west", -mid, 0.0, half, radius),
        ("accel_apron_east", mid, 0.0, half, radius),
    ]


def _apron_geoms(radius: float) -> list[str]:
    return [
        f'    <geom name="{name}" type="box" '
        f'size="{hx:.4f} {hy:.4f} {_APRON_HALF_Z}" '
        f'pos="{cx:.4f} {cy:.4f} {-_APRON_HALF_Z}" material="groundplane" '
        f'condim="3" conaffinity="15"/>'
        for name, cx, cy, hx, hy in _apron_strips(radius)
    ]


def single_tile(config: HFieldConfig) -> HFieldConfig:
    """One tile and the narrowest gap: an arena that shows a single level."""
    return dataclasses.replace(config, tiles=1, reach=config.patch_radius)


def single_tile_spec(scene: Path, config: HFieldConfig) -> Any:
    """`scene` with its tiled arena cut to `single_tile`, the level centred
    at the origin. For CPU rendering: the renderer meshes the whole hfield,
    which for the full arena is millions of cells."""
    layout = TileLayout(single_tile(config))
    spec = mj.MjSpec.from_file(str(scene))
    hfield = spec.hfields[0]
    hfield.file = ""
    hfield.nrow = hfield.ncol = layout.nodes
    hfield.size = [
        layout.half,
        layout.half,
        2 * config.amplitude,
        config.base_z,
    ]
    hfield.userdata = np.full(layout.nodes**2, 0.5, np.float32).tolist()
    for name, cx, cy, hx, hy in _apron_strips(layout.half):
        geom = spec.geom(name)
        geom.pos = [cx, cy, -_APRON_HALF_Z]
        geom.size = [hx, hy, _APRON_HALF_Z]
    return spec


def build_accel_scene_xml(config: HFieldConfig, hfield_file: str) -> str:
    half = TileLayout(config).half
    size = (
        f"{half:.4f} {half:.4f} {2 * config.amplitude:.6f} {config.base_z:.4f}"
    )
    apron = "\n".join(_apron_geoms(half))
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
"""  # noqa: E501 (MJCF template: wrapping would change the written scene)


def build_accel_arena(config: HFieldConfig, seed: int = 0) -> terrain.Arena:
    layout = TileLayout(config)
    grid = layout.nodes
    radius = layout.half
    hfield_data = np.full((grid, grid), 0.5, dtype=np.float32)
    # MuJoCo rescales a loaded hfield to [0, 1], sinking a constant 0.5 to
    # -amplitude; two far corners at 0 and 1 make the rescale the identity
    hfield_data[0, 0] = 0.0
    hfield_data[-1, -1] = 1.0
    lookup = (2 * hfield_data - 1) * config.amplitude
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
        tile_size=2 * config.patch_radius,
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
    return terrain.Arena(
        spec=spec, boxes=(), hfield_data=hfield_data, lookup=lookup
    )


def write_accel_arena(
    config: HFieldConfig, seed: int = 0, kind: str = "accel"
) -> None:
    arena = build_accel_arena(config, seed=seed)
    out = paths.terrain_paths(kind)
    terrain.write_hfield_bin(out["hfield"], arena.hfield_data)
    spec = arena_spec(config, arena)
    hfield_file = os.path.relpath(out["hfield"], _robot_meshdir()).replace(
        os.sep, "/"
    )
    out["scene"].write_text(build_accel_scene_xml(config, hfield_file))
    out["spec"].write_text(json.dumps(spec, indent=2))
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


def arena_spec(config: HFieldConfig, arena: terrain.Arena) -> dict[str, Any]:
    return {
        **terrain.spec_to_dict(arena.spec),
        "tile_layout": tile_layout(config),
    }


def tile_layout(config: HFieldConfig) -> dict[str, float]:
    layout = TileLayout(config)
    return {
        "tiles": layout.per_side,
        "level_grid": config.grid_size,
        "gap": round(layout.gap, 6),
    }


def check_arena(spec_path: Path, config: HFieldConfig, kind: str) -> None:
    rebuild = (
        f"build it with `just build-accel-arena "
        f"arena.patch_radius={config.patch_radius} "
        f"arena.cell_size={config.cell_size} "
        f"arena.amplitude={config.amplitude} "
        f"arena.coarse_n={config.coarse_n} "
        f"arena.pad_radius={config.pad_radius} "
        f"arena.pad_taper={config.pad_taper} "
        f"arena.tiles={config.tiles} "
        f"arena.reach={config.reach}` (the run's `arena` overrides)"
    )
    if not spec_path.exists():
        raise FileNotFoundError(f"no {kind} arena at {spec_path}; {rebuild}")
    spec = json.loads(spec_path.read_text())
    layout = spec["tile_layout"]
    built = {
        "pad_radius": spec["pad_radius"],
        "amplitude": spec["hfield"]["elevation_z"] / 2,
        **{k: layout[k] for k in ("tiles", "level_grid", "gap")},
    }
    wanted = {
        "pad_radius": config.pad_radius,
        "amplitude": config.amplitude,
        **tile_layout(config),
    }
    # amplitude sets the hfield's height scale: a mismatch runs silently
    if not all(np.isclose(built[k], wanted[k]) for k in wanted):
        raise ValueError(
            f"the {kind} arena at {spec_path} was built with {built}, this "
            f"run needs {wanted}; re{rebuild}"
        )


@hydra.main(config_path=CONF_DIR, config_name="arena", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    cfg = to_dataclass(dict_cfg, ArenaBuildConfig)
    write_accel_arena(cfg.arena, seed=cfg.seed)


if __name__ == "__main__":
    main()
