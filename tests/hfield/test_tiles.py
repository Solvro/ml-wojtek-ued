import jax.numpy as jnp
import numpy as np
import pytest

from config.hfield_config import HFieldConfig
from hfield.tiles import TileLayout
from scripts.brax_ued import hfield_config, parse_args

SMALL = HFieldConfig(patch_radius=1.0, tiles=3, reach=1.5)


def benchmark_layout(robot: str = "wojtek") -> TileLayout:
    flags = ["--mode", "dr", "--name", "t", "--robot", robot]
    return TileLayout(hfield_config(parse_args(flags)))


def node(layout: TileLayout, xy: np.ndarray) -> tuple[int, int]:
    """(row, col) of the arena node at world xy."""
    col, row = np.round((xy + layout.half) / layout.config.cell_size).astype(int)
    return int(row), int(col)


def distinct_levels(layout: TileLayout, k: int) -> np.ndarray:
    g = layout.config.grid_size
    ramp = np.linspace(0.0, 1.0, g * g, dtype=np.float32)
    return np.stack([np.roll(ramp, 7 * i) for i in range(k)])


def test_benchmark_gap_covers_reach() -> None:
    layout = benchmark_layout()
    assert layout.gap == pytest.approx(7.0)
    assert layout.nodes == 4161
    assert layout.leak_radius == pytest.approx(layout.config.reach)
    assert layout.n_tiles == 256


def test_go1_gap_scales_with_the_robot() -> None:
    layout = benchmark_layout("go1")
    assert layout.config.reach == pytest.approx(25.0)
    assert layout.gap == pytest.approx(17.5)


def test_gap_has_a_flat_node_when_reach_is_inside_the_patch() -> None:
    layout = TileLayout(HFieldConfig(patch_radius=3.0, reach=1.0))
    assert layout.gap_cells == 2


def test_tile_i_holds_level_i_at_its_centre() -> None:
    layout = TileLayout(SMALL)
    g, cell = SMALL.grid_size, SMALL.cell_size
    levels = distinct_levels(layout, layout.n_tiles)
    grid = np.asarray(layout.assemble(jnp.asarray(levels))).reshape(
        layout.nodes, layout.nodes
    )
    local = np.array([0.35, -0.6])
    j_col, j_row = np.round((local + SMALL.patch_radius) / cell).astype(int)
    for i, centre in enumerate(layout.centres()):
        row, col = node(layout, centre + local)
        assert grid[row, col] == levels[i].reshape(g, g)[j_row, j_col]
        r0, c0 = node(layout, centre - SMALL.patch_radius)
        np.testing.assert_array_equal(
            grid[r0 : r0 + g, c0 : c0 + g], levels[i].reshape(g, g)
        )


def test_gaps_and_unused_tiles_stay_flat() -> None:
    layout = TileLayout(SMALL)
    g = SMALL.grid_size
    levels = np.ones((2, g * g), np.float32)
    grid = np.asarray(layout.assemble(jnp.asarray(levels))).reshape(
        layout.nodes, layout.nodes
    )
    on_level = np.zeros_like(grid, dtype=bool)
    for centre in layout.centres()[:2]:
        r0, c0 = node(layout, centre - SMALL.patch_radius)
        on_level[r0 : r0 + g, c0 : c0 + g] = True
    assert (grid[on_level] == 1.0).all()
    assert (grid[~on_level] == 0.5).all()


def test_neighbouring_patches_are_a_gap_apart() -> None:
    layout = TileLayout(SMALL)
    first, second = layout.centres()[:2]
    edge_to_edge = second[0] - first[0] - 2 * SMALL.patch_radius
    assert edge_to_edge == pytest.approx(layout.gap)
    assert second[1] == first[1]


def test_assemble_refuses_more_levels_than_tiles() -> None:
    layout = TileLayout(SMALL)
    levels = jnp.zeros((layout.n_tiles + 1, SMALL.grid_size**2))
    with pytest.raises(ValueError, match="hfield_params.tiles=4"):
        layout.assemble(levels)


def test_tile_xy_is_level_major() -> None:
    layout = TileLayout(SMALL)
    xy = np.asarray(layout.tile_xy(3, 2))
    centres = layout.centres()
    np.testing.assert_array_equal(xy, np.repeat(centres[:3], 2, axis=0))


def test_leaked_once_past_the_gap() -> None:
    layout = TileLayout(SMALL)
    centre = jnp.asarray(layout.centres()[0])
    inside = centre + jnp.array([layout.leak_radius - 0.01, 0.0])
    across = centre + jnp.array([0.0, -layout.leak_radius - 0.01])
    assert not bool(layout.leaked(inside, centre))
    assert bool(layout.leaked(across, centre))


def test_leaked_counts_what_the_robot_reads_ahead() -> None:
    layout = TileLayout(SMALL)
    centre = jnp.asarray(layout.centres()[0])
    xy = centre + jnp.array([layout.leak_radius - 0.5, 0.0])
    assert not bool(layout.leaked(xy, centre, sensing=0.4))
    assert bool(layout.leaked(xy, centre, sensing=0.6))


def test_check_reach_refuses_past_the_gap() -> None:
    layout = TileLayout(SMALL)
    layout.check_reach(layout.leak_radius, "at the edge")
    with pytest.raises(ValueError, match=r"past the gap.*HFieldConfig.reach >= 2"):
        layout.check_reach(layout.leak_radius + 0.1, "past the gap")
