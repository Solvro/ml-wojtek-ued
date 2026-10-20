"""The Playground quadrupeds on the shared tiled arena, jax backend on CPU.

Every robot of `robots.registry` gets the same checks: the scene compiles
with the arena, the env spawns on and reads its own tile, a pit lowers only
its robot, the fall rule follows the terrain under the trunk, and the
measured standing height is what the robot's constants say.
"""

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from config.hfield_config import HFieldConfig
from env.functional import FunctionalEnv
from evaluation.physics_variants import full_collision
from hfield.tiles import TileLayout
from robots.registry import TERRAIN_ROBOTS
from robots.scale import hfield_config
from robots.spot import SpotTerrain
from robots.terrain import FALL_HEIGHT_RATIO, WOJTEK_HEIGHT, playground_backend

ARENA = HFieldConfig(
    patch_radius=1.5,
    amplitude=0.3,
    pad_radius=0.55,
    pad_taper=0.2,
    tiles=2,
    reach=2.0,
)
STEPS = 60
SETTLE_STEPS = 300
PIT_M = 0.03  # Wojtek units

ROBOTS = list(TERRAIN_ROBOTS)


class Setup:
    def __init__(self, robot: str):
        self.config = hfield_config(robot, ARENA)  # ty: ignore[invalid-argument-type]
        self.layout = TileLayout(self.config)
        self.base: Any = TERRAIN_ROBOTS[robot](self.config, "jax")
        self.env = FunctionalEnv(robot=self.base, domain_randomize=True)

    def flat(self) -> jax.Array:
        n = self.layout.nodes
        return jnp.full(n * n, 0.5, jnp.float32)

    def level_arena(self, levels: dict[int, float]) -> jax.Array:
        g = self.config.grid_size
        grid = np.full((self.layout.n_tiles, g * g), 0.5, np.float32)
        for tile, value in levels.items():
            grid[tile] = value
        return self.layout.assemble(jnp.asarray(grid))

    def rollout(
        self, arena: jax.Array
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Zero action, env i on tile i: (z, xy - tile, done) per step."""
        env, layout = self.env, self.layout
        action = jnp.zeros(env.action_size)

        def one(arena: jax.Array, key: jax.Array, tile: jax.Array) -> Any:
            _, state = env.reset(key, arena, tile_xy=tile)

            def step(state: Any, _x: None) -> tuple[Any, Any]:
                out = (state.data.qpos[2], state.data.qpos[:2] - tile)
                _, state, _, done, _ = env.step(key, state, action, arena)
                return state, (*out, done)

            return jax.lax.scan(step, state, None, STEPS)[1]

        keys = jnp.stack([jax.random.key(0)] * layout.n_tiles)
        tiles = layout.tile_xy(layout.n_tiles, 1)
        z, xy, done = jax.jit(jax.vmap(one, in_axes=(None, 0, 0)))(
            arena, keys, tiles
        )
        return np.asarray(z), np.asarray(xy), np.asarray(done)


@pytest.fixture(scope="module", params=ROBOTS)
def setup(request: pytest.FixtureRequest) -> Iterator[Setup]:
    yield Setup(request.param)


def test_scene_compiles_with_one_hfield_floor_first(setup: Setup) -> None:
    m = setup.base.mj_model
    nodes = setup.layout.nodes
    assert (m.nhfield, m.hfield_nrow[0], m.hfield_ncol[0]) == (1, nodes, nodes)
    assert m.geom_type[0] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert m.geom(0).name == "floor"
    assert m.hfield_size[0, 0] == pytest.approx(setup.layout.half, rel=1e-4)


def test_scene_scales_with_the_robot(setup: Setup) -> None:
    scale = type(setup.base).SCALE
    assert setup.config.patch_radius == pytest.approx(
        ARENA.patch_radius * scale
    )
    assert setup.config.amplitude == pytest.approx(ARENA.amplitude * scale)


def test_reset_spawns_on_the_requested_tile(setup: Setup) -> None:
    arena = setup.flat()
    centres = setup.layout.tile_xy(setup.layout.n_tiles, 1)
    for tile in (0, 3):
        _, state = setup.env.reset(
            jax.random.key(1), arena, tile_xy=centres[tile]
        )
        # Playground's Go1 reset jitters the spawn by up to 0.5 m
        offset = np.abs(np.asarray(state.data.qpos[:2] - centres[tile])).max()
        assert offset <= 0.5
    assert (
        float(state.data.qpos[2])
        > FALL_HEIGHT_RATIO * type(setup.base).STANDING_HEIGHT
    )


def test_reset_on_an_arena_requires_the_tile(setup: Setup) -> None:
    with pytest.raises(ValueError, match="tile_xy"):
        setup.env.reset(jax.random.key(0), setup.flat())


def test_terrain_height_reads_the_tiles_level(setup: Setup) -> None:
    amp = setup.config.amplitude
    arena = setup.level_arena({1: 0.75, 2: 0.0})
    centres = setup.layout.tile_xy(setup.layout.n_tiles, 1)
    with setup.env._swapped_model(arena, None, centres[0]):
        height = np.asarray(setup.base.terrain_height(centres))
    # the spawn pad of a level is flat only for generated levels: here the
    # level is a constant, so its centre reads that constant
    np.testing.assert_allclose(height, [0.0, 0.5 * amp, -amp, 0.0], atol=1e-4)


def test_pit_tile_lowers_only_its_own_robot(setup: Setup) -> None:
    scale = type(setup.base).SCALE
    pit_m = PIT_M * scale
    pit = 0.5 - pit_m / (2 * setup.config.amplitude)
    z_flat, _, _ = setup.rollout(setup.flat())
    z, xy, done = setup.rollout(setup.level_arena({1: pit}))
    late = slice(-STEPS // 4, None)
    drop = (z_flat[1, late] - z[1, late]).mean()
    # the zero-action stand keeps rocking: the pit robot sits roughly pit_m
    # lower, every other tile's robot is where it was without the pit
    assert pit_m / 2 < drop < 1.5 * pit_m
    np.testing.assert_allclose(
        z[[0, 2, 3]], z_flat[[0, 2, 3]], atol=3e-3 * scale
    )
    assert not done.any()
    assert np.abs(xy).max() < setup.config.pad_radius


def test_standing_height_is_the_measured_scale(setup: Setup) -> None:
    cls = type(setup.base)
    centre = setup.layout.tile_xy(1, 1)[0]
    arena, action = setup.flat(), jnp.zeros(setup.env.action_size)
    key = jax.random.key(0)

    @jax.jit
    def settle(arena: jax.Array) -> jax.Array:
        _, state = setup.env.reset(key, arena, tile_xy=centre)

        def step(state: Any, _x: None) -> tuple[Any, jax.Array]:
            _, state, _, _, _ = setup.env.step(key, state, action, arena)
            return state, state.data.qpos[2]

        return jax.lax.scan(step, state, None, SETTLE_STEPS)[1]

    standing = float(settle(arena)[-SETTLE_STEPS // 4 :].mean())
    assert standing == pytest.approx(cls.STANDING_HEIGHT, rel=0.03)
    assert standing / WOJTEK_HEIGHT == pytest.approx(cls.SCALE, abs=0.1)


def test_commands_walk_forward_at_the_robots_size(setup: Setup) -> None:
    cls = type(setup.base)
    keys = jax.random.split(jax.random.key(0), 2000)
    vx, vy, wz = np.asarray(jax.vmap(setup.base.sample_command)(keys)).T
    if cls is SpotTerrain:
        top_vx, top_vy = 1.0, 0.8  # Playground's Spot command range
    else:
        top_vx, top_vy = 0.6 * cls.SCALE, 0.2 * cls.SCALE
    assert vx.min() >= 0.2 * cls.SCALE - 1e-6
    assert vx.max() <= top_vx + 1e-6
    assert vx.max() > 0.98 * top_vx
    assert np.abs(vy).max() <= top_vy + 1e-6
    assert np.abs(wz).max() <= 0.4 + 1e-6


def posed(setup: Setup, qpos: jax.Array) -> mjx.Data:
    base = setup.base
    data = mjx.make_data(base.mj_model, impl="jax")
    return mjx.forward(base.mjx_model, data.replace(qpos=qpos))


def test_fall_rule_fires_when_toppled_not_at_home(setup: Setup) -> None:
    base, cls = setup.base, type(setup.base)
    centre = setup.layout.tile_xy(1, 1)[0]
    home = jnp.asarray(base.mj_model.key("home").qpos).at[:2].set(centre)
    home = home.at[2].set(cls.STANDING_HEIGHT)
    flip = jnp.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
    with setup.env._swapped_model(setup.flat(), None, centre):
        assert not bool(base._get_termination(posed(setup, home)))
        toppled = home.at[3:7].set(flip)
        assert bool(base._get_termination(posed(setup, toppled)))
        sunk = home.at[2].set(0.5 * cls.STANDING_HEIGHT)
        assert bool(base._get_termination(posed(setup, sunk)))


def test_fall_rule_measures_height_above_the_terrain_under_the_trunk(
    setup: Setup,
) -> None:
    base, cls = setup.base, type(setup.base)
    amp = setup.config.amplitude
    centre = setup.layout.tile_xy(2, 1)[1]
    home = jnp.asarray(base.mj_model.key("home").qpos).at[:2].set(centre)
    arena = setup.level_arena({1: 0.0})
    with setup.env._swapped_model(arena, None, centre):
        on_floor = home.at[2].set(-amp + cls.STANDING_HEIGHT)
        assert not bool(base._get_termination(posed(setup, on_floor)))
        # the same world height is a fall over flat ground when the pit is
        # deeper than the trunk's fall margin
        assert amp > (1 - FALL_HEIGHT_RATIO) * cls.STANDING_HEIGHT
    with setup.env._swapped_model(setup.flat(), None, centre):
        assert bool(base._get_termination(posed(setup, on_floor)))


def test_playground_robots_run_on_jax() -> None:
    assert playground_backend("auto") == "jax"
    assert playground_backend("jax") == "jax"
    with pytest.raises(ValueError, match="backend=jax"):
        playground_backend("warp")


def test_domain_randomization_only_where_playground_has_one(
    setup: Setup,
) -> None:
    name = type(setup.base).__name__
    model, _ = setup.env.sample_dr(jax.random.key(0), 3)
    if name in ("Go1Terrain", "Go2Terrain"):
        assert model is not None
        friction = np.asarray(model.geom_friction)[:, 0, 0]
        assert ((friction >= 0.4) & (friction <= 1.0)).all()
    else:
        assert model is None


def colliding(model: Any) -> int:
    return int((model.geom_contype | model.geom_conaffinity).astype(bool).sum())


def test_full_collision_variant_only_where_a_model_exists(setup: Setup) -> None:
    base = setup.base
    if base.FULL_COLLISION is None:
        with pytest.raises(ValueError, match="no full-collision model"):
            full_collision(base.mj_model, base)
        return
    full = full_collision(base.mj_model, base)
    assert colliding(full) > colliding(base.mj_model)


def test_a_diverged_step_does_not_end_the_next_episode(setup: Setup) -> None:
    env, arena = setup.env, setup.flat()
    key = jax.random.key(0)
    tile = setup.layout.tile_xy(1, 1)[0]
    _, state = env.reset(key, arena, tile_xy=tile)
    nan_action = jnp.full(env.action_size, jnp.nan)
    _, state, _, _, info = env.step(key, state, nan_action, arena)
    assert float(info["diverged"]) == 1.0
    zero = jnp.zeros(env.action_size)
    for _ in range(3):
        _, state, reward, _, info = env.step(key, state, zero, arena)
        assert float(info["diverged"]) == 0.0
        assert np.isfinite(float(reward))
