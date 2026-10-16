"""FunctionalEnv on the shared tiled arena: each env sees its own tile.

Builds a 2x2-tile arena under the `test` kind and the jax backend on CPU;
needs the robot model (`just build`).
"""

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from config.hfield_config import HFieldConfig
from hfield.arena import write_accel_arena
from hfield.level_space import ParamSpace
from hfield.tiles import TileLayout
from utils.env import FunctionalEnv
from utils.env_overrides import (
    accel_arena_overrides,
    merge_overrides,
    movement_overrides,
    terrain_overrides,
)
from wojtek_env import paths

CONFIG = HFieldConfig(
    patch_radius=1.5,
    amplitude=0.3,
    pad_radius=0.55,
    pad_taper=0.2,
    tiles=2,
    reach=2.0,
)
LAYOUT = TileLayout(CONFIG)
STEPS = 40
PIT_M = 0.05
SCAN = 25  # height_scan_clean, the last critic obs component

pytestmark = pytest.mark.skipif(
    not paths.ROBOT_XML.exists(), reason="robot model not built; run `just build`"
)


@pytest.fixture(scope="module")
def env() -> Iterator[FunctionalEnv]:
    write_accel_arena(CONFIG, kind="test")
    overrides = merge_overrides(
        accel_arena_overrides(CONFIG.pad_radius, "test"),
        movement_overrides(),
        terrain_overrides(),
        {
            "sim": {"num_envs": LAYOUT.n_tiles, "backend": "jax"},
            "no_progress": {"enable": False},
            "push": {"enable": False},
            "command": {"zero_prob": 1.0, "height": (0.14, 0.14)},
        },
    )
    yield FunctionalEnv(task="joystick", env_overrides=overrides)
    for path in paths.terrain_paths("test").values():
        path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def run(env: FunctionalEnv) -> Any:
    """(levels on tiles, key per env) -> per-step critic obs, z, xy, dones;
    zero action, env i on tile i."""
    action = jnp.zeros(env.action_size)

    def one(arena: jax.Array, key: jax.Array, tile: jax.Array) -> Any:
        obs, state = env.reset(key, arena, tile_xy=tile)

        def step(carry: Any, _: None) -> tuple[Any, Any]:
            obs, state = carry
            out = (obs.critic, state.data.qpos[2], state.data.qpos[:2] - tile)
            obs, state, _reward, done, _info = env.step(key, state, action, arena)
            return (obs, state), (*out, done)

        _, out = jax.lax.scan(step, (obs, state), None, STEPS)
        return out

    @jax.jit
    def rollout(levels: jax.Array, keys: jax.Array) -> Any:
        arena = LAYOUT.assemble(levels)
        tiles = LAYOUT.tile_xy(LAYOUT.n_tiles, 1)
        return jax.vmap(one, in_axes=(None, 0, 0))(arena, keys, tiles)

    return rollout


def param_levels() -> np.ndarray:
    space = ParamSpace(CONFIG)
    return space.render(space.scaled(np.random.default_rng(0), LAYOUT.n_tiles, 0.6))


def test_a_level_reads_the_same_on_every_tile(run: Any) -> None:
    levels = param_levels()
    keys = jax.random.split(jax.random.key(0), LAYOUT.n_tiles)
    perm = np.array([2, 0, 3, 1])
    base = run(jnp.asarray(levels), keys)
    # level perm[i] (and its env's key) moved to tile i, new neighbours
    moved = run(jnp.asarray(levels[perm]), keys[perm])
    flat = run(jnp.full_like(jnp.asarray(levels), 0.5), keys)
    critic, z, xy, done = (np.asarray(x) for x in base)
    critic_m, z_m, xy_m, done_m = (np.asarray(x) for x in moved)
    scan = critic[..., -SCAN:]
    # the height scan reads the level ahead of the pad, wherever it lies;
    # float32 rounding of the tile offset perturbs the standing robot by
    # ~0.5 mm (chaotic spawn settle), not by a level's worth
    np.testing.assert_allclose(critic_m[..., -SCAN:], scan[perm], atol=1e-2)
    assert np.abs(scan - np.asarray(flat[0])[..., -SCAN:]).max() > 0.1
    np.testing.assert_allclose(z_m, z[perm], atol=2e-3)
    np.testing.assert_allclose(xy_m, xy[perm], atol=5e-3)
    np.testing.assert_array_equal(done_m, done[perm])


def test_pit_tile_lowers_only_its_own_robot(run: Any) -> None:
    g = CONFIG.grid_size
    pit = 0.5 - PIT_M / (2 * CONFIG.amplitude)
    levels = np.full((LAYOUT.n_tiles, g * g), 0.5, np.float32)
    levels[1] = pit
    keys = jnp.stack([jax.random.key(0)] * LAYOUT.n_tiles)
    _, z, xy, done = (np.asarray(x) for x in run(jnp.asarray(levels), keys))
    late = z[:, -STEPS // 4 :].mean(axis=1)
    # physics and the fall check's lookup both follow the env's tile: the
    # pit robot stands PIT_M lower and does not register a fall
    assert late[1] == pytest.approx(late[0] - PIT_M, abs=3e-3)
    np.testing.assert_allclose(late[[0, 2, 3]], late[0], atol=2e-3)
    assert not done.any()
    assert np.abs(xy).max() < CONFIG.pad_radius


def test_reset_on_an_arena_requires_the_tile(env: FunctionalEnv) -> None:
    arena = LAYOUT.assemble(jnp.asarray(param_levels()))
    with pytest.raises(ValueError, match="tile_xy"):
        env.reset(jax.random.key(0), arena)
