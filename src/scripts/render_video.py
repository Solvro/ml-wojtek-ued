"""Video of a `brax_ued.py` checkpoint walking generated levels.

Env, network and level space are rebuilt from the checkpoint's sidecar
(`<run>.json`, written by `brax_ued.py`). The deterministic policy runs on
MJX (tile 0 of the arena) under a held velocity command; the recorded
states are replayed through an offscreen MuJoCo renderer, on a one-tile copy
of the arena, with a camera tracking the base. One clip per `--level D:SEED` (a `scaled` level of
difficulty D drawn from rng seed SEED), plus `demo.mp4` = all clips in order.
Needs the built assets; MUJOCO_GL=cgl on macOS, egl/osmesa headless.
"""

import argparse
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mediapy
import mujoco
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground._src.locomotion.go1 import base as go1_base

from config.hfield_config import HFieldConfig
from hfield.arena import single_tile, single_tile_spec
from hfield.level_space import make_level_space
from hfield.tiles import TileLayout
from robots.go1 import Go1Terrain, scene_xml
from scripts.brax_ued import (
    ROBOT_SCALE,
    check_spec_arena,
    load_run,
    make_env,
    spec_network,
)
from utils.env import FunctionalEnv

# mujoco is a C extension that ty cannot introspect
mj: Any = mujoco

Rollout = Callable[[jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]]


def parse_level(text: str) -> tuple[float, int]:
    difficulty, sep, seed = text.partition(":")
    try:
        level = float(difficulty), int(seed)
    except ValueError:
        level = None
    if not sep or level is None or not 0.0 <= level[0] <= 1.0:
        raise argparse.ArgumentTypeError(
            f"level {text!r} must be DIFFICULTY:SEED with difficulty in "
            "[0, 1] and an integer seed, e.g. 0.75:3"
        )
    return level


def frame_steps(n_steps: int, ctrl_dt: float, fps: int) -> np.ndarray:
    """Control step shown in each video frame, nearest in time."""
    n_frames = round(n_steps * ctrl_dt * fps)
    steps = np.round(np.arange(n_frames) / (fps * ctrl_dt)).astype(int)
    return np.minimum(steps, n_steps - 1)


def hold_command(info: dict[str, Any], command: jax.Array) -> dict[str, Any]:
    held = {**info, "command": info["command"].at[:3].set(command)}
    if "steps_since_cmd" in held:
        # stops Wojtek resampling the command every resample_steps
        held["steps_since_cmd"] = jnp.zeros_like(held["steps_since_cmd"])
    if "steps_until_next_cmd" in held:
        # stops Go1 resampling when its countdown reaches 0
        counter = held["steps_until_next_cmd"]
        held["steps_until_next_cmd"] = jnp.full_like(counter, 2**30)
    return held


def make_rollout(
    env: FunctionalEnv,
    spec: dict[str, Any],
    params: Any,
    command: jax.Array,
    n_steps: int,
) -> Rollout:
    """(key, arena hfield_data, tile_xy) -> per-step qpos before the step,
    and fell flags."""
    make_policy = ppo_networks.make_inference_fn(spec_network(spec))
    policy = make_policy(params, deterministic=True)

    def step(state: Any, hfield: jax.Array) -> tuple[Any, tuple[jax.Array, ...]]:
        state = state.replace(info=hold_command(state.info, command))
        action, _ = policy(state.obs, jax.random.key(0))
        _, nxt, _, done, info = env.step(jax.random.key(0), state, action, hfield)
        return nxt, (state.data.qpos, done * (1.0 - info["truncation"]))

    @jax.jit
    def rollout(
        key: jax.Array, hfield: jax.Array, tile_xy: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        _, state = env.reset(key, hfield, tile_xy=tile_xy)
        _, (qpos, fell) = jax.lax.scan(
            lambda s, _: step(s, hfield), state, None, n_steps
        )
        return qpos, fell

    return rollout


def render_model(base: Any, config: HFieldConfig) -> Any:
    """The env's scene with its arena cut to one tile, the level at the
    origin (`single_tile`)."""
    if isinstance(base, Go1Terrain):
        xml = scene_xml(single_tile(config))
        return mj.MjModel.from_xml_string(xml, assets=go1_base.get_assets())
    return single_tile_spec(Path(base._terrain.files["scene"]), config).compile()


def level_on_single_tile(config: HFieldConfig, hfield: np.ndarray) -> np.ndarray:
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


def render_frames(
    model: Any,
    hfield: np.ndarray,
    qpos: np.ndarray,
    steps: np.ndarray,
    args: argparse.Namespace,
    scale: float,
) -> Iterator[np.ndarray]:
    model.hfield_data[:] = hfield
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
    data = mj.MjData(model)
    camera = tracking_camera(model, scale)
    # the renderer uploads the hfield at creation, so build it per level
    with mj.Renderer(model, args.height, args.width) as renderer:
        for i in steps:
            data.qpos[:] = qpos[i]
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera)
            yield renderer.render()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint", required=True, help="brax_ued.py .pkl, sidecar .json beside it"
    )
    p.add_argument("--out_dir", default="artifacts/videos")
    p.add_argument(
        "--level",
        type=parse_level,
        action="append",
        required=True,
        help="DIFFICULTY:SEED, repeatable; one clip each",
    )
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument(
        "--command",
        type=float,
        nargs=3,
        default=[0.3, 0.0, 0.2],
        metavar=("VX", "VY", "WZ"),
        help="held command (m/s, m/s, rad/s); the default arc stays on the arena",
    )
    p.add_argument("--seed", type=int, default=0, help="reset key")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec, params = load_run(Path(args.checkpoint))
    config = HFieldConfig(**spec["hfield"])
    check_spec_arena(spec)
    env = make_env(spec, num_envs=1, train=False)
    scale = ROBOT_SCALE[spec["robot"]]
    base: Any = env._env.unwrapped
    n_steps = round(args.seconds / base.dt)
    if n_steps > env._env.episode_length:
        raise ValueError(
            f"--seconds {args.seconds} is {n_steps} control steps, past the "
            f"episode length {env._env.episode_length}; the env would reset"
        )
    space = make_level_space(
        spec["level_space"], config, n_edits=1, edit_scale=0.0, robot_scale=scale
    )
    rollout = make_rollout(env, spec, params, jnp.asarray(args.command), n_steps)
    layout = TileLayout(config)
    tile_xy = layout.tile_xy(1, 1)[0]
    model = render_model(base, config)
    steps = frame_steps(n_steps, base.dt, args.fps)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    size = (args.height, args.width)
    with mediapy.VideoWriter(out / "demo.mp4", size, fps=args.fps) as demo:
        for difficulty, seed in args.level:
            level = space.scaled(np.random.default_rng(seed), 1, difficulty)
            hfield = space.render(level)[0]
            arena = layout.assemble(jnp.asarray(hfield)[None])
            qpos, fell = rollout(jax.random.key(args.seed), arena, tile_xy)
            qpos, fell = np.asarray(qpos), np.asarray(fell)
            # the render model holds the level at the origin
            qpos[:, :2] -= np.asarray(tile_xy)
            falls = np.flatnonzero(fell) * base.dt
            travel = np.linalg.norm(qpos[-1, :2] - qpos[0, :2])
            name = f"{Path(args.checkpoint).stem}_d{difficulty}_s{seed}.mp4"
            print(f"{name}: falls at {np.round(falls, 2)} s, travel {travel:.2f} m")
            with mediapy.VideoWriter(out / name, size, fps=args.fps) as clip:
                for frame in render_frames(
                    model,
                    level_on_single_tile(config, hfield),
                    qpos,
                    steps,
                    args,
                    scale,
                ):
                    clip.add_image(frame)
                    demo.add_image(frame)


if __name__ == "__main__":
    main()
