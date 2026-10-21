from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mediapy
import numpy as np
from brax.io import model as brax_model
from brax.training.agents.ppo import networks as ppo_networks

from config.schema import NetworkConfig, RenderConfig
from env.factory import check_wojtek_arena, make_env
from env.functional import FunctionalEnv
from hfield.level_space import LevelSpace, make_level_space
from hfield.tiles import TileLayout
from rendering.scene import (
    level_on_single_tile,
    mj,
    render_model,
    tracking_camera,
)
from robots.scale import hfield_config, robot_scale
from training.ppo import make_network

Rollout = Callable[
    [jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]
]


def parse_level(text: str) -> tuple[float, int]:
    difficulty, sep, seed = text.partition(":")
    try:
        level = float(difficulty), int(seed)
    except ValueError:
        level = None
    if not sep or level is None or not 0.0 <= level[0] <= 1.0:
        raise ValueError(
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
    """`info` with `command` held: every robot's resampling counter reset."""
    held = {**info, "command": info["command"].at[:3].set(command)}
    if "steps_since_cmd" in held:
        # Wojtek resamples every resample_steps
        held["steps_since_cmd"] = jnp.zeros_like(held["steps_since_cmd"])
    if "steps_until_next_cmd" in held:
        # Go1 resamples when its countdown reaches 0
        counter = held["steps_until_next_cmd"]
        held["steps_until_next_cmd"] = jnp.full_like(counter, 2**30)
    if "step" in held:
        # Barkour and Spot resample (Barkour also kicks) off this counter
        held["step"] = jnp.zeros_like(held["step"])
    return held


def make_rollout(
    env: FunctionalEnv,
    params: Any,
    command: jax.Array,
    n_steps: int,
    network: NetworkConfig,
) -> Rollout:
    """(key, arena hfield_data, tile_xy) -> per-step qpos before the step,
    and fell flags."""
    make_policy = ppo_networks.make_inference_fn(make_network(env, network))
    policy = make_policy(params, deterministic=True)

    def step(
        state: Any, hfield: jax.Array
    ) -> tuple[Any, tuple[jax.Array, ...]]:
        state = state.replace(info=hold_command(state.info, command))
        action, _ = policy(state.obs, jax.random.key(0))
        _, nxt, _, done, info = env.step(
            jax.random.key(0), state, action, hfield
        )
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


def render_frames(
    model: Any,
    hfield: np.ndarray,
    qpos: np.ndarray,
    steps: np.ndarray,
    *,
    cfg: RenderConfig,
    scale: float,
) -> Iterator[np.ndarray]:
    model.hfield_data[:] = hfield
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, cfg.width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, cfg.height)
    data = mj.MjData(model)
    camera = tracking_camera(model, scale)
    # the renderer uploads the hfield at creation, so build it per level
    with mj.Renderer(model, cfg.height, cfg.width) as renderer:
        for i in steps:
            data.qpos[:] = qpos[i]
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera)
            yield renderer.render()


def check_render_config(cfg: RenderConfig) -> list[tuple[float, int]]:
    levels = [parse_level(text) for text in cfg.levels]
    if len(cfg.command) != 3:
        raise ValueError(
            f"command={cfg.command} must be [VX, VY, WZ] (m/s, m/s, rad/s)"
        )
    return levels


def control_steps(cfg: RenderConfig, dt: float, episode_length: int) -> int:
    n_steps = round(cfg.seconds / dt)
    if n_steps > episode_length:
        raise ValueError(
            f"seconds={cfg.seconds} is {n_steps} control steps, past the "
            f"episode length {episode_length}; the env would reset"
        )
    return n_steps


class LevelRecorder:
    """The policy rolled out on a level, replayed through the renderer."""

    def __init__(self, cfg: RenderConfig, params: Any) -> None:
        self.cfg = cfg
        self.config = hfield_config(cfg.robot, cfg.arena)
        check_wojtek_arena(cfg.robot.name, self.config)
        env = make_env(
            cfg.robot, cfg.backend, self.config, num_envs=1, train=False
        )
        self.scale = robot_scale(cfg.robot)
        self.base: Any = env._env.unwrapped
        n_steps = control_steps(cfg, self.base.dt, env._env.episode_length)
        self.space: LevelSpace = make_level_space(
            cfg.level_space.kind,
            self.config,
            n_edits=1,
            edit_scale=0.0,
            robot_scale=self.scale,
        )
        command = jnp.asarray(cfg.command)
        self.rollout = make_rollout(env, params, command, n_steps, cfg.network)
        self.layout = TileLayout(self.config)
        self.tile_xy = self.layout.tile_xy(1, 1)[0]
        self.model = render_model(self.base, self.config)
        self.steps = frame_steps(n_steps, self.base.dt, cfg.fps)

    def clip(
        self, difficulty: float, seed: int
    ) -> tuple[str, Iterator[np.ndarray]]:
        """The clip's file name and frames; the rollout runs now."""
        cfg = self.cfg
        level = self.space.scaled(np.random.default_rng(seed), 1, difficulty)
        hfield = self.space.render(level)[0]
        arena = self.layout.assemble(jnp.asarray(hfield)[None])
        key = jax.random.key(cfg.seed)
        qpos, fell = self.rollout(key, arena, self.tile_xy)
        # a copy: np.asarray of a jax array is read-only
        qpos, fell = np.array(qpos), np.asarray(fell)
        # the render model holds the level at the origin
        qpos[:, :2] -= np.asarray(self.tile_xy)
        falls = np.flatnonzero(fell) * self.base.dt
        travel = np.linalg.norm(qpos[-1, :2] - qpos[0, :2])
        name = f"{Path(cfg.checkpoint).stem}_d{difficulty}_s{seed}.mp4"
        print(f"{name}: falls at {np.round(falls, 2)} s, travel {travel:.2f} m")
        return name, render_frames(
            self.model,
            level_on_single_tile(self.config, hfield),
            qpos,
            self.steps,
            cfg=cfg,
            scale=self.scale,
        )


def render(cfg: RenderConfig) -> None:
    """One clip per `levels` entry, plus `demo.mp4` with all of them."""
    levels = check_render_config(cfg)
    recorder = LevelRecorder(cfg, brax_model.load_params(cfg.checkpoint))
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    size = (cfg.height, cfg.width)
    with mediapy.VideoWriter(out / "demo.mp4", size, fps=cfg.fps) as demo:
        for difficulty, seed in levels:
            name, frames = recorder.clip(difficulty, seed)
            with mediapy.VideoWriter(out / name, size, fps=cfg.fps) as clip:
                for frame in frames:
                    clip.add_image(frame)
                    demo.add_image(frame)
