"""Checkpoint + training config -> artifacts/export/<name>/ (policy.onnx,
config.yaml, README.md)."""

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx
from brax.io import model as brax_model
from brax.training.agents.ppo import networks as ppo_networks
from omegaconf import DictConfig, OmegaConf

from config.schema import (
    ExportConfig,
    Robot,
    TrainConfig,
    as_dict,
    to_dataclass,
)
from env.factory import check_wojtek_arena, make_env, tile_reset
from env.functional import FunctionalEnv
from export.contract import Contract, build_contract
from export.onnx_graph import build_model
from export.readme import render_readme
from export.verify import Targets, Verification, verify
from hfield.level_space import make_level_space
from hfield.tiles import TileLayout
from robots.scale import ROBOT_SCALE, hfield_config
from training.ppo import make_network
from training.summary import read_summary
from training.unroll import make_unroll

ENV_LEVELS = 4  # levels of the rollout that yields real observations
ENV_PER_LEVEL = 2
ENV_STEPS = 50
ENV_DIFFICULTY = 0.5
OUTPUT_FILES = ("policy.onnx", "config.yaml", "README.md")
# where / on what this machine runs: a `config=` file from another machine
# must not override these
LOCAL_FIELDS = ("backend", "checkpoint_dir", "export_dir")


def training_config(dict_cfg: DictConfig) -> TrainConfig:
    """The composed export config as the training config. `config` names a
    YAML (a run's `.hydra/config.yaml`, an export's `config.yaml`) whose
    values replace the composed ones, except `LOCAL_FIELDS`; fields it lacks
    (older runs) keep the composed conf/train.yaml values."""
    path = dict_cfg.get("config")
    if path:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"config={path}: no such file; pass the run's "
                "<run dir>/.hydra/config.yaml or an export's config.yaml"
            )
        local = {k: dict_cfg[k] for k in LOCAL_FIELDS}
        dict_cfg = OmegaConf.merge(  # ty: ignore[invalid-assignment]
            dict_cfg, OmegaConf.load(path), local
        )
    full = to_dataclass(dict_cfg, ExportConfig)
    return TrainConfig(
        **{
            f.name: getattr(full, f.name)
            for f in dataclasses.fields(TrainConfig)
        }
    )


def checkpoint_params(checkpoint: Path) -> Any:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint {checkpoint} not found; pass checkpoint=<the .pkl "
            "a run wrote to its checkpoint_dir>"
        )
    return brax_model.load_params(str(checkpoint))


def make_export_env(
    cfg: TrainConfig,
) -> tuple[FunctionalEnv, TileLayout, Any]:
    robot = cfg.robot.name
    config = hfield_config(robot, cfg.arena)
    check_wojtek_arena(robot, config)
    space = make_level_space(
        cfg.level_space.kind,
        config,
        n_edits=cfg.level_space.n_edits,
        edit_scale=cfg.level_space.edit_scale,
        robot_scale=ROBOT_SCALE[robot],
    )
    num_envs = ENV_LEVELS * ENV_PER_LEVEL
    env = make_env(robot, cfg.backend, config, num_envs=num_envs, train=True)
    return env, TileLayout(config), space


def policy_fn(env: FunctionalEnv, cfg: TrainConfig, params: Any) -> Any:
    network = make_network(env, cfg.network)
    make_policy = ppo_networks.make_inference_fn(network)
    deterministic = make_policy(params, deterministic=True)
    key = jax.random.key(0)

    def policy(obs: jax.Array) -> jax.Array:
        return deterministic({"state": obs}, key)[0]

    return policy


def target_fn(env: FunctionalEnv, contract: Contract) -> Targets:
    """The env's own action -> joint target mapping."""
    base = env.base
    scale = jnp.asarray([j.action_scale for j in contract.joints])
    lo = jnp.asarray([j.target_min for j in contract.joints])
    hi = jnp.asarray([j.target_max for j in contract.joints])
    if contract.stance is None:
        origin = jnp.asarray(base._default_pose)
        return lambda obs, action: jnp.clip(origin + action * scale, lo, hi)
    index = contract.stance.height_index

    def targets(obs: jax.Array, action: jax.Array) -> jax.Array:
        stance = jax.vmap(base._height_ctrl)(obs[:, index])
        return jnp.clip(stance + action * scale, lo, hi)

    return targets


def env_observations(
    env: FunctionalEnv,
    layout: TileLayout,
    space: Any,
    cfg: TrainConfig,
    params: Any,
) -> np.ndarray:
    """Actor observations of the deterministic policy walking real levels."""
    network = make_network(env, cfg.network)
    make_policy = ppo_networks.make_inference_fn(network)
    levels = space.scaled(np.random.default_rng(0), ENV_LEVELS, ENV_DIFFICULTY)
    hfield = jax.jit(layout.assemble)(jnp.asarray(space.render(levels)))
    tile_xy = layout.tile_xy(ENV_LEVELS, ENV_PER_LEVEL)
    keys = jax.random.split(jax.random.key(0), tile_xy.shape[0])
    _, state = tile_reset(env)(keys, hfield, tile_xy)
    unroll = make_unroll(env, make_policy, None)
    _, data = unroll(
        params,
        state,
        jax.random.key(1),
        hfield,
        None,
        deterministic=True,
        length=ENV_STEPS,
    )
    obs = np.asarray(data.observation["state"], np.float32)
    return obs.reshape(-1, obs.shape[-1])


def write_outputs(
    out_dir: Path, model: onnx.ModelProto, cfg: TrainConfig, readme: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_dir / "policy.onnx")
    OmegaConf.save(OmegaConf.create(as_dict(cfg)), out_dir / "config.yaml")
    (out_dir / "README.md").write_text(readme)


def export_policy(
    cfg: TrainConfig, checkpoint: Path, out_root: Path
) -> tuple[Path, Verification]:
    params = checkpoint_params(checkpoint)
    env, layout, space = make_export_env(cfg)
    contract = build_contract(Robot(cfg.robot.name), env)
    model = build_model(params, cfg.network, contract)
    ver = verify(
        model,
        params,
        contract,
        policy_fn(env, cfg, params),
        target_fn(env, contract),
        env_observations(env, layout, space, cfg, params),
    )
    readme = render_readme(
        cfg.name, cfg, contract, ver, read_summary(checkpoint)
    )
    out_dir = out_root / cfg.name
    write_outputs(out_dir, model, cfg, readme)
    return out_dir, ver
