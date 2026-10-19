"""DR vs ACCEL on the heightfield benchmark, with brax's PPO as the student.

brax's own `ppo.train` fixes each env's model for the whole run, so levels
cannot change per env over training. This loop keeps brax's PPO pieces
(networks, running observation normalizer, `compute_ppo_loss`, its
mujoco_playground locomotion hyperparameters: 20-step unrolls, 4 epochs x
32 minibatches, discount 0.97) and drives `FunctionalEnv`, which takes the
shared tiled arena's `hfield_data` and a per-env physics DR draw at every
step.

One outer iteration = `num_lanes` levels x `envs_per_lane` envs: level i is
written into tile i of the arena (`hfield.tiles`), its lane's envs reset onto
that tile and step for `level_steps` (in `unroll_length` chunks, a PPO
update after each chunk when training). Per-lane regret = mean positive GAE
advantage (positive value loss) over the iteration.

- `mode=dr`: every iteration trains on fresh uniform levels at full
  difficulty.
- `mode=accel`: ACCEL with batched lanes. Replay iterations (prob
  `replay_prob` once the buffer holds `num_lanes` levels) train on levels
  drawn rank-prioritized from the buffer, refresh their scores, and queue
  their mutations. Explore iterations do not train: they score the queued
  children plus fresh *easy* levels (control points within
  `generator_scale` of flat) and add them to the buffer, which keeps the
  highest-regret `buffer_size` levels.
- `mode=plr`: robust PLR (PLR-perp) - ACCEL's buffer and replay without
  mutation; explore iterations score fresh uniform (DR) levels.
- `robot=go1|barkour|spot|go2`: a Playground quadruped (`robots/`) instead
  of Wojtek. The arena, level lengths, commands and `min_travel` scale by the
  robot's `SCALE`.

All evaluate the deterministic policy every `eval_every` iterations on
fresh uniform levels at several difficulties: success = no fall and at least
`min_travel` m from spawn after `level_steps`.

Configured by Hydra: `conf/train.yaml`, typed by `config.schema.TrainConfig`.
"""

import dataclasses
import functools
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from brax import math as brax_math
from brax.io import model as brax_model
from brax.training import types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from omegaconf import DictConfig

from config.hfield_config import HFieldConfig
from config.schema import (
    CONF_DIR,
    Backend,
    Curriculum,
    Generator,
    ModeConfig,
    Mutation,
    Robot,
    Score,
    TrainConfig,
    as_dict,
    to_dataclass,
)
from hfield.arena import check_arena
from hfield.level_space import (
    PARAMS,
    LevelSpace,
    ParamSpace,
    make_level_space,
    mean_abs_height,
)
from hfield.tiles import TileLayout
from modules.generator import LevelGenerator
from robots.registry import TERRAIN_ROBOTS
from utils.env import FunctionalEnv
from utils.env_overrides import (
    accel_arena_overrides,
    merge_overrides,
    movement_overrides,
    terrain_overrides,
    terrain_task_overrides,
)
from wojtek_env import paths

PPO: dict[str, Any] = {
    "unroll_length": 20,
    "num_minibatches": 32,
    "num_updates_per_batch": 4,
    "discounting": 0.97,
    "gae_lambda": 0.95,
    "learning_rate": 3e-4,
    "entropy_cost": 1e-2,
    "clipping_epsilon": 0.2,
    "max_grad_norm": 1.0,
}
EVAL_DIFFICULTIES = (0.25, 0.5, 0.75, 1.0)


ROBOT_SCALE = {"wojtek": 1.0, **{n: r.SCALE for n, r in TERRAIN_ROBOTS.items()}}


def body_velocity(data: Any) -> jax.Array:
    # free-joint linear qvel is world-frame; commands are body-frame
    return brax_math.inv_rotate(data.qvel[:3], data.qpos[3:7])


def hfield_config(robot: Robot, arena: HFieldConfig) -> HFieldConfig:
    """The `arena` config (Wojtek units) at the robot's size: every length
    scales, slopes do not."""
    s = ROBOT_SCALE[robot]
    return dataclasses.replace(
        arena,
        patch_radius=arena.patch_radius * s,
        cell_size=arena.cell_size * s,
        amplitude=arena.amplitude * s,
        pad_radius=arena.pad_radius * s,
        pad_taper=arena.pad_taper * s,
        reach=arena.reach * s,
    )


def check_wojtek_arena(robot: Robot, config: HFieldConfig) -> None:
    # Playground robots build their arena from `config` at construction;
    # Wojtek loads it
    if robot == Robot.wojtek:
        check_arena(paths.terrain_paths("accel")["spec"], config, "accel")


def make_env(
    robot: Robot,
    backend: Backend,
    config: HFieldConfig,
    *,
    num_envs: int,
    train: bool,
) -> FunctionalEnv:
    if robot in TERRAIN_ROBOTS:
        terrain_robot = TERRAIN_ROBOTS[robot](
            config, backend=backend, num_envs=num_envs
        )
        return FunctionalEnv(domain_randomize=train, robot=terrain_robot)
    overrides = wojtek_overrides(config, backend, num_envs, train)
    return FunctionalEnv(
        task="joystick", env_overrides=overrides, domain_randomize=train
    )


def wojtek_overrides(
    config: HFieldConfig, backend: Backend, num_envs: int, train: bool
) -> dict[str, Any]:
    return merge_overrides(
        accel_arena_overrides(config.pad_radius, backend),
        movement_overrides(),
        terrain_overrides(),
        terrain_task_overrides(),
        {"sim": {"num_envs": num_envs}},
        # eval scores falls and travel; the stochastic cut only adds noise
        {} if train else {"no_progress": {"enable": False}},
    )


def make_network(env: FunctionalEnv) -> ppo_networks.PPONetworks:
    obs_size = env.observation_size
    return ppo_networks.make_ppo_networks(
        {"state": (obs_size.actor,), "privileged_state": (obs_size.critic,)},
        env.action_size,
        # brax types the hook narrower than its own normalizer
        preprocess_observations_fn=running_statistics.normalize,  # ty: ignore[invalid-argument-type]
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        activation=jax.nn.elu,
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )


class Student:
    """brax PPO networks, optimizer and jitted unroll / update / regret."""

    def __init__(self, env: FunctionalEnv, key: jax.Array):
        obs_size = env.observation_size
        self.network = make_network(env)
        self.make_policy = ppo_networks.make_inference_fn(self.network)
        policy_key, value_key = jax.random.split(key)
        self.params = ppo_losses.PPONetworkParams(
            policy=self.network.policy_network.init(policy_key),
            value=self.network.value_network.init(value_key),
        )
        self.normalizer = running_statistics.init_state(
            {
                "state": specs.Array((obs_size.actor,), jnp.dtype("float32")),
                "privileged_state": specs.Array(
                    (obs_size.critic,), jnp.dtype("float32")
                ),
            }
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(PPO["max_grad_norm"]),
            optax.adam(PPO["learning_rate"]),
        )
        self.opt_state = self.optimizer.init(self.params)
        self.update = jax.jit(self._update)
        self.regret = jax.jit(self._regret)

    def inference_params(self) -> tuple[Any, Any, Any]:
        return (self.normalizer, self.params.policy, self.params.value)

    def _loss(self, params, normalizer, data, key):
        return ppo_losses.compute_ppo_loss(
            params,
            normalizer,
            data,
            key,
            self.network,
            entropy_cost=PPO["entropy_cost"],
            discounting=PPO["discounting"],
            gae_lambda=PPO["gae_lambda"],
            clipping_epsilon=PPO["clipping_epsilon"],
        )

    def _update(self, params, opt_state, normalizer, data, key):
        normalizer = running_statistics.update(normalizer, data.observation)
        grad_fn = jax.value_and_grad(self._loss, has_aux=True)

        def minibatch(carry, batch):
            params, opt_state, key = carry
            key, loss_key = jax.random.split(key)
            (_, metrics), grads = grad_fn(params, normalizer, batch, loss_key)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            return (optax.apply_updates(params, updates), opt_state, key), metrics

        def epoch(carry, _):
            params, opt_state, key = carry
            key, perm_key = jax.random.split(key)

            def shuffle(x):
                x = jax.random.permutation(perm_key, x)
                return x.reshape((PPO["num_minibatches"], -1, *x.shape[1:]))

            batches = jax.tree.map(shuffle, data)
            return jax.lax.scan(minibatch, (params, opt_state, key), batches)

        (params, opt_state, _), metrics = jax.lax.scan(
            epoch, (params, opt_state, key), None, PPO["num_updates_per_batch"]
        )
        return params, opt_state, normalizer, jax.tree.map(jnp.mean, metrics)

    def _regret(self, params, normalizer, data):
        value_apply = self.network.value_network.apply
        data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), data)
        values = value_apply(normalizer, params.value, data.observation)
        last_obs = jax.tree.map(lambda x: x[-1], data.next_observation)
        bootstrap = value_apply(normalizer, params.value, last_obs)
        truncation = data.extras["state_extras"]["truncation"]
        _, advantages = ppo_losses.compute_gae(
            truncation=truncation,
            termination=(1 - data.discount) * (1 - truncation),
            rewards=data.reward,
            values=values,
            bootstrap_value=bootstrap,
            lambda_=PPO["gae_lambda"],
            discount=PPO["discounting"],
        )
        return jnp.maximum(advantages, 0.0).mean(axis=0)


def make_unroll(env: FunctionalEnv, make_policy: Any, dr_in_axes: Any):
    def step_one(state, action, hfield, dr_model):
        _, state, reward, done, info = env.step(
            jax.random.key(0), state, action, hfield, dr_model
        )
        return state, reward, done, info["truncation"], info["diverged"]

    # the arena's hfield_data is shared by every env (warp needs it unbatched)
    v_step = jax.vmap(step_one, in_axes=(0, 0, None, dr_in_axes))

    @functools.partial(jax.jit, static_argnames=("deterministic", "length"))
    def unroll(inference, state, key, hfield, dr_model, deterministic, length):
        policy = make_policy(inference, deterministic=deterministic)

        def step(carry, _):
            state, key = carry
            key, act_key = jax.random.split(key)
            action, extras = policy(state.obs, act_key)
            cmd_vx = state.info["command"][:, 0]
            next_state, reward, done, truncation, diverged = v_step(
                state, action, hfield, dr_model
            )
            transition = types.Transition(
                observation=state.obs,
                action=action,
                reward=reward,
                discount=1.0 - done,
                next_observation=next_state.obs,
                # brax types extras as an array; a nested dict is what it reads
                extras={  # ty: ignore[invalid-argument-type]
                    "policy_extras": extras,
                    "state_extras": {"truncation": truncation},
                    "log": {
                        "fell": done * (1.0 - truncation),
                        "diverged": diverged,
                        "reward": reward,
                        "cmd_vx": cmd_vx,
                        "forward_vel": jax.vmap(body_velocity)(next_state.data)[:, 0],
                        "xy": next_state.data.qpos[:, :2],
                    },
                },
            )
            return (next_state, key), transition

        (state, _), data = jax.lax.scan(step, (state, key), None, length)
        return state, jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), data)

    return unroll


class LevelBuffer:
    """Highest-regret levels, sampled rank-prioritized with staleness."""

    def __init__(self, capacity: int, temperature: float = 0.3, staleness: float = 0.3):
        self.capacity = capacity
        self.temperature = temperature
        self.staleness = staleness
        self.levels: list[np.ndarray] = []
        self.scores = np.zeros(0)
        self.last_seen = np.zeros(0)
        self.roughness = np.zeros(0)

    def __len__(self) -> int:
        return len(self.levels)

    def add(
        self, levels: np.ndarray, scores: np.ndarray, now: int, roughness: np.ndarray
    ) -> None:
        self.levels.extend(levels)
        self.scores = np.concatenate([self.scores, scores])
        self.last_seen = np.concatenate([self.last_seen, np.full(len(scores), now)])
        self.roughness = np.concatenate([self.roughness, roughness])
        if len(self.levels) > self.capacity:
            keep = np.argsort(self.scores)[-self.capacity :]
            self.levels = [self.levels[i] for i in keep]
            self.scores = self.scores[keep]
            self.last_seen = self.last_seen[keep]
            self.roughness = self.roughness[keep]

    def sample(self, rng: np.random.Generator, n: int, now: int) -> np.ndarray:
        ranks = np.empty(len(self)).astype(int)
        ranks[np.argsort(-self.scores)] = np.arange(1, len(self) + 1)
        score_p = (1.0 / ranks) ** (1.0 / self.temperature)
        score_p /= score_p.sum()
        stale = now - self.last_seen + 1e-6
        stale_p = stale / stale.sum()
        p = (1 - self.staleness) * score_p + self.staleness * stale_p
        return rng.choice(len(self), size=n, p=p)

    def update(self, idx: np.ndarray, scores: np.ndarray, now: int) -> None:
        self.scores[idx] = scores
        self.last_seen[idx] = now


def min_travel(robot: Robot, wojtek_m: float) -> float:
    return wojtek_m * ROBOT_SCALE[robot]


def run_levels(
    ctx: dict[str, Any],
    levels: np.ndarray,
    train: bool,
    student: "Student | None" = None,
) -> dict[str, Any]:
    """One outer iteration on `levels` (one per lane). Returns lane regret."""
    env, cfg = ctx["env"], ctx["cfg"]
    student = student or ctx["student"]
    key = ctx["next_key"]()
    rendered = ctx["space"].render(levels)
    layout: TileLayout = ctx["layout"]
    hfield = ctx["assemble"](jnp.asarray(rendered))
    tile_xy = layout.tile_xy(cfg.num_lanes, cfg.envs_per_lane)
    n = cfg.num_lanes * cfg.envs_per_lane
    dr_model, _ = env.sample_dr(ctx["next_key"](), n)
    reset_keys = jax.random.split(key, n)
    _, state = ctx["reset"](reset_keys, hfield, dr_model, tile_xy)
    start_xy = state.data.qpos[:, :2]
    regret = jnp.zeros(n)
    episode_return = jnp.zeros(n)
    fell_any = jnp.zeros(n)
    leaked = jnp.zeros(n, bool)
    logs = []
    for _ in range(cfg.level_steps // PPO["unroll_length"]):
        state, data = ctx["unroll"](
            student.inference_params(),
            state,
            ctx["next_key"](),
            hfield,
            dr_model,
            deterministic=False,
            length=PPO["unroll_length"],
        )
        if train:
            student.params, student.opt_state, student.normalizer, _ = student.update(
                student.params,
                student.opt_state,
                student.normalizer,
                data,
                ctx["next_key"](),
            )
        regret = regret + student.regret(student.params, student.normalizer, data)
        episode_return = episode_return + data.reward.sum(axis=1)
        fell_any = jnp.maximum(fell_any, data.extras["log"]["fell"].max(axis=1))
        xy = data.extras["log"]["xy"]
        reads = layout.leaked(xy, tile_xy[:, None], env.travel.sensing)
        leaked = leaked | reads.any(axis=1)
        logs.append(jax.tree.map(jnp.mean, data.extras["log"]))
    lane_regret = np.asarray(regret.reshape(cfg.num_lanes, -1).mean(axis=1))
    lane_returns = np.asarray(episode_return.reshape(cfg.num_lanes, -1))
    travel = jnp.linalg.norm(state.data.qpos[:, :2] - start_xy, axis=-1)
    threshold = min_travel(cfg.robot.name, cfg.min_travel)
    success = np.asarray((fell_any == 0) & (travel >= threshold), dtype=float)
    lane_success = success.reshape(cfg.num_lanes, -1).mean(axis=1)
    lane_regret = lane_regret / (cfg.level_steps // PPO["unroll_length"])
    # learnability (SFL): highest where the student sometimes succeeds
    learnability = lane_success * (1.0 - lane_success)
    log = jax.tree.map(lambda *x: float(np.mean(x)), *logs)
    return {
        "regret": lane_regret,
        "score": learnability if cfg.mode.score == Score.learnability else lane_regret,
        "lane_returns": lane_returns,
        "train_success": float(success.mean()),
        "learnability_mean": float(learnability.mean()),
        "falls_per_step": log["fell"],
        # env steps that diverged and were ended (FunctionalEnv._end_diverged)
        "diverged_per_step": log["diverged"],
        "reward_per_step": log["reward"],
        "cmd_vx": log["cmd_vx"],
        "forward_vel": log["forward_vel"],
        "travel": float(travel.mean()),
        # envs whose feet or height scan reached another lane's level
        "tile_leak_frac": float(leaked.mean()),
        "lane_roughness": mean_abs_height(ctx["space"].config, rendered),
        "levels": levels,
    }


def evaluate(ctx: dict[str, Any], rng: np.random.Generator) -> dict[str, float]:
    cfg, out = ctx["cfg"], {}
    per_chunk = min(cfg.eval_levels, ctx["layout"].n_tiles)
    threshold = min_travel(cfg.robot.name, cfg.min_travel)
    for difficulty in EVAL_DIFFICULTIES:
        levels = ctx["space"].scaled(rng, cfg.eval_levels, difficulty)
        rendered = jnp.asarray(ctx["space"].render(levels))
        chunks = [
            eval_chunk(ctx, rendered[start : start + per_chunk])
            for start in range(0, cfg.eval_levels, per_chunk)
        ]
        fell, travel, forward_vel, cmd_vx = (np.concatenate(x) for x in zip(*chunks))
        tag = f"eval/d{difficulty}"
        out[f"{tag}/success_rate"] = float(((fell == 0) & (travel >= threshold)).mean())
        out[f"{tag}/fall_rate"] = float(fell.mean())
        out[f"{tag}/mean_travel"] = float(travel.mean())
        out[f"{tag}/forward_vel"] = float(forward_vel.mean())
        out[f"{tag}/cmd_vx"] = float(cmd_vx.mean())
    return out


def eval_chunk(ctx: dict[str, Any], rendered: jax.Array) -> tuple[np.ndarray, ...]:
    """Two deterministic episodes per level, levels on the arena's tiles:
    per env fell, travel, mean forward and commanded velocity."""
    hfield = ctx["assemble"](rendered)
    tile_xy = ctx["layout"].tile_xy(rendered.shape[0], 2)
    keys = jax.random.split(ctx["next_key"](), tile_xy.shape[0])
    _, state = ctx["eval_reset"](keys, hfield, tile_xy)
    start_xy = state.data.qpos[:, :2]
    state, data = ctx["eval_unroll"](
        ctx["student"].inference_params(),
        state,
        ctx["next_key"](),
        hfield,
        None,
        deterministic=True,
        length=ctx["cfg"].level_steps,
    )
    log = data.extras["log"]
    travel = jnp.linalg.norm(log["xy"][:, -1] - start_xy, axis=-1)
    return (
        np.asarray(log["fell"].max(axis=1)),
        np.asarray(travel),
        np.asarray(log["forward_vel"].mean(axis=1)),
        np.asarray(log["cmd_vx"].mean(axis=1)),
    )


def level_composition(space: LevelSpace, levels: np.ndarray) -> dict[str, float]:
    """Mean physical parameter values of the levels the student ran on."""
    if not isinstance(space, ParamSpace):
        return {}
    physical = [space.physical(level) for level in levels]
    out = {}
    for p in PARAMS:
        values = np.array([ph[p.name] for ph in physical])
        out[f"level/{p.name}"] = float(values.mean())
        if p.low < 0.0:
            out[f"level/{p.name}_abs"] = float(np.abs(values).mean())
    distance = np.abs(levels[:, : len(PARAMS)] - space.flat)
    out["level/genome_dist"] = float(distance.mean())
    return out


def easy_levels(ctx: dict[str, Any], n: int) -> np.ndarray:
    return ctx["space"].easy(ctx["rng"], n, ctx["cfg"].mode.generator_scale)


def dr_levels(ctx: dict[str, Any], curriculum: Curriculum) -> np.ndarray:
    n, rng, space = ctx["cfg"].num_lanes, ctx["rng"], ctx["space"]
    if curriculum == Curriculum.dr_easy:
        return easy_levels(ctx, n)
    if curriculum == Curriculum.dr_mixed:
        # a per-level difficulty in [0, 1]: covers the easy levels too
        return np.concatenate(
            [space.scaled(rng, 1, d) for d in rng.uniform(0.0, 1.0, n)]
        )
    return space.uniform(rng, n)


def mutate(ctx: dict[str, Any], level: np.ndarray) -> np.ndarray:
    cfg = ctx["cfg"]
    # symmetric local edits smooth as often as they roughen; "grow" also
    # scales the whole genome away from flat so each generation has a direction
    grow_max = cfg.mode.grow_max if cfg.mode.mutation == Mutation.grow else None
    return ctx["space"].mutate(level, ctx["rng"], grow_max)


def buffer_stats(buffer: LevelBuffer) -> dict[str, float]:
    if len(buffer) < 2:
        return {}
    rough = buffer.roughness
    score_ranks = np.argsort(np.argsort(buffer.scores))
    rough_ranks = np.argsort(np.argsort(rough))
    corr = np.corrcoef(score_ranks, rough_ranks)[0, 1]
    return {
        "buffer_roughness_mean": float(rough.mean()),
        "buffer_roughness_p90": float(np.percentile(rough, 90)),
        "buffer_score_roughness_spearman": float(corr) if np.isfinite(corr) else 0.0,
    }


def accel_iteration(
    ctx: dict[str, Any], buffer: LevelBuffer, it: int
) -> dict[str, Any]:
    cfg, rng = ctx["cfg"], ctx["rng"]
    if len(buffer) >= cfg.num_lanes and rng.random() < cfg.mode.replay_prob:
        idx = buffer.sample(rng, cfg.num_lanes, it)
        levels = np.stack([buffer.levels[i] for i in idx])
        out = run_levels(ctx, levels, train=True)
        buffer.update(idx, out["score"], it)
        if cfg.mode.curriculum == Curriculum.accel:
            ctx["children"] = np.stack([mutate(ctx, level) for level in levels])
        return {**out, "replay": 1, **buffer_stats(buffer)}
    fresh = cfg.num_lanes - len(ctx["children"])
    if cfg.mode.curriculum == Curriculum.plr:
        new = ctx["space"].uniform(rng, fresh)
    elif cfg.mode.generator == Generator.easy:
        new = easy_levels(ctx, fresh)
    else:
        new = dr_levels(ctx, Curriculum.dr_mixed)[:fresh]
    levels = np.concatenate([ctx["children"], new]) if len(ctx["children"]) else new
    ctx["children"] = np.zeros((0, *ctx["space"].shape))
    out = run_levels(ctx, levels, train=False)
    buffer.add(levels, out["score"], it, out["lane_roughness"])
    return {**out, "replay": 0, **buffer_stats(buffer)}


class Adversary:
    """PAIRED's learned level designer, trained by single-step PPO on regret."""

    def __init__(self, mode: ModeConfig, shape: tuple[int, ...], key: jax.Array):
        self.latent_dim = mode.latent_dim
        self.generator = LevelGenerator(
            latent_dim=mode.latent_dim, dense_dim=128, shape=shape, key=key
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.5), optax.adam(mode.gen_learning_rate)
        )
        self.opt_state = self.optimizer.init(eqx.filter(self.generator, eqx.is_array))
        self.entropy_coef = mode.gen_entropy_coef
        self.sample = eqx.filter_jit(lambda g, z, keys: jax.vmap(g)(z, keys)[:2])
        self.step = eqx.filter_jit(self._step)

    def _loss(self, generator, z, theta, old_log_prob, advantages):
        new_log_prob, entropy = jax.vmap(generator.evaluate)(z, theta)
        ratio = jnp.exp(jnp.clip(new_log_prob - old_log_prob, -20.0, 20.0))
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        clipped = jnp.clip(ratio, 0.8, 1.2) * advantages
        policy_loss = -jnp.mean(jnp.minimum(ratio * advantages, clipped))
        return policy_loss - self.entropy_coef * jnp.mean(entropy)

    def _step(self, generator, opt_state, z, theta, old_log_prob, advantages):
        loss, grads = eqx.filter_value_and_grad(self._loss)(
            generator, z, theta, old_log_prob, advantages
        )
        updates, opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(generator, eqx.is_array)
        )
        generator = eqx.apply_updates(generator, updates)
        log_std = jnp.clip(generator.log_std, -5.0, 2.0)
        return eqx.tree_at(lambda g: g.log_std, generator, log_std), opt_state, loss

    def update(self, z, theta, old_log_prob, regret, epochs: int = 5) -> float:
        advantages = regret - regret.mean()
        loss = jnp.zeros(())
        for _ in range(epochs):
            self.generator, self.opt_state, loss = self.step(
                self.generator, self.opt_state, z, theta, old_log_prob, advantages
            )
        return float(loss)


def paired_iteration(ctx: dict[str, Any]) -> dict[str, Any]:
    cfg, adversary = ctx["cfg"], ctx["adversary"]
    z = jax.random.normal(ctx["next_key"](), (cfg.num_lanes, adversary.latent_dim))
    keys = jax.random.split(ctx["next_key"](), cfg.num_lanes)
    theta, log_prob = adversary.sample(adversary.generator, z, keys)
    levels = ctx["space"].from_generated(np.asarray(theta), ctx["rng"])
    out = run_levels(ctx, levels, train=True)
    antagonist = run_levels(ctx, levels, train=True, student=ctx["antagonist"])
    prot_mean = out["lane_returns"].mean(axis=1)
    antag_max = antagonist["lane_returns"].max(axis=1)
    # PAIRED regret: what the antagonist shows is possible on the level, minus
    # what the protagonist averages there
    paired_regret = np.maximum(antag_max - prot_mean, 0.0)
    gen_loss = adversary.update(z, theta, log_prob, jnp.asarray(paired_regret))
    return {
        **out,
        "replay": 1,
        "paired_regret_mean": float(paired_regret.mean()),
        "prot_return": float(prot_mean.mean()),
        "antag_return": float(antagonist["lane_returns"].mean()),
        "antag_forward_vel": antagonist["forward_vel"],
        "gen_loss": gen_loss,
    }


def build(cfg: TrainConfig) -> dict[str, Any]:
    space_cfg = cfg.level_space
    config = hfield_config(cfg.robot.name, cfg.arena)
    space = make_level_space(
        space_cfg.kind,
        config,
        n_edits=space_cfg.n_edits,
        edit_scale=space_cfg.edit_scale,
        robot_scale=ROBOT_SCALE[cfg.robot.name],
    )
    check_wojtek_arena(cfg.robot.name, config)
    layout = TileLayout(config)
    if cfg.num_lanes > layout.n_tiles:
        raise ValueError(
            f"num_lanes={cfg.num_lanes} exceeds the arena's {layout.n_tiles} "
            "tiles, one level each; rebuild it with a larger arena.tiles"
        )
    num_envs = cfg.num_lanes * cfg.envs_per_lane
    robot, backend = cfg.robot.name, cfg.backend
    env = make_env(robot, backend, config, num_envs=num_envs, train=True)
    reach = env.travel.reach(cfg.level_steps, env.travel.sensing)
    layout.check_reach(reach, f"level_steps={cfg.level_steps}")
    eval_levels = min(cfg.eval_levels, layout.n_tiles)
    eval_env = make_env(robot, backend, config, num_envs=2 * eval_levels, train=False)
    key_state = {"key": jax.random.key(cfg.seed)}

    def next_key() -> jax.Array:
        key_state["key"], key = jax.random.split(key_state["key"])
        return key

    student = Student(env, next_key())
    paired = cfg.mode.curriculum == Curriculum.paired
    antagonist = Student(env, next_key()) if paired else None
    adversary = Adversary(cfg.mode, space.gen_shape, next_key()) if paired else None
    _, dr_in_axes = env.sample_dr(jax.random.key(0), num_envs)
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None, dr_in_axes, 0)))
    eval_reset = jax.jit(
        jax.vmap(lambda k, h, t: eval_env.reset(k, h, tile_xy=t), in_axes=(0, None, 0))
    )
    return {
        "cfg": cfg,
        "env": env,
        "student": student,
        "antagonist": antagonist,
        "adversary": adversary,
        "space": space,
        "layout": layout,
        "assemble": jax.jit(layout.assemble),
        "rng": np.random.default_rng(cfg.seed),
        "next_key": next_key,
        "reset": reset,
        "eval_reset": eval_reset,
        "unroll": make_unroll(env, student.make_policy, dr_in_axes),
        "eval_unroll": make_unroll(eval_env, student.make_policy, None),
        "children": np.zeros((0, *space.shape)),
    }


@hydra.main(config_path=CONF_DIR, config_name="train", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    train(to_dataclass(dict_cfg, TrainConfig))


def train(cfg: TrainConfig) -> None:
    if cfg.level_steps % PPO["unroll_length"]:
        raise ValueError(
            f"level_steps={cfg.level_steps} must be a multiple of "
            f"unroll_length={PPO['unroll_length']}"
        )
    launch = time.time()
    run = wandb.init(
        entity=cfg.entity or None,
        project=cfg.project,
        name=cfg.name,
        tags=["brax-ued", cfg.mode.curriculum, cfg.robot.name, *cfg.tags],
        config={**as_dict(cfg), "ppo": PPO},
    )
    ctx = build(cfg)
    buffer = LevelBuffer(cfg.mode.buffer_size)
    steps_per_iteration = cfg.num_lanes * cfg.envs_per_lane * cfg.level_steps
    # the PAIRED antagonist simulates (and trains on) the same levels again
    sims_per_iteration = 2 if cfg.mode.curriculum == Curriculum.paired else 1
    env_steps, it, start = 0, 0, time.time()
    sim_steps, student_train_steps = 0, 0
    while env_steps < cfg.num_timesteps:
        out: dict[str, Any]
        if cfg.mode.curriculum in (Curriculum.accel, Curriculum.plr):
            out = accel_iteration(ctx, buffer, it)
        elif cfg.mode.curriculum == Curriculum.paired:
            out = paired_iteration(ctx)
        else:
            levels = dr_levels(ctx, cfg.mode.curriculum)
            out = {**run_levels(ctx, levels, train=True), "replay": 1}
        env_steps += steps_per_iteration
        sim_steps += sims_per_iteration * steps_per_iteration
        # replay == 1 exactly when the student's PPO updated on this data
        student_train_steps += out["replay"] * steps_per_iteration
        regret = np.asarray(out.pop("regret"))
        out["level_roughness_m"] = float(np.mean(out.pop("lane_roughness")))
        out.update(level_composition(ctx["space"], out.pop("levels")))
        for key in ("lane_returns", "score"):
            out.pop(key)
        log = {
            "env_steps": env_steps,
            "iteration": it,
            **out,
            "regret_mean": float(regret.mean()),
            "regret_max": float(regret.max()),
            "buffer_size": len(buffer),
            "buffer_score_mean": float(buffer.scores.mean()) if len(buffer) else 0.0,
            "steps_per_sec": env_steps / (time.time() - start),
            "sim_steps": sim_steps,
            "student_train_steps": student_train_steps,
            "elapsed_s": time.time() - launch,
        }
        if it % cfg.eval_every == 0 or env_steps >= cfg.num_timesteps:
            log.update(evaluate(ctx, np.random.default_rng(1234)))
        run.log(log)
        print(
            {k: round(v, 3) if isinstance(v, float) else v for k, v in log.items()},
            flush=True,
        )
        it += 1
    if cfg.checkpoint_dir:
        path = Path(cfg.checkpoint_dir) / f"{cfg.name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        brax_model.save_params(str(path), ctx["student"].inference_params())
    run.summary["final/elapsed_h"] = (time.time() - launch) / 3600
    run.summary["final/sim_steps"] = sim_steps
    run.summary["final/student_train_steps"] = student_train_steps
    run.finish()


if __name__ == "__main__":
    main()
