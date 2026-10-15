"""DR vs ACCEL on the heightfield benchmark, with brax's PPO as the student.

brax's own `ppo.train` fixes each env's model for the whole run, so levels
cannot change per env over training. This loop keeps brax's PPO pieces
(networks, running observation normalizer, `compute_ppo_loss`, its
mujoco_playground locomotion hyperparameters: 20-step unrolls, 4 epochs x
32 minibatches, discount 0.97) and drives `FunctionalEnv`, which takes a
per-env `hfield_data` and physics DR draw at every step.

One outer iteration = `num_lanes` levels x `envs_per_lane` envs, reset onto
their level and stepped for `level_steps` (in `unroll_length` chunks, a PPO
update after each chunk when training). Per-lane regret = mean positive GAE
advantage (positive value loss) over the iteration.

- `--mode dr`: every iteration trains on fresh uniform levels at full
  difficulty.
- `--mode accel`: ACCEL with batched lanes. Replay iterations (prob
  `replay_prob` once the buffer holds `num_lanes` levels) train on levels
  drawn rank-prioritized from the buffer, refresh their scores, and queue
  their mutations. Explore iterations do not train: they score the queued
  children plus fresh *easy* levels (control points within
  `generator_scale` of flat) and add them to the buffer, which keeps the
  highest-regret `buffer_size` levels.
- `--mode plr`: robust PLR (PLR-perp) - ACCEL's buffer and replay without
  mutation; explore iterations score fresh uniform (DR) levels.
- `--robot go1`: Playground's Go1 (`robots/go1.py`) instead of Wojtek. The
  arena, level lengths, commands and `min_travel` scale by `GO1_SCALE`.
- `--profile deploy`: w01-tek's `terrain_blind_v4_4d` env and network, so
  `export_policy.py` can ship the checkpoint to the robot (docs/deploy.md),
  on a 12x12 m arena (`accel_deploy`). `benchmark` (default) is the v10 setup.

Writes `<name>.json` beside `<name>.pkl` at launch: everything needed to
rebuild the env and network (`load_run`).

All evaluate the deterministic policy every `eval_every` iterations on
fresh uniform levels at several difficulties: success = no fall and at least
`min_travel` m from spawn after `level_steps`.
"""

import argparse
import dataclasses
import functools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from brax.io import model as brax_model
from brax.training import types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks

from config.hfield_config import HFieldConfig
from deploy.contract import TRAINING_ONLY_KEYS
from hfield.arena import check_arena
from hfield.level_space import (
    PARAMS,
    LevelSpace,
    ParamSpace,
    make_level_space,
    mean_abs_height,
)
from modules.generator import LevelGenerator
from robots.go1 import GO1_SCALE
from utils.env import FunctionalEnv, ObsSize
from utils.env_overrides import (
    accel_arena_overrides,
    deploy_dr_config,
    deploy_overrides,
    merge_overrides,
    movement_overrides,
    terrain_overrides,
    terrain_task_overrides,
)
from utils.ppo_helpers import body_velocity
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


ROBOT_SCALE = {"wojtek": 1.0, "go1": GO1_SCALE}

# the deploy command box reaches 1.2 m/s and leaves the 6x6 m benchmark patch
# within ~2.5 s of a 6 s level; coarse_n 25 keeps its ~0.5 m feature size
ARENAS: dict[str, dict[str, Any]] = {
    "benchmark": {"arena": "accel", "patch_radius": 3.0, "coarse_n": 13},
    "deploy": {"arena": "accel_deploy", "patch_radius": 6.0, "coarse_n": 25},
}
ACTIVATIONS = {"elu": jax.nn.elu, "silu": jax.nn.silu}
NETWORKS: dict[str, dict[str, Any]] = {
    "benchmark": {
        "policy_hidden_layer_sizes": [512, 256, 128],
        "value_hidden_layer_sizes": [512, 256, 128],
        "activation": "elu",
        "distribution_type": "tanh_normal",
        "policy_obs_key": "state",
        "value_obs_key": "privileged_state",
    },
    # w01-tek train.py build_ppo_params: Playground's Go1 network_factory with
    # brax's default SiLU and tanh-normal, what the robot runtime evaluates
    "deploy": {
        "policy_hidden_layer_sizes": [512, 256, 128],
        "value_hidden_layer_sizes": [512, 256, 128],
        "activation": "silu",
        "distribution_type": "tanh_normal",
        "policy_obs_key": "state",
        "value_obs_key": "privileged_state",
    },
}


def hfield_config(args: argparse.Namespace) -> HFieldConfig:
    """The arena at the robot's size: every length scales, slopes do not."""
    s = ROBOT_SCALE[args.robot]
    arena = ARENAS[args.profile]
    return HFieldConfig(
        patch_radius=arena["patch_radius"] * s,
        cell_size=HFieldConfig().cell_size * s,
        amplitude=args.amplitude * s,
        coarse_n=args.coarse_n,
        pad_radius=args.pad_radius * s,
        pad_taper=args.pad_taper * s,
        n_edits=args.n_edits,
        edit_scale=args.edit_scale,
    )


def load_env_override(path: str) -> dict[str, Any]:
    """`--env_override` file: a JSON object of TRAINING_ONLY env keys."""
    try:
        override = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"--env_override {path}: cannot read JSON ({e})") from e
    if not isinstance(override, dict):
        raise TypeError(f"--env_override {path}: expected a JSON object")
    refused = sorted(set(override) - TRAINING_ONLY_KEYS)
    if refused:
        raise ValueError(
            f"--env_override {path}: {refused} are not TRAINING_ONLY keys "
            "(src/deploy/contract.py); keys that shape the deploy contract or "
            "the actor's inputs cannot be overridden"
        )
    return override


def env_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.robot == "go1":
        return None
    arena = accel_arena_overrides(args.pad_radius, ARENAS[args.profile]["arena"])
    if args.profile == "deploy":
        # upstream's terrain block configures its tile curriculum arena; the
        # accel arena replaces it (training-only, docs/deploy.md)
        upstream = {k: v for k, v in deploy_overrides().items() if k != "terrain"}
        profile = merge_overrides(upstream, arena)
    else:
        profile = merge_overrides(
            arena, movement_overrides(), terrain_overrides(), terrain_task_overrides()
        )
    if args.env_override:
        return merge_overrides(profile, load_env_override(args.env_override))
    return profile


def git_commit() -> dict[str, Any]:
    src = Path(__file__).resolve().parents[1]

    def git(*cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(src), *cmd], capture_output=True, text=True, check=False
        )

    head = git("rev-parse", "HEAD")
    if head.returncode:
        # a frozen WCSS copy has no .git; its GIT_COMMIT file names the commit
        stamp = src.parent / "GIT_COMMIT"
        if stamp.exists():
            return {"git_commit": stamp.read_text().strip(), "git_dirty": None}
        print(
            f"warning: {src} is not a git checkout and {stamp} does not exist; "
            "the sidecar records git_commit=null",
            file=sys.stderr,
        )
        return {"git_commit": None, "git_dirty": None}
    dirty = bool(git("status", "--porcelain", "--", ".").stdout.strip())
    return {"git_commit": head.stdout.strip(), "git_dirty": dirty}


def run_spec(args: argparse.Namespace) -> dict[str, Any]:
    """Everything `make_env` / `make_network` need, JSON-serializable."""
    return {
        "profile": args.profile,
        "robot": args.robot,
        "task": "joystick",
        "arena": ARENAS[args.profile]["arena"],
        "hfield": dataclasses.asdict(hfield_config(args)),
        "level_space": args.level_space,
        "env_overrides": env_overrides(args),
        "dr_config": deploy_dr_config() if args.profile == "deploy" else None,
        "network": NETWORKS[args.profile],
        "init_from": args.init_from,
        "args": vars(args),
    }


def sidecar_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(".json")


def write_sidecar(
    checkpoint: Path, spec: dict[str, Any], obs_size: ObsSize, action_size: int
) -> Path:
    sidecar = {
        **spec,
        "obs_size": {"actor": obs_size.actor, "critic": obs_size.critic},
        "action_size": action_size,
        **git_commit(),
    }
    path = sidecar_path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar, indent=2) + "\n")
    return path


def load_run(checkpoint: Path) -> tuple[dict[str, Any], Any]:
    """A checkpoint's sidecar and params, checked against each other."""
    path = sidecar_path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(
            f"no sidecar {path} for {checkpoint}: brax_ued.py writes it beside "
            "the .pkl. For a checkpoint from before sidecars, write it with "
            f"`uv run python src/scripts/write_sidecar.py --checkpoint "
            f"{checkpoint} <the run's brax_ued.py flags>`"
        )
    spec = json.loads(path.read_text())
    params = brax_model.load_params(str(checkpoint))
    normalizer = params[0]
    got = {k: int(normalizer.mean[k].shape[-1]) for k in ("state", "privileged_state")}
    want = {
        "state": spec["obs_size"]["actor"],
        "privileged_state": spec["obs_size"]["critic"],
    }
    if got != want:
        raise ValueError(
            f"{checkpoint} has observation sizes {got}, its sidecar {path} says "
            f"{want}; the two files come from different runs"
        )
    return spec, params


def check_spec_arena(spec: dict[str, Any]) -> None:
    if spec["robot"] == "wojtek":
        kind = spec["arena"]
        config = HFieldConfig(**spec["hfield"])
        check_arena(paths.terrain_paths(kind)["spec"], config, kind)


def make_env(spec: dict[str, Any], num_envs: int, train: bool) -> FunctionalEnv:
    if spec["robot"] == "go1":
        config = HFieldConfig(**spec["hfield"])
        return FunctionalEnv(domain_randomize=train, go1_hfield=config)
    overrides = merge_overrides(
        spec["env_overrides"],
        {"sim": {"num_envs": num_envs}},
        # eval scores falls and travel; the stochastic cut only adds noise
        {} if train else {"no_progress": {"enable": False}},
    )
    return FunctionalEnv(
        task=spec["task"],
        env_overrides=overrides,
        domain_randomize=train,
        dr_config=spec["dr_config"],
    )


def make_network(
    network: dict[str, Any], obs_size: ObsSize, action_size: int
) -> ppo_networks.PPONetworks:
    kwargs = dict(network)
    activation = ACTIVATIONS[kwargs.pop("activation")]
    return ppo_networks.make_ppo_networks(
        {"state": (obs_size.actor,), "privileged_state": (obs_size.critic,)},
        action_size,
        # brax types the hook narrower than its own normalizer
        preprocess_observations_fn=running_statistics.normalize,  # ty: ignore[invalid-argument-type]
        activation=activation,
        **kwargs,
    )


def spec_network(spec: dict[str, Any]) -> ppo_networks.PPONetworks:
    size = ObsSize(spec["obs_size"]["actor"], spec["obs_size"]["critic"])
    return make_network(spec["network"], size, spec["action_size"])


class Student:
    """brax PPO networks, optimizer and jitted unroll / update / regret."""

    def __init__(
        self,
        env: FunctionalEnv,
        network: dict[str, Any],
        key: jax.Array,
        clipping_epsilon: float,
    ):
        self.clipping_epsilon = clipping_epsilon
        obs_size = env.observation_size
        self.network = make_network(network, obs_size, env.action_size)
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

    def warm_start(self, params: Any, source: str) -> None:
        """Normalizer, policy and value from a brax_ued checkpoint; the
        optimizer state starts fresh."""
        normalizer, policy, value = params
        for name, got, want in (
            ("policy", policy, self.params.policy),
            ("value", value, self.params.value),
            ("normalizer mean", normalizer.mean, self.normalizer.mean),
        ):
            got_shapes = jax.tree.map(jnp.shape, got)
            want_shapes = jax.tree.map(jnp.shape, want)
            if got_shapes != want_shapes:
                raise ValueError(
                    f"--init_from {source}: {name} shapes {got_shapes} differ from "
                    f"this run's {want_shapes}; warm start from a checkpoint of the "
                    "same profile and network"
                )
        self.normalizer = jax.tree.map(jnp.asarray, normalizer)
        self.params = ppo_losses.PPONetworkParams(
            policy=jax.tree.map(jnp.asarray, policy),
            value=jax.tree.map(jnp.asarray, value),
        )
        self.opt_state = self.optimizer.init(self.params)

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
            clipping_epsilon=self.clipping_epsilon,
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
        return state, reward, done, info["truncation"]

    v_step = jax.vmap(step_one, in_axes=(0, 0, 0, dr_in_axes))

    @functools.partial(jax.jit, static_argnames=("deterministic", "length"))
    def unroll(inference, state, key, hfield, dr_model, deterministic, length):
        policy = make_policy(inference, deterministic=deterministic)

        def step(carry, _):
            state, key = carry
            key, act_key = jax.random.split(key)
            action, extras = policy(state.obs, act_key)
            cmd = state.info["command"][:, :3]
            next_state, reward, done, truncation = v_step(
                state, action, hfield, dr_model
            )
            linvel = jax.vmap(body_velocity)(next_state.data)
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
                        "reward": reward,
                        "cmd_vx": cmd[:, 0],
                        "forward_vel": linvel[:, 0],
                        "cmd": cmd,
                        # per-term rewards as the env reports them: before
                        # reward.scales and the x dt
                        "reward_terms": {
                            k.removeprefix("reward/"): v
                            for k, v in next_state.metrics.items()
                            if k.startswith("reward/")
                        },
                        # free-joint angular qvel is body-frame: [5] = yaw rate
                        "vel": jnp.concatenate(
                            [linvel[:, :2], next_state.data.qvel[:, 5:6]], axis=-1
                        ),
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


def min_travel(robot: str, metres: float) -> float:
    return metres * ROBOT_SCALE[robot]


def run_levels(
    ctx: dict[str, Any],
    levels: np.ndarray,
    train: bool,
    student: "Student | None" = None,
) -> dict[str, Any]:
    """One outer iteration on `levels` (one per lane). Returns lane regret."""
    env, args = ctx["env"], ctx["args"]
    student = student or ctx["student"]
    key = ctx["next_key"]()
    rendered = ctx["space"].render(levels)
    hfield = jnp.repeat(jnp.asarray(rendered), args.envs_per_lane, axis=0)
    dr_model, _ = env.sample_dr(ctx["next_key"](), args.num_lanes * args.envs_per_lane)
    reset_keys = jax.random.split(key, hfield.shape[0])
    _, state = ctx["reset"](reset_keys, hfield, dr_model)
    start_xy = state.data.qpos[:, :2]
    regret = jnp.zeros(hfield.shape[0])
    episode_return = jnp.zeros(hfield.shape[0])
    fell_any = jnp.zeros(hfield.shape[0])
    logs = []
    for _ in range(args.level_steps // PPO["unroll_length"]):
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
        logs.append(jax.tree.map(jnp.mean, data.extras["log"]))
    lane_regret = np.asarray(regret.reshape(args.num_lanes, -1).mean(axis=1))
    lane_returns = np.asarray(episode_return.reshape(args.num_lanes, -1))
    travel = jnp.linalg.norm(state.data.qpos[:, :2] - start_xy, axis=-1)
    reached = travel >= min_travel(args.robot, args.min_travel)
    success = np.asarray((fell_any == 0) & reached, dtype=float)
    lane_success = success.reshape(args.num_lanes, -1).mean(axis=1)
    lane_regret = lane_regret / (args.level_steps // PPO["unroll_length"])
    # learnability (SFL): highest where the student sometimes succeeds
    learnability = lane_success * (1.0 - lane_success)
    log = jax.tree.map(lambda *x: float(np.mean(x)), *logs)
    return {
        "regret": lane_regret,
        "score": learnability if args.score == "learnability" else lane_regret,
        "lane_returns": lane_returns,
        "train_success": float(success.mean()),
        "learnability_mean": float(learnability.mean()),
        "falls_per_step": log["fell"],
        "reward_per_step": log["reward"],
        "cmd_vx": log["cmd_vx"],
        "forward_vel": log["forward_vel"],
        **{f"reward_term/{k}": v for k, v in log["reward_terms"].items()},
        "travel": float(travel.mean()),
        "lane_roughness": mean_abs_height(ctx["space"].config, rendered),
        "levels": levels,
    }


def evaluate(ctx: dict[str, Any], rng: np.random.Generator) -> dict[str, float]:
    args, out = ctx["args"], {}
    for difficulty in EVAL_DIFFICULTIES:
        levels = ctx["space"].scaled(rng, args.eval_levels, difficulty)
        hfield = jnp.repeat(jnp.asarray(ctx["space"].render(levels)), 2, axis=0)
        keys = jax.random.split(ctx["next_key"](), hfield.shape[0])
        _, state = ctx["eval_reset"](keys, hfield)
        start_xy = state.data.qpos[:, :2]
        state, data = ctx["eval_unroll"](
            ctx["student"].inference_params(),
            state,
            ctx["next_key"](),
            hfield,
            None,
            deterministic=True,
            length=args.level_steps,
        )
        log = data.extras["log"]
        fell = np.asarray(log["fell"].max(axis=1))
        travel = np.asarray(jnp.linalg.norm(log["xy"][:, -1] - start_xy, axis=-1))
        tag = f"eval/d{difficulty}"
        out[f"{tag}/success_rate"] = float(
            ((fell == 0) & (travel >= min_travel(args.robot, args.min_travel))).mean()
        )
        out[f"{tag}/fall_rate"] = float(fell.mean())
        out[f"{tag}/mean_travel"] = float(travel.mean())
        out[f"{tag}/forward_vel"] = float(log["forward_vel"].mean())
        out[f"{tag}/cmd_vx"] = float(log["cmd_vx"].mean())
    return out


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
    return ctx["space"].easy(ctx["rng"], n, ctx["args"].generator_scale)


def dr_levels(ctx: dict[str, Any], mode: str) -> np.ndarray:
    n, rng, space = ctx["args"].num_lanes, ctx["rng"], ctx["space"]
    if mode == "dr_easy":
        return easy_levels(ctx, n)
    if mode == "dr_mixed":
        # a per-level difficulty in [0, 1]: covers the easy levels too
        return np.concatenate(
            [space.scaled(rng, 1, d) for d in rng.uniform(0.0, 1.0, n)]
        )
    return space.uniform(rng, n)


def mutate(ctx: dict[str, Any], level: np.ndarray) -> np.ndarray:
    args = ctx["args"]
    # symmetric local edits smooth as often as they roughen; "grow" also
    # scales the whole genome away from flat so each generation has a direction
    grow_max = args.grow_max if args.mutation == "grow" else None
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
    args, rng = ctx["args"], ctx["rng"]
    if len(buffer) >= args.num_lanes and rng.random() < args.replay_prob:
        idx = buffer.sample(rng, args.num_lanes, it)
        levels = np.stack([buffer.levels[i] for i in idx])
        out = run_levels(ctx, levels, train=True)
        buffer.update(idx, out["score"], it)
        if args.mode == "accel":
            ctx["children"] = np.stack([mutate(ctx, level) for level in levels])
        return {**out, "replay": 1, **buffer_stats(buffer)}
    fresh = args.num_lanes - len(ctx["children"])
    if args.mode == "plr":
        new = ctx["space"].uniform(rng, fresh)
    elif args.generator == "easy":
        new = easy_levels(ctx, fresh)
    else:
        new = dr_levels(ctx, "dr_mixed")[:fresh]
    levels = np.concatenate([ctx["children"], new]) if len(ctx["children"]) else new
    ctx["children"] = np.zeros((0, *ctx["space"].shape))
    out = run_levels(ctx, levels, train=False)
    buffer.add(levels, out["score"], it, out["lane_roughness"])
    return {**out, "replay": 0, **buffer_stats(buffer)}


class Adversary:
    """PAIRED's learned level designer, trained by single-step PPO on regret."""

    def __init__(
        self, args: argparse.Namespace, shape: tuple[int, ...], key: jax.Array
    ):
        self.latent_dim = args.latent_dim
        self.generator = LevelGenerator(
            latent_dim=args.latent_dim, dense_dim=128, shape=shape, key=key
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.5), optax.adam(args.gen_learning_rate)
        )
        self.opt_state = self.optimizer.init(eqx.filter(self.generator, eqx.is_array))
        self.entropy_coef = args.gen_entropy_coef
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
    args, adversary = ctx["args"], ctx["adversary"]
    z = jax.random.normal(ctx["next_key"](), (args.num_lanes, adversary.latent_dim))
    keys = jax.random.split(ctx["next_key"](), args.num_lanes)
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


def build(args: argparse.Namespace) -> dict[str, Any]:
    spec = run_spec(args)
    check_spec_arena(spec)
    space = make_level_space(
        args.level_space,
        HFieldConfig(**spec["hfield"]),
        n_edits=args.n_edits,
        edit_scale=args.edit_scale,
        robot_scale=ROBOT_SCALE[args.robot],
    )
    num_envs = args.num_lanes * args.envs_per_lane
    env = make_env(spec, num_envs, train=True)
    eval_env = make_env(spec, 2 * args.eval_levels, train=False)
    if args.checkpoint_dir:
        checkpoint = Path(args.checkpoint_dir) / f"{args.name}.pkl"
        path = write_sidecar(checkpoint, spec, env.observation_size, env.action_size)
        print(f"wrote {path}", flush=True)
    key_state = {"key": jax.random.key(args.seed)}

    def next_key() -> jax.Array:
        key_state["key"], key = jax.random.split(key_state["key"])
        return key

    student = Student(env, spec["network"], next_key(), args.clipping_epsilon)
    if args.init_from:
        init_spec, params = load_run(Path(args.init_from))
        if init_spec["network"] != spec["network"]:
            raise ValueError(
                f"--init_from {args.init_from}: network {init_spec['network']} "
                f"differs from this run's {spec['network']}"
            )
        student.warm_start(params, args.init_from)
    paired = args.mode == "paired"
    antagonist = (
        Student(env, spec["network"], next_key(), args.clipping_epsilon)
        if paired
        else None
    )
    adversary = Adversary(args, space.gen_shape, next_key()) if paired else None
    _, dr_in_axes = env.sample_dr(jax.random.key(0), num_envs)
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, 0, dr_in_axes)))
    eval_reset = jax.jit(jax.vmap(lambda k, h: eval_env.reset(k, h)))
    return {
        "args": args,
        "env": env,
        "student": student,
        "antagonist": antagonist,
        "adversary": adversary,
        "space": space,
        "rng": np.random.default_rng(args.seed),
        "next_key": next_key,
        "reset": reset,
        "eval_reset": eval_reset,
        "unroll": make_unroll(env, student.make_policy, dr_in_axes),
        "eval_unroll": make_unroll(eval_env, student.make_policy, None),
        "children": np.zeros((0, *space.shape)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        required=True,
        choices=["dr", "dr_easy", "dr_mixed", "accel", "plr", "paired"],
    )
    p.add_argument("--profile", default="benchmark", choices=list(ARENAS))
    p.add_argument("--robot", default="wojtek", choices=list(ROBOT_SCALE))
    p.add_argument("--name", required=True)
    p.add_argument("--entity", default="")
    p.add_argument("--project", default="wojtyla")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_timesteps", type=int, default=300_000_000)
    p.add_argument("--num_lanes", type=int, default=256)
    p.add_argument("--envs_per_lane", type=int, default=32)
    p.add_argument("--level_steps", type=int, default=300)
    p.add_argument("--amplitude", type=float, default=0.3)
    p.add_argument("--level_space", default="grid", choices=["grid", "param"])
    p.add_argument(
        "--coarse_n", type=int, default=None, help="default: the profile's arena"
    )
    p.add_argument("--pad_radius", type=float, default=0.55)
    p.add_argument("--pad_taper", type=float, default=0.2)
    p.add_argument("--replay_prob", type=float, default=0.8)
    p.add_argument("--buffer_size", type=int, default=4000)
    p.add_argument("--generator_scale", type=float, default=0.1)
    p.add_argument("--n_edits", type=int, default=8)
    p.add_argument("--edit_scale", type=float, default=0.15)
    p.add_argument("--mutation", default="local", choices=["local", "grow"])
    p.add_argument("--grow_max", type=float, default=1.3)
    p.add_argument("--score", default="pvl", choices=["pvl", "learnability"])
    p.add_argument("--generator", default="easy", choices=["easy", "mixed"])
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--gen_learning_rate", type=float, default=1e-4)
    p.add_argument("--gen_entropy_coef", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--eval_levels", type=int, default=512)
    p.add_argument("--min_travel", type=float, default=1.0)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument(
        "--env_override",
        default=None,
        help="JSON file of TRAINING_ONLY env keys merged onto the profile's",
    )
    p.add_argument("--clipping_epsilon", type=float, default=PPO["clipping_epsilon"])
    p.add_argument(
        "--init_from",
        default=None,
        help="brax_ued .pkl (sidecar beside it) to start the student from",
    )
    args = p.parse_args(argv)
    if args.profile == "deploy" and args.robot != "wojtek":
        p.error("--profile deploy is Wojtek's robot setup; use --robot wojtek")
    if args.env_override:
        if args.robot == "go1":
            p.error("--env_override edits Wojtek's env config; Go1 has none")
        try:
            load_env_override(args.env_override)
        except (ValueError, TypeError) as e:
            p.error(str(e))
    if args.coarse_n is None:
        args.coarse_n = ARENAS[args.profile]["coarse_n"]
    return args


def main() -> None:
    args = parse_args()
    if args.level_steps % PPO["unroll_length"]:
        raise ValueError(
            f"--level_steps={args.level_steps} must be a multiple of "
            f"unroll_length={PPO['unroll_length']}"
        )
    launch = time.time()
    run = wandb.init(
        entity=args.entity or None,
        project=args.project,
        name=args.name,
        tags=["brax-ued", args.mode, args.robot],
        config={
            **vars(args),
            "ppo": {**PPO, "clipping_epsilon": args.clipping_epsilon},
        },
    )
    ctx = build(args)
    buffer = LevelBuffer(args.buffer_size)
    steps_per_iteration = args.num_lanes * args.envs_per_lane * args.level_steps
    # the PAIRED antagonist simulates (and trains on) the same levels again
    sims_per_iteration = 2 if args.mode == "paired" else 1
    env_steps, it, start = 0, 0, time.time()
    sim_steps, student_train_steps = 0, 0
    while env_steps < args.num_timesteps:
        out: dict[str, Any]
        if args.mode in ("accel", "plr"):
            out = accel_iteration(ctx, buffer, it)
        elif args.mode == "paired":
            out = paired_iteration(ctx)
        else:
            levels = dr_levels(ctx, args.mode)
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
        if it % args.eval_every == 0 or env_steps >= args.num_timesteps:
            log.update(evaluate(ctx, np.random.default_rng(1234)))
        run.log(log)
        print(
            {k: round(v, 3) if isinstance(v, float) else v for k, v in log.items()},
            flush=True,
        )
        it += 1
    if args.checkpoint_dir:
        path = Path(args.checkpoint_dir) / f"{args.name}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        brax_model.save_params(str(path), ctx["student"].inference_params())
    run.summary["final/elapsed_h"] = (time.time() - launch) / 3600
    run.summary["final/sim_steps"] = sim_steps
    run.summary["final/student_train_steps"] = student_train_steps
    run.finish()


if __name__ == "__main__":
    main()
