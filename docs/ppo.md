# PPO

The on-policy learner shared by both trainers. Plain PPO lives in
`scripts/ppo.py`; ACCEL reuses the same model and helpers inside its curriculum
loop ([ued-accel.md](ued-accel.md)).

## Model — `modules/ppo_modules.py`

`A2C` is an `equinox.Module` (asymmetric actor-critic):

- `actor_mean` — MLP (`Linear → LayerNorm → act → Linear → LayerNorm → act →
  Linear`), width `dense_dim`, activation `relu` or `tanh` (config).
- `actor_log_std` — a free `action_dim` vector (state-independent std).
- `critic` — same MLP shape, scalar output, reads the privileged obs.
- Action distribution: `distrax.MultivariateNormalDiag` squashed through
  `Tanh` then scaled by `action_high` (a `distrax.Transformed`).

Calls:

- `__call__(obs, key, action=None)` → `(action, log_prob, value, entropy)`.
  Samples an action if none is passed (rollout) or scores a given one
  (PPO update). `entropy` is `-log_prob` (a per-sample proxy).
- `deterministic_action(obs)` → `action_high * tanh(actor_mean)`, used at eval.

Pytrees: `Obs(actor, critic)` and `Episode(obs, next_obs, action, log_prob,
value, entropy, reward, done)`.

## Helpers — `utils/ppo_helpers.py`

**Rollout** (data collection, no gradient):

- `policy_step` — one env step under `lax.scan`; optional `hfield_data` threads
  the ACCEL level through.
- `rollout` — reset + scan for `steps_in_episode`.
- `batch_rollout` — `vmap` over `envs_num` keys, one model, one level.

**Advantage:**

- `compute_gae` — reverse `lax.scan` GAE (`gamma`, `lam`), returns advantages +
  returns.
- `calculate_episode_advantage` — bootstraps the last value from
  `next_obs.critic[:, -1]`, vmapped over the batch.

**Loss / update:**

- `ppo_loss` — clipped surrogate (`epsilon`) + `value_coef * MSE` +
  `entropy_coef * -entropy`. Advantages are normalised per minibatch; the ratio
  is computed from a fresh forward pass vs the stored `old_log_prob` (the
  stop-gradient behaviour is analysed in
  [ued-accel.md](ued-accel.md#stop-gradient)).
- `make_step` — `eqx.filter_value_and_grad(ppo_loss)` → optimizer update. **The
  only place a gradient is taken in the whole codebase.**
- `run_ppo_epochs` — flattens the batch, loops `epochs × minibatches` with fresh
  permutations, calls `make_step`. Guards each step: `is_finite` skips a NaN/Inf
  update, and `clip_log_std` clamps `actor_log_std` to `[-5, 0.5]`.

**Eval:** `eval_step`/`eval_rollout`/`eval` roll out deterministic actions on the
default terrain and return mean summed reward across `envs_num` envs.

## Trainer script — `scripts/ppo.py`

Fixed loop for `training_params.iterations`:

1. `batch_rollout` → `calculate_episode_advantage` → `run_ppo_epochs`.
2. Log `loss`, `mean_reward`, `learning_rate`; eval every `eval_every`.
3. `save_checkpoint` every `save_every`.

Setup pieces reused by ACCEL: an `optax.chain(clip_by_global_norm, adam)` with a
`linear_schedule` LR (start→end over `iterations × epochs × minibatches`), all
hot functions wrapped in `eqx.filter_jit`, and `FunctionalEnv` for the env.

## Domain randomization — `scripts/ppo_dr.py`

Plain PPO on `wojtek_env`'s baseline physics domain randomization instead of
the nominal model. Same loop as `scripts/ppo.py`, same config (`Config`, name
`"config"`) — the only difference is `FunctionalEnv(..., domain_randomize=True)`
and a per-iteration `dr_model` fed into `batch_rollout`.

- `FunctionalEnv.sample_dr(rng, envs_num)` calls
  `wojtek_env.envs.randomize.make_domain_randomize` (`dr_cfg=None`: floor
  friction, base/link mass, one gain/kd scale — the library's baseline five
  fields) with `envs_num` independent rngs, returning `(dr_model, dr_in_axes)`
  — one randomized `mjx.Model` draw per parallel env, batched on the
  randomized fields only, plus the `jax.vmap` axis spec for that batching.
  `dr_in_axes` only depends on which fields are randomized (not on rng
  values), so it is sampled once before the training loop and bound into
  `batch_rollout` via `functools.partial`; `dr_model` is resampled every
  iteration and passed as a normal (traced) `batch_rollout` argument.
- `batch_rollout`/`rollout`/`policy_step` (`utils/ppo_helpers.py`) and
  `FunctionalEnv.reset`/`step` (`utils/env.py`) all take an optional
  `dr_model`/`dr_in_axes` pair, defaulting to `None` — a no-op for
  `scripts/ppo.py`/`scripts/accel.py`/`scripts/accel_top_k.py`. Inside the
  vmapped rollout, `FunctionalEnv._swapped_model` swaps `dr_model` into
  `_mjx_model` for the call, the same trick `_swapped_hfield` already used for
  the ACCEL heightfield (see
  [environment.md](environment.md#hfield-mutability)) — DR and hfield
  swapping compose since both just `tree_replace` fields onto the same model.
- Eval (`eval`/`eval_rollout`/`eval_step`) does not take a `dr_model`, so
  `mean_return` is always measured on the nominal (non-randomized) physics.

## Checkpoints — `utils/checkpoint.py`

`orbax` `CheckpointManager` with a `Composite` of `model` + `opt_state`.
`save_checkpoint`/`restore_checkpoint` partition the equinox model into arrays
(saved) and static (recombined on restore); `restore_checkpoint`'s
`template_opt_state` is optional — pass `None` (eval's case) to restore just
the model, skipping the optimizer item entirely.

**Naming convention.** Every trainer script declares a module-level `ALGO`
constant (`"ppo"`, `"ppo_dr"`, `"accel"`, `"accel_top_k"`) and writes to
`artifacts/<algo>/<run>/`, where `<run>` is `wandb_params.name` if set, else a
`run-<unix_ts>` fallback (`resolve_run_name`) — the same name is used for the
wandb run / jsonl `Logger` filename / checkpoint directory, so all three line
up. This keeps the four algorithms' checkpoints from colliding in one flat
`artifacts/` dir (the previous behaviour) and lets `scripts/eval.py` find
"the latest model of algo X" via `latest_run_dir` (newest directory by mtime
under `artifacts/<algo>/`) and `CheckpointManager.latest_step()`.

Each script also writes a `meta.json` sidecar (`save_meta`/`load_meta`) next
to the run's checkpoints, once, right after building the model: the `A2C`
shape (`dense_dim`, `activation`, `action_dim`, `actor_obs_dim`,
`critic_obs_dim`, `action_high`) plus the env construction args (`env_task`,
`env_overrides`, `env_seed`). This is what lets `eval.py` rebuild a template
model and `FunctionalEnv` and restore a checkpoint **without** re-running the
training script's Hydra config.

## Eval script — `scripts/eval.py`

A `pyrallis` CLI over a dataclass (`EvalConfig`, not Hydra — it needs no
training config, only the run's `meta.json`): `just eval --algo
{ppo,ppo_dr,accel,accel_top_k} [--run NAME] [--step N] [--artifacts_dir DIR]
[--envs_num N] [--steps N] [--seed N]`. `algo` is `utils/checkpoint.Algo`, a
`str` `Enum` — invalid values fail fast with a clear error instead of
`argparse`'s `choices=`. `pyrallis` also accepts `--config_path
some.yaml` to load flags from a file instead of/in addition to the CLI.

1. Resolve the run directory (`--run`, else `latest_run_dir`) and load its
   `meta.json`.
2. Rebuild `FunctionalEnv(task=meta.env_task, env_overrides=meta.env_overrides,
   seed=meta.env_seed)` and a template `A2C` from the meta shape fields.
3. `restore_checkpoint` (model-only, `--step` or latest) and run the same
   `eval` used during training (deterministic actions, nominal physics —
   `dr_model`/`hfield_data` are not part of eval regardless of `--algo`).
4. Print `mean_return`.

It is a skeleton: one metric, no video/rollout dump, no batch comparison
across runs — extend it here as evaluation needs grow.

## Levels-generalization eval — `scripts/eval_levels.py`

Same `pyrallis`/`EvalLevelsConfig` CLI shape and run-resolution as
`scripts/eval.py`, but measures generalization to freshly generated
heightfield levels instead of one fixed course: `--algo
{ppo,ppo_dr,accel,accel_top_k} [--run NAME] [--step N] [--num_levels N]
[--episodes_per_level N] [--chunk_size N] [--steps N] [--seed N] [--output
path.npz]`. Forces the `accel` terrain arena on regardless of the checkpoint's
own training arena — needs `just build-accel-arena` first. Generates
`num_levels` levels in `chunk_size`-sized batches (one `jit` compile, reused
across chunks since every chunk is the same shape), evaluates
`episodes_per_level` episodes each via `utils/ppo_helpers.eval_levels`, prints
`mean_reward`/`success_rate`, and optionally dumps the per-level
reward/terminated arrays to `--output`.

## Logging — `utils/logging.py`

`wandb` when `wandb_params.enable`, else a `Logger` that appends one JSON record
per `log()` to `logs/<name>.log` (jax/numpy arrays coerced via `.tolist()`).
`quiet_dependencies()` mutes absl/jax/orbax chatter. Both expose the same
`log`/`finish` interface, so scripts don't branch on the backend.

Config knobs for all of the above: [configuration.md](configuration.md).
