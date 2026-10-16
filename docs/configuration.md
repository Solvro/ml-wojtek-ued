# Configuration

Hydra structured configs (`src/config/`), registered with the `ConfigStore` and
selected by name in each script's `@hydra.main`:

- `scripts/ppo.py` → `Config` (name `"config"`, `config/ppo_config.py`)
- `scripts/accel.py` → `AccelConfig` (name `"accel_config"`, `config/accel_config.py`)

Override any field on the CLI with a dotted path:

```bash
just train training_params.epochs=8 model_params.activation=relu
just accel accel_params.replay_prob=0.7 hfield_params.coarse_n=11 wandb_params.enable=true
```

`AccelConfig` reuses the PPO param groups (`EnvironmentParams`, `ModelParams`,
`TrainingParams`, `EvaluationParams`, `WandbParams`) and adds `HFieldConfig` +
`AccelParams`, plus `warm_start_algo`/`warm_start_run` (both `None` by
default): if set, restores the actor's weights from that run's latest
checkpoint before training starts, leaving the critic freshly initialized
(`utils/ppo_helpers.py:warm_start_actor`) — e.g. warm-starting ACCEL from a
`ppo_dr` run that already walks. Requires the donor's `actor_obs_dim` to
match (same `obs.state`); raises otherwise. `Config` uses the first four plus
`WandbParams`.

## EnvironmentParams

| field | default | meaning |
|---|---|---|
| `name` | `"joystick"` | task (`joystick`/`getup`/`jump`) |
| `envs_num` | `4` | parallel rollout envs (vmap width) |
| `steps_in_episode` | `200` | steps per rollout |
| `seed` | `42` | master RNG seed |

## ModelParams

| field | default | meaning |
|---|---|---|
| `hidden_dims` | `[512, 256, 128]` | MLP hidden widths (actor + critic) |
| `activation` | `"elu"` | `"elu"`, `"relu"` or `"tanh"` |

## TrainingParams

| field | default | meaning |
|---|---|---|
| `iterations` | `1000` | PPO outer loop count (plain PPO only) |
| `epochs` | `10` | PPO epochs per update |
| `minibatches` | `32` | minibatches per epoch |
| `learning_rate` | `3e-4` | Adam LR start |
| `end_learning_rate` | `1e-6` | LR end (linear schedule) |
| `gamma` | `0.95` | discount |
| `lam` | `0.95` | GAE lambda |
| `epsilon` | `0.1` | PPO clip |
| `value_coef` | `0.5` | value loss weight |
| `entropy_coef` | `0.01` | entropy bonus weight |
| `max_grad_norm` | `0.5` | global grad clip |
| `adam_eps` | `1e-5` | Adam epsilon |
| `artifacts_dir` | `"artifacts"` | checkpoint base dir; each script writes under `<artifacts_dir>/<algo>/<run>/` |
| `save_every` | `1000` | checkpoint interval |
| `eval_every` | `100` | eval interval (iterations) |
| `max_to_keep` | `5` | checkpoints retained |

Note: `epsilon`/`value_coef`/`entropy_coef` are the PPO defaults baked into
`ppo_loss`; the schedule spans `iterations × epochs × minibatches` steps (ACCEL
uses `accel_params.max_iterations` in that product instead).

## EvaluationParams

| field | default | meaning |
|---|---|---|
| `envs_num` | `5` | eval envs |
| `steps` | `200` | eval steps |

## WandbParams

| field | default | meaning |
|---|---|---|
| `enable` | `false` | wandb on/off (off → jsonl `Logger`) |
| `entity` | `""` | wandb entity |
| `project` | `""` | wandb project |
| `name` | `None` | run name — also the jsonl filename **and** the checkpoint run directory (`artifacts/<algo>/<name>/`); unset falls back to `run-<unix_ts>` (`utils.checkpoint.resolve_run_name`) |
| `tags` | `None` | wandb tags |

## HFieldConfig {#hfield}

Plain dataclass (`config/hfield_config.py`); the level genome envelope. See
[ued-accel.md](ued-accel.md#level-representation).

| field | default | meaning |
|---|---|---|
| `patch_radius` | `3.0` | m, half-width of the square patch |
| `cell_size` | `0.05` | m, fine hfield resolution |
| `coarse_n` | `9` | control points per axis (genome size) |
| `amplitude` | `0.12` | m, peak height/depth off the neutral plane |
| `difficulty` | `1.0` | runtime terrain-intensity scale in `[0,1]`, applied at render time within the fixed `amplitude` envelope; driven by `scripts/accel_top_k.py`'s terrain curriculum, see [ued-accel.md](ued-accel.md#terrain-curriculum) |
| `base_z` | `1.0` | m, solid base under the hfield (MJCF needs > 0) |
| `pad_radius` | `0.6` | m, flat spawn disc at the centre |
| `pad_taper` | `0.3` | m, ramp width pad → full terrain |
| `n_edits` | `4` | control points touched per mutation |
| `edit_scale` | `0.1` | max ± change to an edited control point |
| `tiles` | `16` | tiles per side of the shared arena, one level each (256) |
| `reach` | `10.0` | m from its tile centre a robot can read terrain (feet, height scan) within a level; the gap between tiles is `reach - patch_radius` ([tiled arena](environment.md#tiled-arena)) |
| `grid_size` | *(property)* | `round(2·patch_radius/cell_size)+1`, fine node count of one level |

`patch_radius`, `cell_size`, `amplitude`, `base_z`, `tiles`, `reach` set the
compile-time arena (static `hfield_size`, node count); changing them needs a
fresh arena (`just build-accel-arena`). `coarse_n`/`n_edits`/`edit_scale` shape the genome
and mutations and can vary freely between runs.

`patch_radius=3.0` (was `6.0`): at `ctrl_dt=0.02`/`steps_in_episode=200` a
rollout is 4s, so even a fast-command robot covers only a few metres —
most of a 12m-diameter patch was never reached, let alone scored by regret.
`edit_scale=0.1` (was `0.5`): `HFieldManager.mutate`'s intent is a *small*
nudge from an already-curated parent (see the `to_hfield_data` docstring in
[ued-accel.md](ued-accel.md#level-representation)); `0.5` was half the
entire `[0,1]` genome range per edited point, closer to a fresh `generate()`
than an incremental edit.

## AccelParams {#accel}

| field | default | meaning |
|---|---|---|
| `buffer_size` | `100` | level buffer capacity (K) |
| `initial_fill_ratio` | `0.9` | fraction seeded with random levels at start |
| `replay_prob` | `0.5` | P(replay) vs generate each iteration |
| `num_lanes` | `64` | *(`scripts/accel_top_k.py`)* independently generated/mutated levels scored per iteration |
| `envs_per_lane` | `64` | *(`scripts/accel_top_k.py`)* envs per lane; `num_lanes * envs_per_lane` is the rollout batch |
| `top_k` | `16` | *(`scripts/accel_top_k.py`)* highest-regret lanes (of `num_lanes`) curated into the buffer |
| `max_iterations` | `10000` | hard cap on the loop |
| `min_iterations` | `1000` | curation warmup before the plateau check arms; also the terrain curriculum's ramp length, see [ued-accel.md](ued-accel.md#terrain-curriculum) |
| `terrain_coarse_n_start` | `3` | *(`scripts/accel_top_k.py`)* `coarse_n` at the flattest curriculum stage |
| `terrain_coarse_n_stages` | `4` | *(`scripts/accel_top_k.py`)* discrete `coarse_n` steps up to `hfield_params.coarse_n` |
| `plateau_window` | `100` | iterations averaged into the smoothed regret/reward |
| `plateau_patience` | `200` | iterations without improvement on *either* signal → converge |
| `plateau_rel_tol` | `0.01` | relative regret drop / scaled reward rise that counts as an improvement |

Buffer entry (both scripts) is score-based, not threshold-based:
`utils/buffer.py`'s `LevelBuffer.add(level, score)` accepts unconditionally
below `buffer_size`, then displaces the buffer's current weakest entry only
if the new score beats it — `top_k` layers a per-iteration rank cut on top
of that for `scripts/accel_top_k.py`
(see [ued-accel.md](ued-accel.md#the-two-branches-algorithm-1-vanilla-scriptsaccelpy)).
The `plateau_*` group is the convergence criterion, shared by both scripts
([ued-accel.md](ued-accel.md#convergence--plateaustopper)) — convergence now
requires both `regret_score` and `mean_reward` to have stopped improving for
`plateau_patience` iterations, not regret alone. Tune convergence with
`plateau_patience`/`plateau_rel_tol` (scale-free), not an absolute regret
number.
