# UED / ACCEL

The research core: an adversarial curriculum that generates and replays
heightfield levels to train the student. Two halves — the **heightfield levels**
(`src/hfield/`) and the **ACCEL algorithm** (`src/ued/`) — driven by two
trainer scripts that share the same library code but implement different
curation strategies: `scripts/accel.py` (vanilla, single-level-per-iteration,
paper-faithful Algorithm 1) and `scripts/accel_top_k.py` (batched lanes +
top-k curation, see below).

## Level representation {#level-representation}

A level is a `coarse_n × coarse_n` array in `[0,1]` (a "genome"), bilinearly
upsampled to the fine MJX heightfield. Chosen over raw full-res edits or the
terrain generator's tile-type params because it is small, smooth by
construction, keeps a flat spawn pad, and edits directly — while reusing
`terrain.py`'s `_bilinear`/pad-taper primitives.

Values are used directly with no per-level min/max stretch: `0.5` is always the
neutral plane, so a small mutation stays physically small across levels — which
is required to keep the fixed elevation envelope (below) valid.

### `HFieldConfig` — the envelope (`config/hfield_config.py`)

Frozen dataclass; one place for every knob. `patch_radius`, `cell_size` (fine
resolution), `coarse_n` (genome size), `amplitude` (peak height off the neutral
plane), `base_z`, `pad_radius`/`pad_taper` (flat spawn disc + ramp),
`n_edits`/`edit_scale` (mutation strength). `grid_size` (property) is the fine
node count. Full field table: [configuration.md](configuration.md#hfield).

### `HFieldManager` (`hfield/manager.py`)

Pure given an rng; no hidden state beyond the config:

- `generate(rng)` — a fresh `coarse_n × coarse_n` uniform level.
- `mutate(level, rng)` — nudge `n_edits` random cells by `±edit_scale`, clip to
  `[0,1]` (the ACCEL edit).
- `to_hfield_data(level)` — bilinear upsample to `grid_size`, apply the pad
  taper (flat centre disc), recentre on `0.5`; returns the `float32`
  `hfield_data` MuJoCo reads.
- `height_range(data)` — physical min/max height for the current amplitude.

### Why swapping a level is free

`hfield_data` is a `jax.Array` leaf of `mjx.Model`, so it is swappable and
vmappable per env **with no MJX recompilation**; `hfield_size` is static. This
is the whole reason a heightfield works as an ACCEL genome at RL scale — the
mechanism (`tree_replace` in `FunctionalEnv`) is in
[environment.md](environment.md#hfield-mutability).

### Arena assets {#arena}

`hfield/arena.py` writes the static **"accel"** MJCF arena the ACCEL env loads:
a flat (`data = 0.5`) heightfield sized to the config envelope, box "apron"
strips around it, plus the spec/lookup sidecars — via `terrain.py`'s writers.
Run with `just build-accel-arena` before `just accel`/`just accel-top-k`. The
per-iteration level
is swapped into this arena's `hfield_data` at rollout time; the static asset
just fixes the compile-time envelope.

`hfield/playground.py` is a standalone offscreen render demo (`python -m
hfield.playground [--levels N] [--seed N]`) that builds a chain of generated +
mutated levels and renders each as `.ppm`. It proves generation/mutation/MJCF
injection; it does not touch mjx/vmap/PPO.

## The ACCEL algorithm (`src/ued/`)

Split by role so the file is not one grab-bag:

| Module | Contents | Role |
|---|---|---|
| `types.py` | `RolloutFn`, `Curator`, `BatchCurator`, `PPOTrainer` | interfaces / data structures |
| `regret.py` | `positive_value_loss`, `positive_value_loss_per_lane` | the regret metric (Eq. 5), single-level and per-lane |
| `levels.py` | `level_to_hfield_data(_batch)`, `sample_(batch_)from_buffer`, `generate_batch`, `mutate_batch`, `fill_buffer` | buffer/level plumbing |
| `stopper.py` | `PlateauStopper` | convergence |
| `curriculum.py` | `decide_replay`, `score_and_curate`/`generate_branch`/`replay_branch` (vanilla) + `score_and_curate_top_k`/`generate_branch_top_k`/`replay_branch_top_k` (batched) | Algorithm 1, two variants |

Dependency direction is a DAG: `types`/`regret`/`levels`/`stopper` are leaves;
`curriculum` composes them plus `utils.ppo_helpers.run_ppo_epochs`.

- **`Curator`** (NamedTuple, vanilla) — the level `buffer` (`utils/buffer.py`,
  a capacity-bounded priority list), `hfield_manager`, `rng`, `replay_prob`.
- **`BatchCurator`** (NamedTuple, batched top-k) — same buffer/hfield_manager/
  rng/replay_prob, plus `num_lanes`, `top_k`.
- **`PPOTrainer`** (NamedTuple, shared) — the jitted `rollout` / `advantage` /
  `make_step` callables plus `epochs`/`minibatches`. Built once per script.
- **Regret** = `positive_value_loss(advantages)` = `mean(max(advantage, 0))`
  (per-lane variant: `positive_value_loss_per_lane`): how much the critic
  under-estimated return on a level. High regret = a level the student
  handles surprisingly badly.

### The two branches (Algorithm 1) — vanilla (`scripts/accel.py`)

Paper-faithful: one level per iteration. `decide_replay` flips a
`replay_prob` coin (only if the buffer is non-empty):

**generate** (`generate_branch`) — no policy update:
1. `hfield_manager.generate` a fresh level.
2. `score_and_curate`: roll out the student on it, score regret, add to the
   buffer.

**replay** (`replay_branch`) — the only branch that trains the student:
1. `sample_from_buffer` a stored level `θ`.
2. Roll out, GAE, **`run_ppo_epochs`** (the PPO update).
3. `mutate` θ → θ′, `score_and_curate` θ′ (score the edit, curate it).

`score_and_curate` is the shared "roll out → regret → curate" step both
branches use. `LevelBuffer.add(level, score)` (`utils/buffer.py`) does the
curation: below capacity it always accepts; once full, a new level replaces
the buffer's current *weakest* (lowest-score) entry only if it beats that
score, otherwise it's rejected. There is no absolute regret threshold — a
fixed threshold is scale-dependent (same problem the stopper's rel-tol
design avoids, below) and, at `0.0`, was a no-op: `positive_value_loss` is
`>= 0` by construction, so every scored level passed. Freshly seeded levels
from `fill_buffer` carry a placeholder score of `-inf`, so real scored
levels always displace unscored ones first.

### The two branches — batched lanes + top-k (`scripts/accel_top_k.py`)

A deliberate departure from vanilla ACCEL's one-level-per-iteration loop,
aimed at faster curriculum convergence rather than just wall-clock
parallelization: each iteration scores `num_lanes` independently
generated/mutated levels at once (`envs_per_lane` envs each,
`num_lanes * envs_per_lane` = the total rollout batch, matching the env
budget the vanilla script spends on one repeated level) instead of
replicating a single level across every env. `decide_replay` still flips one
`replay_prob` coin per iteration (only if the buffer is non-empty), applied
to the whole batch:

**generate** (`generate_branch_top_k`) — no policy update:
1. `generate_batch`: `num_lanes` fresh levels.
2. `score_and_curate_top_k`: roll out the student on all lanes, score regret
   per lane, add the `top_k` highest-regret levels to the buffer.

**replay** (`replay_branch_top_k`) — the only branch that trains the student:
1. `sample_batch_from_buffer`: `num_lanes` stored levels `θ_i` (with
   replacement).
2. Roll out all lanes, GAE, **one `run_ppo_epochs` call on the pooled batch**
   — the update sees `num_lanes` diverse levels' worth of experience instead
   of `num_lanes * envs_per_lane` copies of the same level (this is the part
   that directly targets vanilla ACCEL's "trains on the same env repeatedly"
   cost).
3. `mutate_batch`: θ_i → θ_i′ for every lane, `score_and_curate_top_k` the
   mutated children (score with the just-updated student, curate the
   `top_k`).

`score_and_curate_top_k` is the shared "roll out → per-lane regret → top-k
add" step both branches use — entry into the buffer is rank-based within each
iteration's batch (`np.argsort(scores)[-top_k:]`), then each of those `top_k`
levels goes through the same `LevelBuffer.add` weakest-entry eviction as
vanilla. See `todo.md` for a deferred follow-up (evolving multiple mutations
of the same parent and keeping the best one).

### Stop-gradient / rollout is off-tape {#stop-gradient}

JAX has no global autograd tape — a gradient exists only where `grad` is called,
and that is **only** `make_step` (`eqx.filter_value_and_grad`). So:

- Rollouts (`batch_rollout`, or `lanes_rollout` vmapping it over lanes for the
  top-k variant) are plain jitted forward passes; nothing differentiates
  through them.
- In `ppo_loss` the stored `episode.log_prob` enters as the constant
  `old_log_prob`; the gradient flows only through a fresh forward pass of the
  current model — the standard PPO stop-gradient on the data-collecting policy
  (paper's "collect τ with stop-gradient ϕ⊥").
- The generate branch (`generate_branch`/`generate_branch_top_k`) computes
  regret but runs no optimizer at all — the score only gates buffer entry.

Advantages/returns are materialised once and passed as constant array arguments
into `ppo_loss`, so GAE targets are detached by construction.

## Convergence — `PlateauStopper`

An absolute regret threshold is scale-dependent and ambiguous (low regret can
mean "solved everything" or "generator stuck"), so convergence is **relative**
— and it now requires **both** the regret and the reward signal to plateau,
not regret alone:

- Tracks a rolling mean of each iteration's `regret_score` **and**
  `mean_reward` over `plateau_window`.
- An iteration counts as "improved" if the smoothed regret dropped by
  `> plateau_rel_tol`, OR the smoothed reward rose by more than
  `plateau_rel_tol * max(1.0, |best_reward_so_far|)` (the reward margin
  scales with the best reward seen, floored at `1.0`, since reward can be
  negative or near zero where a plain relative check is unstable).
- `_stale` resets to `0` whenever *either* signal improves; convergence
  triggers only once `_stale >= plateau_patience` — i.e. neither signal has
  improved for that many iterations in a row.
- Arms only after `min_iterations` (curation warmup — early regret is not yet
  meaningful).
- `max_iterations` remains a hard cap.

This closes a real failure mode: `positive_value_loss` (regret) can plateau
near zero either because the student mastered the curriculum, or because the
critic simply learned to predict a *consistently bad* policy just as
accurately — regret alone can't tell the two apart. Replaying a real
`accel.py` run's logged history through the old (regret-only) stopper showed
it converging at iteration ~1233 while `mean_reward` was still climbing
(-14.5 → +14.6, still rising, not flattened); replaying the same history
through the dual-signal stopper does not converge at all — it correctly
keeps training alive as long as either signal is still moving.

Fed the `regret_score`/`mean_reward` from **both** branches — the regret of a
freshly proposed level for the current student (vanilla: that one level's
score; top-k: the mean across `num_lanes` lanes, with `max_regret_score` also
logged for visibility but not fed to the stopper) — the signal for "generator
can no longer challenge the student", i.e. curriculum saturation.
`smoothed_regret`/`smoothed_reward` are logged each iteration.

## Trainer scripts — `scripts/accel.py` / `scripts/accel_top_k.py`

Both build the same optax/jit/`FunctionalEnv` stack as `scripts/ppo.py` (with
the `accel` terrain arena and an `hfield_data`-swapping env, plus
`utils.env_overrides.movement_overrides()`/`terrain_overrides()` —
[ppo.md](ppo.md#shared-env-overrides--utilsenv_overridespy)), plus:
`LevelBuffer` seeded by `fill_buffer` (`initial_fill_ratio`), a `Curator`
(`accel.py`) or `BatchCurator` (`accel_top_k.py`), a `PPOTrainer`, and a
`PlateauStopper`. The `while not converged` loop dispatches generate/replay,
logs `regret_score`/`mean_reward`/`buffer_fill_ratio`/`smoothed_regret` (plus
`max_regret_score` for the top-k script), evals every `eval_every`,
checkpoints, and stops on plateau or `max_iterations`. The two scripts differ
only in which curriculum functions and rollout builder they wire up
(`generate_branch`/`replay_branch`/`batch_rollout` vs.
`generate_branch_top_k`/`replay_branch_top_k`/`lanes_rollout`) — everything
else (model, optimizer, checkpointing, eval, logging) is identical, aside
from `ALGO` (`"accel"` vs `"accel_top_k"`), which keeps their checkpoints in
separate `artifacts/<algo>/` trees ([ppo.md](ppo.md#checkpoints--utilscheckpointpy)).

One Hydra quirk: `HFieldManager` needs `HFieldConfig.grid_size` (a property),
which Hydra's structured DictConfig does not expose, so each script
materialises `cfg.hfield_params` back into a real `HFieldConfig` via
`OmegaConf.to_object`.

All knobs (`AccelParams` + reused PPO params): [configuration.md](configuration.md#accel).
