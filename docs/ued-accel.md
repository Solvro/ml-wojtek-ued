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

Plain dataclass (not frozen — `frozen=True` marks the Hydra structured-config
node read-only and blocks CLI overrides like `hfield_params.amplitude=...`);
one place for every knob. `patch_radius`, `cell_size` (fine
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
[environment.md](environment.md#hfield-mutability). Caveat: per-env levels
only reach the physics on the `jax` backend — on `warp` every env in a batch
collides with the first env's level (same section).

### Arena assets {#arena}

`hfield/arena.py` writes the static **"accel"** MJCF arena the ACCEL env loads:
a flat (`data = 0.5`) heightfield sized to the config envelope, box "apron"
strips around it, plus the spec/lookup sidecars — via `terrain.py`'s writers.
Two far corner nodes are pinned to `0`/`1`: MuJoCo rescales a loaded hfield
to span `[0, 1]`, so an all-`0.5` file compiles to `0` and sinks the static
ground to `-amplitude`. That broke `real_pose_ref` (its standing-height settle
runs on the static model, so every commanded height mapped to the most
extended pose and zero action tripped the four-bar toggle fall within
~0.35 s) and any rollout without an `hfield_data` swap.
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
| `curriculum.py` | `decide_replay`, `terrain_curriculum_step`, `score_and_curate`/`generate_branch`/`replay_branch` (vanilla) + `score_and_curate_top_k`/`generate_branch_top_k`/`replay_branch_top_k` (batched) | Algorithm 1, two variants, plus the terrain curriculum |

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

## Terrain curriculum (`scripts/accel_top_k.py`) {#terrain-curriculum}

`amplitude` is baked into the compiled MJCF arena at `just build-accel-arena`
time (`hfield/arena.py`'s `size`/`pos_z`), so it cannot change once the env is
loaded. `terrain_curriculum_step` (`ued/curriculum.py`) instead ramps
difficulty within that fixed envelope, driven by `iteration / min_iterations`
(clamped to `[0,1]`) each loop iteration, called before `decide_replay`:

- **`difficulty`** (`HFieldConfig`, default `1.0`) — a continuous `[0,1]`
  scale `HFieldManager.to_hfield_data` applies to the level's deviation from
  the neutral `0.5` plane, so `0` renders perfectly flat and `1` is the full
  `amplitude`. Applied at render time, not generation time, so it needs no
  buffer changes — even a level curated early at low difficulty re-renders at
  the current difficulty when replayed later.
- **`coarse_n`** — moves through `terrain_coarse_n_stages` discrete steps
  from `terrain_coarse_n_start` up to `hfield_params.coarse_n`, instead of
  continuously: `coarse_n` sets the shape of the buffered level arrays
  (`LevelBuffer` entries are raw `coarse_n × coarse_n` genomes), and mixing
  shapes breaks `np.stack` in `sample_batch_from_buffer`/`mutate_batch`. Each
  stage transition clears the buffer (`LevelBuffer.clear`) and reseeds it
  with `fill_buffer` at the new resolution.

Both ramps reach their end value by `min_iterations` — the same point the
`PlateauStopper` arms — so regret/reward trends the stopper measures are
always read against the fully-difficult terrain, not a moving target. Logged
each iteration as `terrain_difficulty`/`terrain_coarse_n`.

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

## DR vs ACCEL on brax PPO — `scripts/brax_ued.py` {#brax-ued}

Our own PPO (`utils/ppo_helpers`) did not learn to walk on this env within
~0.6B steps, while brax's PPO (mujoco_playground recipe) did in ~40M
(`scripts/brax_ppo.py`). `brax_ued.py` keeps brax's PPO pieces (networks,
running obs normalizer, `compute_ppo_loss`, GAE, 20-step unrolls, 4 epochs x
32 minibatches, discount 0.97) and drives `FunctionalEnv`, because
`ppo.train` fixes each env's model for the whole run and cannot swap levels.

- Outer iteration: `num_lanes` levels x `envs_per_lane` envs, reset onto
  their level (plus a fresh physics DR draw), stepped `level_steps` in
  20-step unrolls; when training, a PPO update follows every unroll.
  Lane regret = mean positive GAE advantage over the iteration.
- `--mode dr`: fresh uniform levels at full difficulty, train every
  iteration.
- `--mode dr_easy`: DR over ACCEL's easy generator (control points within
  `generator_scale` of flat) — isolates the curriculum from "train easy".
- `--mode dr_mixed`: uniform levels scaled by a per-level difficulty
  drawn from `[0, 1]`.
- `--mode accel`: replay iterations (`replay_prob`, once the buffer holds
  `num_lanes` levels) train on rank+staleness-prioritized buffer levels,
  refresh their scores and queue their mutations; explore iterations do not
  train and score the queued children plus fresh easy levels (control points
  within `generator_scale` of flat). The buffer keeps the `buffer_size`
  highest-regret levels.
- ACCEL curriculum switches (defaults reproduce plain ACCEL):
  - `--mutation grow` — after the local edits, scale the level's deviation
    from flat by `U(1, grow_max)` (default 1.3). Plain symmetric edits
    smooth as often as they roughen; plain ACCEL's buffer stayed at the easy
    generator's roughness (~0.03 m) for the whole run.
  - `--score learnability` — level score `p(1-p)` (SFL), `p` = fraction of
    the lane's envs that never fell and ended >= `min_travel` m out,
    instead of positive value loss.
  - `--generator mixed` — explore iterations draw fresh levels from the
    `dr_mixed` distribution instead of the easy generator.
  - Logged: `buffer_roughness_mean`/`_p90`, `buffer_score_roughness_spearman`
    (rank correlation of level score vs roughness), `children_roughness`,
    `train_success`, `learnability_mean`.
  - WCSS variants (`brax_ued.sbatch` `MODES`): `accel_grow` = grow + 16
    edits; `accel_sfl` = grow + 16 edits + learnability.
- `--mode paired`: PAIRED (port of `feat/paired-implementation`'s design
  onto the brax student). A learned adversary (`modules/generator.py`,
  latent seed -> sigmoid-squashed Gaussian over the genome) proposes
  `num_lanes` levels; protagonist and antagonist (two brax students) both
  train on them; regret = antagonist max return - protagonist mean return
  per level, clipped at 0, is the adversary's reward (single-step PPO, 5
  epochs, HiEnt `gen_entropy_coef=0.1`, no BC distillation yet).
  `env_steps`/`num_timesteps` count protagonist steps only; the antagonist
  doubles the real simulation cost (`sim_steps` counts both).
- `--mode plr`: robust PLR (PLR⊥) - ACCEL's buffer, rank/staleness replay
  and score refresh, but no mutation; explore iterations score fresh
  uniform (DR) levels.
- `--robot go1`: Playground's Go1 instead of Wojtek, see
  [Go1](#go1).
- `--profile {benchmark,deploy}` (default `benchmark` = everything above,
  the v10 setup). `deploy`: w01-tek `terrain_blind_v4_4d`'s env, DR and SiLU
  network on the 12x12 m `accel_deploy` arena, exportable to the robot —
  [deploy.md](deploy.md#profile).
- Sidecar: with `--checkpoint_dir`, `<name>.json` is written beside
  `<name>.pkl` at launch (profile, robot, arena, `HFieldConfig`, level
  space, resolved env overrides, DR, network, flags, obs sizes, git commit);
  every consumer rebuilds env and network from it
  ([deploy.md](deploy.md#sidecar)). `brax_ued.py` also checks the built
  arena's geometry against the run before training.
- Eval every `eval_every` iterations: deterministic policy on fresh uniform
  levels at difficulties 0.25/0.5/0.75/1.0 — `eval/d<d>/success_rate` (no
  fall and >= `min_travel` m from spawn), fall rate, travel, forward vs
  commanded speed.
- Curriculum composition (`--level_space param` only), every iteration over
  the levels the student ran on (replayed, explored or PAIRED-generated):
  `level/<param>` = mean physical value (`ParamSpace.physical`, robot
  units), `level/slope_deg_abs` / `level/step_height_m_abs` = mean |value|
  of the signed parameters, `level/genome_dist` = mean |genome - `flat`|.
- Compute accounting, every iteration: `sim_steps` (every training env step
  simulated: PAIRED's antagonist and ACCEL/PLR explore iterations included,
  eval rollouts not), `student_train_steps` (env steps whose data fed a
  student PPO update), `elapsed_s` (wall clock since launch, compile
  included). The run summary ends with `final/elapsed_h`, `final/sim_steps`,
  `final/student_train_steps` (`paper_tables.py` compute table).
- WCSS: `brax_ued.sbatch` array (tasks 0-2 dr, 3-5 accel, seeds 0-2),
  `PROFILE=deploy` for the deploy profile; assets must be built beforehand
  since tasks start together.
- The arena must match the run's geometry (`hfield.arena.check_arena`:
  `pad_radius`, height scale, grid size); the error names the
  `just build-accel-arena arena=<kind> hfield_params...` command for it.

## Level spaces — `hfield/level_space.py` {#level-spaces}

`brax_ued.py` takes `--level_space {grid,param}` (eval scripts read it from
the sidecar); every mode samples, mutates and renders levels through it.

- `grid` (default, v7 runs): the original `coarse_n x coarse_n` control-point
  genome. i.i.d. noise, so levels at one scale are statistically alike: the
  v7 final eval measured per-level success correlation between passes of
  only 0.02–0.17, and ACCEL landed within seed noise of DR on its own easy
  generator (DR-easy).
- `param`: 8 terrain parameters in [0, 1] plus a noise seed (last
  entry, never mutated). `PARAMS` maps them to physical ranges sized to the
  ~0.10 m robot (v10): noise amplitude 0–0.10 m and wavelength 0.2–1.0 m,
  radial slope −20…+18°, radial stairs −0.08…+0.05 m high and 0.15–0.60 m
  wide, gaps 0–0.06 m wide (0.15 m deep, every 0.6 m), blocks 0–0.06 m high
  and 0.15–0.60 m wide. Features are radial (or seeded fields) so heading
  does not matter.
  - v8 used slope ±20°, stairs ±0.08 m, gaps 0–0.25 m, blocks 0–0.10 m:
    best methods solved ~8% of uniform levels. v10 caps uphill slope,
    step-up, gap and block at the best v8 policies' 50% envelope
    (`axis/*/envelope`); v8 policies solve ~39% of the v10 box (estimated
    from v8 `random/d1.0` levels that fall inside it). Features compound,
    so uniform levels still range from trivial to unsolved.
  - `flat` = genome of zero slope/step, so it is off 0.5 on asymmetric ranges.
  - DR: uniform genome; `scaled(d)` shrinks it toward `flat` (no features)
    by d; `easy(s)` = `scaled(s)`; DR-mixed draws d per level.
  - ACCEL: `mutate` edits `--n_edits` (WCSS: 2) parameters by ±`--edit_scale`
    (0.1); `--mutation grow` also scales the genome away from `flat`.
  - PAIRED: the generator emits the 8 parameters; a random seed is appended.
- WCSS: `slurm/brax_ued.sbatch` / `slurm/eval_brax.sbatch` take
  `SPACE=param`, `PREFIX` (run-name prefix, `v10`), `ROBOT`,
  `NUM_TIMESTEPS` (train), `NSEEDS` / `SEED0` (array task = mode x
  NSEEDS + seed offset; default seeds 0-2, extra seeds e.g. `NSEEDS=2
  SEED0=3`) and `SRC` (`<dir>/src` of a frozen copy with the
  built `assets/` beside it - `wojtek_env.paths` resolves assets from
  `src/..` - so queued jobs keep the submitted code; put
  `git rev-parse HEAD > <dir>/GIT_COMMIT` there for the sidecar).
- `robot_scale` (`ParamSpace`, eval suite): every length range (amplitude,
  wavelength, stairs, gaps, blocks, gap depth/period) times the robot's size
  relative to Wojtek; slopes unchanged. Envelopes stay in Wojtek units.

## Go1 — `robots/go1.py` {#go1}

Second robot for the benchmark: open serial chain (Go1) vs Wojtek's closed
chain. `--robot go1` in `brax_ued.py` (evals read it from the sidecar), `ROBOT=go1` on WCSS
(run prefix `v9_go1`).

- Playground `Go1JoystickRoughTerrain` (feet-only collisions, its reward,
  blind actor), on the jax backend, with:
  - the ACCEL arena: one `hfield` geom `floor` (geom 0, so Playground's
    friction DR hits it), sized by `HFieldConfig`;
  - forward commands: Wojtek's `vx (0.2, 0.6)`, `vy (-0.2, 0.2)` times
    `GO1_SCALE`, `wz (-0.4, 0.4)`;
  - Wojtek's fall rule at Go1's size: trunk below 0.6 x 0.258 m above the
    terrain under it, or gravity z > -0.4;
  - critic gets a clean 5x5 height scan (Wojtek's grid x `GO1_SCALE`).
- `GO1_SCALE = 2.5` (standing trunk 0.258 m vs Wojtek 0.10 m) scales the
  arena (`hfield_config`), level lengths, commands and `min_travel`.
- Physics DR: Playground's `go1/randomize.py`.
- Needs `mujoco_menagerie` under Playground's `external_deps` (meshes);
  `ensure_menagerie_exists()` clones it, by hand on WCSS (no AVX on login).

## Final eval — `scripts/eval_brax.py` {#eval-brax}

Loads a `brax_ued.py` checkpoint (`--checkpoint <run>.pkl`; robot, level
space, arena and network from its sidecar `<run>.json`), deterministic
policy, no physics DR, no `no_progress` cut. Success = no fall and >=
`min_travel` m from spawn after `level_steps`.

- `random/d<d>`: `num_levels` (2048) genome levels, the same genomes at
  each difficulty 0.25/0.5/0.75/1.0.
- `suite/<terrain>_<k>`: held-out structured terrains no method trains on,
  written straight into `hfield_data`, radial so heading does not matter:
  stairs up/down (2k cm steps every 0.3 m), slopes up/down (5k°), trenches
  (10 cm deep, 5k cm wide, every 0.5 m), waves (±2k cm, 0.6 m), random
  blocks (±2k cm, 0.3 m cells); k = 1..4. Heights clip at ±`amplitude`.
- **CVaR** (`cvar<alpha>_success`, alpha 0.1 / 0.25): every level runs in two
  independent passes; the worst-alpha levels ranked on one pass are scored
  on the other, both ways, averaged (cross-fitting). Ranking and scoring
  on the same episodes is biased low by selection noise: with 2000 levels
  of true success 0.5 and 4 episodes each, same-pass CVaR25 reads 0.19,
  cross-fitted 0.49.
- **Command-conditioned success** (`<set>/cmd_success_rate`, secondary):
  no fall and the env's sampled command tracked, judged like the deploy
  tracking reward (`tracking_relative`). Per step, body frame:
  `lin_rel = |v_xy - cmd_xy| / max(|cmd_xy|, 0.3 m/s x ROBOT_SCALE)`,
  `yaw_rel = |wz - cmd_wz| / max(|cmd_wz|, 0.7 rad/s)` (the deploy reward's
  `tracking_rel_floor_lin` / `_ang`). Tracked = episode means of both
  < 0.5, i.e. within 50% of the command (`TRACK_*` in `eval_brax.py`).
  Steps within 1 s of the reset or of a command change are a start-up
  transient and not scored: the env resamples every 250 steps (5 s), so a
  6 s level scores 1-5 s and the second command's last second is all
  transient. Deploy commands include stand, spin and slow draws
  whose |v_xy| x 6 s is below `min_travel` (~19% of draws on
  accel_deploy), which caps `success_rate` below 1. Travel is distance
  from spawn, so backward walks can reach it.
  - Stand-still floor: a robot that never moves passes 12.4% of deploy
    command draws (accel_deploy rows; 100k draws of the env's
    `_sample_command`), nearly all of them the 10.4% zero commands. An
    absolute 0.25 m/s / 0.5 rad/s tolerance gave it 26.9%.
  - `track_lin_err` / `track_yaw_err` = mean absolute errors (m/s, rad/s)
    over episodes without a fall; npz adds `<set>_lin_err`, `_yaw_err`,
    `_lin_rel`, `_yaw_rel` per random/param set.
- `--level_space param` adds `param/uniform` (2048 levels drawn from the
  full 8-parameter space: heterogeneous, so CVaR reads a real tail) and
  capability envelopes `axis/<feature>`: one feature (rough, slope up/down,
  stairs up/down, gaps, blocks) ramps to its full range in 6 steps, the rest
  off; `envelope` = largest |feature| passed at 50% success, `auc` = mean
  success over the ramp. Difficulty 0.25 is too easy to separate methods
  (~89% for anything trained near flat) and is not a headline number.
- `flat/...`: `suite_instances` flat levels (`scaled(0)`), the no-terrain
  reference (success, fall rate, travel, CVaR). Runs last on its own
  rng/key, so every other set keeps its v8 levels and keys.
- 95% bootstrap CIs over levels for mean and CVaR; `level_success_corr`
  between passes; suite CVaR is over the 28 terrain×strength cells.
- Writes `<name>.json` (metrics) and `<name>.npz` to `--out_dir`, and the
  metrics to the wandb run summary. npz arrays (pass x level x episode
  unless noted):
  - `random_d<d>_fell`, `random_d<d>_travel`; `random_levels` (levels x
    genome): the d=1 base genomes of the `random/d*` sets.
  - `suite_success` (28 terrain x strength cells).
  - `param` only: `param_uniform_fell`, `param_uniform_travel`,
    `param_uniform_levels` (levels x genome). `_travel` and the genomes are
    new in v10; v8 npz files lack them. WCSS: `eval_brax.sbatch` (same `MODES` task layout
  as `brax_ued.sbatch`).

## Sim2sim — `scripts/eval_sim2sim.py` {#sim2sim}

Transfer test in place of sim2real: the MJX-trained policy, physics stepped
by CPU MuJoCo under several model/solver variants. Same checkpoint loading,
network and deterministic policy as eval_brax; no physics DR.

- **Only the simulator changes.** Observations, termination, commands and
  control latency are the env's jax code. While the unroll is traced,
  `mjx_env.step` and `WojtekEnv._step_with_latency` are swapped for a
  `jax.pure_callback` (`cpu_physics`); `mjx.step` raises inside it, so a
  missed physics path fails loudly. Reset stays on MJX (it only places the
  robot).
- `CpuPhysics`: one `MjModel` copy + `MjData` per thread
  (`sched_getaffinity` threads), contiguous env slices, `hfield_data`
  reloaded when the next env is on another level. Per env and control
  period it sets qpos/qvel/act/warmstart/time/applied forces, runs
  `mj_step` (latency: `prev` ctrl for `delay` substeps, then `new`), and
  writes back the state plus frames, sensors (mapped by sensor name) and
  actuator forces. Geom frames are rebuilt from body frames with the
  training model's offsets, so a variant may add/drop geoms.
- Diverged env (MuJoCo bad qpos/qvel/qacc warning, or non-finite state):
  base dropped to z = -100 m so both robots score a fall; counted as
  `diverged_per_episode`.
- Variants (`VARIANTS`, one table in code), all on the same levels and reset
  keys:

  | Variant | Change vs training model | Tests |
  |---|---|---|
  | `mjx_match` | none | CPU MuJoCo vs MJX implementation; must match eval_brax within noise |
  | `newton_elliptic` | elliptic cones, 100 iterations / 50 line search (training: pyramidal, Wojtek 2 / Go1 1 iterations) | reliance on the unconverged training solver |
  | `half_dt` | timestep 0.004 -> 0.002 s, 2 CPU steps per training substep (policy rate unchanged); solref time constants below the old 2·dt floor pinned at 0.008 s | integration-step sensitivity only |
  | `full_collision` | Wojtek: the source model's per-link `*_collision` meshes collide with the terrain (`build_model` disables them; spec recompiled for convex hulls). Go1: Playground's `go1_mjx_fullcollisions.xml` | contacts the MJX model drops (shins, thighs, trunk) |
  | `soft_contact` | geom solref time constant x2 (`SOFT_TIMECONST_SCALE`) of the value training runs at: Wojtek 0.001 s (runs at the 0.008 s floor) -> 0.016 s, Go1 0.02 -> 0.04 s; damping ratio, solimp unchanged | contact stiffness, relative to each robot |

- Levels: `flat` (`suite_instances`), `param/uniform` = first `num_levels`
  (default 256) of eval_brax's 2048-level draw (same seed, rows line up
  with eval_brax's npz), suite strengths 1-2 (`suite/<terrain>_<k>`, pooled
  as `suite`); robot-scaled as in eval_brax.
- Metrics per `<variant>/<set>`: `success_rate`, `fall_rate`, `mean_travel`
  (95% bootstrap CIs over levels), `drop` = `mjx_match` - variant success
  (paired bootstrap CI), `diverged_per_episode`; per suite cell
  `success_rate`; `<variant>/elapsed_s`. One pass (`--episodes` per level),
  no CVaR.
- Sanity (2026-10-11, `v8_param_accel_seed0`, 16 flat / 64 uniform / 16
  per suite cell, 4 episodes, same levels and keys): `mjx_match` vs MJX
  (eval_brax rollout) success flat .922 / .922, suite 1-2 .778 / .777,
  param/uniform drop +.02 (95% CI -.016….059). One control step from the
  same state differs by up to 1.6 rad/s in qvel at the training solver
  (2 iterations) but 0.01 at 50 iterations: the gap is the unconverged
  solver, not the bridge.
- Go1 does **not** match on heightfields, and the bridge is not the cause.
  v10 checkpoints, 32 flat / 64 uniform / 32 per suite cell, 4 episodes,
  MJX vs `mjx_match` success:

  | Checkpoint | flat | param/uniform | suite 1-2 |
  |---|---|---|---|
  | `v10_go1_dr_easy_seed1` | .92 / .08 | .25 / .01 | .78 / .06 |
  | `v10_go1_accel_seed0` | .71 / .05 | .11 / .00 | .58 / .03 |

  - Same flat levels with the `floor` hfield swapped for a plane in both
    simulators: MJX .941 / CPU .941 (dr_easy), .754 / .754 (accel). The
    harness (latency/substeps, ctrl, sensors, write-back) reproduces MJX.
  - Root cause: CPU MuJoCo collides a sphere with each hfield prism on its
    own. A Go1 foot (r 0.023 m, up to 15 mm penetration at solref 0.02 s)
    over a triangle edge gets a second contact from the neighbouring prism,
    normal tilted 23-47° off vertical, even on a perfectly flat level. MJX's
    `hfield_sphere` keeps only points within 1 mm of the deepest, so it
    reports one vertical contact per foot (flat hfield = plane in MJX).
    Teacher-forced from MJX states, converged solver on both: one control
    step's qvel error is 1e-5-1e-4 with equal contact sets, up to 0.2 rad/s
    with the extra CPU contacts.
  - Not the solver: 100 Newton iterations on CPU only lifts flat to .51
    (dr_easy) / .18 (accel); `ccd_iterations` 100 changes nothing, libccd
    (`nativeccd` off) is worse (0.0).
  - Not contact sensitivity: on the plane, Go1 dr_easy keeps flat .95
    `soft_contact`, .94 `half_dt`, .95 `newton_elliptic`.
  - Wojtek (0.046 m feet, 0.008 s contacts) shows no such gap; its flat
    success under `soft_contact` is .94 (`mjx_match` .95, 32 levels).
  - Consequence: every Go1 variant on hfields is floored by `mjx_match`
    itself, so Go1 rows measure the CPU hfield collider, not transfer.
- Cost (10-core Apple M-series, default args): Wojtek ~18 min wall / 1.8
  CPU-h for all five variants (160-290 s per variant, `full_collision`
  slowest), Go1 ~7 min / 0.4 CPU-h.
- Writes `<name>.json` and `<name>.npz` (`<variant>_<set>_fell/_travel`,
  levels x episodes) to `--out_dir`; wandb tags `sim2sim`, robot.
- Needs a `param`-space checkpoint (read from the sidecar).
- WCSS: `slurm/eval_sim2sim.sbatch`, same `MODES`/seed array layout, `SRC`,
  `PREFIX` as `eval_brax.sbatch`; `lem-cpu-short`, 16 cores (CPU
  only; Lem CPU nodes share the GPU nodes' AlmaLinux 9, so the venv runs).

## Demo videos — `scripts/render_video.py` {#render-video}

mp4 clips of a `brax_ued.py` checkpoint walking generated levels, for demos
(not an eval: no metrics beyond a printed fall time and travel per clip).

```sh
just render-video --checkpoint artifacts/checkpoints/<run>.pkl \
  --level 0.5:<seed> --level 0.75:<seed> --level 1.0:<seed>
```

- Env, network, robot, level space and arena come from the checkpoint's
  sidecar `<run>.json` ([deploy.md](deploy.md#sidecar)); pre-sidecar
  checkpoints need `just write-sidecar` first. A sidecar whose obs sizes
  differ from the checkpoint fails before rollout.
- Needs the run's arena built with its geometry (v10: `just
  build-accel-arena hfield_params.amplitude=0.3 hfield_params.coarse_n=13
  hfield_params.pad_radius=0.55 hfield_params.pad_taper=0.2`; deploy
  profile: `just build-deploy-arena`). For Wojtek the script checks the
  arena's spec (`pad_radius`, height scale, grid size) first and fails with
  the command filled in; an amplitude mismatch would otherwise run
  silently.
- `--level D:SEED` (repeatable): `space.scaled(default_rng(SEED), 1, D)`,
  one clip each, named `<checkpoint stem>_d<D>_s<SEED>.mp4`; `demo.mp4` in
  the same `--out_dir` (default `artifacts/videos`) = all clips in order.
- Rollout on MJX (eval env: no physics DR, no `no_progress` cut, pushes on),
  deterministic policy, `--command VX VY WZ` held every step (default
  0.3 m/s forward, 0.2 rad/s yaw: an arc; on 24 screened d=1 levels the
  robot got 1.1-3.2 m from spawn in 20 s, so it stays on the 3 m arena).
  The command's height entry keeps the reset draw; the resample counter
  is pinned every step (Wojtek `steps_since_cmd` = 0, Go1
  `steps_until_next_cmd` = 2^30) so the env never resamples.
- `--seconds` (default 20) must fit in one episode (1000 steps = 20 s for
  Wojtek): a fall or truncation auto-resets the robot to spawn, which the
  clip would show as a jump. Pick levels whose printed fall list is empty;
  screen candidates with this single-env rollout, not a vmapped batch:
  the batch's MJX numerics differ enough to flip falls (`d=1.0` seed 19
  passed batched, fell twice single-env).
- Levels with a strong slope clip at ±`amplitude` (0.3 m) within ~1 m of
  the pad (d=1 slope up to -20°), leaving a flat pit floor; for a demo,
  prefer low-slope levels whose rough/stairs/gaps persist outward.
- Rendering: the recorded `qpos` replayed through `mujoco.Renderer`
  (`--width`/`--height`, default 1280x720, `--fps` 30, frames at the
  nearest control step), camera tracking the free-joint body; mp4 via
  `mediapy` (needs `ffmpeg` on `PATH`). `MUJOCO_GL`: `cgl` on macOS,
  `egl` on a GPU node, `osmesa` on a CPU node (recipe defaults to cgl/egl).
- Cost (Apple M-series, CPU jax): ~1 min compile, then ~1 min per 20 s
  clip.
- Team demo (2026-10-15): `v10_wojtek_accel_seed4` (best of the 5 v10
  Wojtek ACCEL seeds by final eval: `random/d1.0` success .423,
  `param/uniform` .412, suite .584, flat .936), `--level 1.0:17` (rough
  8.5 cm / 0.33 m wavelength, stairs -3.2 cm, gaps 2.3 cm, slope 1.2°):
  no fall, 2.89 m from spawn after 20 s.

## Paper tables — `scripts/paper_tables.py` {#paper-tables}

Builds the workshop paper's tables from `eval_brax.py` outputs
(`<prefix>_<method>_seed<k>_final_eval.{json,npz}`); bare booktabs
`tabular`s; the paper (on Overleaf) owns float, caption and label.

```sh
PYTHONPATH=src uv run python src/scripts/paper_tables.py \
  --dump_wandb summaries.json --prefix wojtek=v10_wojtek go1=v10_go1
PYTHONPATH=src uv run python src/scripts/paper_tables.py \
  --eval_dir <evals> --prefix wojtek=v10_wojtek go1=v10_go1 \
  --train_summaries summaries.json --out artifacts/paper/tables
```

- `--prefix robot=prefix` (repeatable): one column group / row group per
  robot. Known methods get paper names (DR, DR-easy, PLR⊥, ACCEL,
  ACCEL-SFL, PAIRED) in that order; other methods follow under their own
  name; missing ones are skipped.
- `main.tex`: success (%) per method x robot: `param/uniform` mean, its
  CVaR25, solvable subset, `suite/success_rate`, `flat/success_rate`;
  mean ± std over seeds, best per column in bold, `--` if absent.
- Solvable subset: `param/uniform` levels that at least one run of the robot
  (any method) passes on >= 1 episode of pass A; each run is scored on pass
  B on them and vice versa, averaged (cross-fitted, like CVaR). Needs
  `param_uniform_travel`; pre-v10 evals warn and leave `--`.
- `envelopes.tex`: the 7 `axis/<feature>/envelope` per robot x method, in
  Wojtek units (cm, deg).
- `compute.tex`: GPU-hours (`final/elapsed_h`), sim and student steps per
  robot x method, from `--train_summaries` (training run name ->
  wandb summary); not written without it.
- `--dump_wandb FILE`: fetches the training-run summaries for the prefixes
  from `dominikgalus9-wroc-aw-university-of-science-and-technology/wojtyla`
  and writes that JSON.
- Robot names in headers: `ROBOT_LABEL` (`wojtek` prints as W01-TEK).

## Paper figures — `scripts/paper_figures.py` {#paper-figures}

```sh
PYTHONPATH=src uv run python src/scripts/paper_figures.py \
  --dump_history history.json --prefix wojtek=v10_wojtek go1=v10_go1
PYTHONPATH=src uv run python src/scripts/paper_figures.py \
  --history history.json --prefix wojtek=v10_wojtek go1=v10_go1 --out artifacts/paper/figs
MUJOCO_GL=cgl JAX_PLATFORMS=cpu PYTHONPATH=src uv run python \
  src/scripts/paper_figures.py --terrains --out artifacts/paper/figs
```

- `--dump_history`: per finished run, `eval/d{0.5,1.0}/success_rate` and the
  `level/*` composition vs `env_steps` from wandb; rows with a NaN are dropped.
- `curves.pdf`: eval success vs env steps, robot x d panels, seed mean with
  min-max band.
- `composition.pdf`: mean physical value of the magnitude parameters and
  genome distance among training levels, Wojtek units, running mean over 9 points.
- `terrains.pdf` / `robots.pdf` (`scripts/paper_terrains.py`): offscreen MuJoCo
  renders of one level per feature at full strength plus uniform samples, and
  both robots on one shared genome. Needs the built assets (`just build`,
  `just build-accel-arena`); `MUJOCO_GL=cgl` on macOS.
- Overleaf: upload the PDFs into a `figs/` folder; `\resultfig` falls back to a
  placeholder while a file is missing.
- `results.csv`: one row per run, every value the tables use.
