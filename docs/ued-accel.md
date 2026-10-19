# UED / ACCEL

The research core: an adversarial curriculum that generates and replays
heightfield levels to train the student. Two halves: the **heightfield levels**
(`src/hfield/`) and the trainer that runs the curriculum,
`scripts/brax_ued.py` ([below](#brax-ued)).

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
node read-only and blocks CLI overrides like `arena.amplitude=...`);
one place for every knob. `patch_radius`, `cell_size` (fine
resolution), `coarse_n` (genome size), `amplitude` (peak height off the neutral
plane), `base_z`, `pad_radius`/`pad_taper` (flat spawn disc + ramp),
`grid_size` (property) is the fine
node count. Mutation strength is not here: it lives in `level_space`. The `arena` config group is one (`conf/arena/v10.yaml`). Full
field table: [configuration.md](configuration.md#hfield).

### `HFieldManager` (`hfield/manager.py`)

Pure given an rng; no hidden state beyond the config:

- `generate(rng)` — a fresh `coarse_n × coarse_n` uniform level.
- `mutate(level, rng, n_edits, edit_scale)` — nudge `n_edits` random cells by `±edit_scale`, clip to
  `[0,1]` (the ACCEL edit).
- `to_hfield_data(level)` — bilinear upsample to `grid_size`, apply the pad
  taper (flat centre disc), recentre on `0.5`; returns the `float32`
  `hfield_data` MuJoCo reads.
- `height_range(data)` — physical min/max height for the current amplitude.

### Why swapping a level is free

`hfield_data` is a `jax.Array` leaf of `mjx.Model`, so it is swappable
**with no MJX recompilation**; `hfield_size` is static. This is the whole
reason a heightfield works as an ACCEL genome at RL scale. Every level of an
iteration goes on its own tile of one shared arena (`hfield/tiles.py`), and
every env collides with that one `hfield_data` — the only layout the warp
backend handles. Mechanism (`tree_replace` in `FunctionalEnv`) and tile
geometry: [environment.md](environment.md#hfield-mutability),
[tiled arena](environment.md#tiled-arena). A level's semantics are
unchanged: genome → `render` → the same `grid_size²` patch, now written into
its tile by `TileLayout.assemble`.

### Arena assets {#arena}

`hfield/arena.py` writes the static **"accel"** MJCF arena the ACCEL env loads:
a flat (`data = 0.5`) heightfield with `tiles x tiles` level patches and
their gaps (`TileLayout`), box "apron" strips around it, plus the
spec/lookup sidecars — via `terrain.py`'s writers. The spec records the
layout (`tile_layout`: tiles per side, level grid, gap); `check_arena`
refuses an arena built for another layout, or a single-level arena from
before the tiles, with the rebuild command.
Two far corner nodes are pinned to `0`/`1`: MuJoCo rescales a loaded hfield
to span `[0, 1]`, so an all-`0.5` file compiles to `0` and sinks the static
ground to `-amplitude`. That broke `real_pose_ref` (its standing-height settle
runs on the static model, so every commanded height mapped to the most
extended pose and zero action tripped the four-bar toggle fall within
~0.35 s) and any rollout without an `hfield_data` swap.
Run `just build-accel-arena` before any script that loads the arena. The
per-iteration levels are assembled into this arena's `hfield_data` at
rollout time; the static asset just fixes the compile-time envelope.

## DR vs ACCEL on brax PPO — `scripts/brax_ued.py` {#brax-ued}

`just train mode=<option> name=<run> [overrides]` (`conf/train.yaml`;
every field: [configuration.md](configuration.md)).

History: an earlier in-repo PPO (own loss and model) did not learn to walk on
this env within ~0.6B steps, while brax's PPO (mujoco_playground recipe) did
in ~40M. That trainer and its ACCEL variants were deleted; `brax_ued.py` is
the only trainer. It keeps brax's PPO pieces (networks,
running obs normalizer, `compute_ppo_loss`, GAE, 20-step unrolls, 4 epochs x
32 minibatches, discount 0.97) and drives `FunctionalEnv`, because
`ppo.train` fixes each env's model for the whole run and cannot swap levels.

- Outer iteration: `num_lanes` levels (at most the arena's `tiles²`, 256)
  written into tiles 0..num_lanes-1 of the shared arena, `envs_per_lane`
  envs reset onto each tile (plus a fresh physics DR draw), stepped
  `level_steps` in 20-step unrolls; when training, a PPO update follows
  every unroll. Lane regret = mean positive GAE advantage over the
  iteration. `tile_leak_frac`: envs that reached a neighbouring tile
  ([tiled arena](environment.md#tiled-arena)).
- Physics backend: `backend` (default `auto`: warp on a GPU, jax on
  CPU), Wojtek and Go1 alike; v10 and earlier runs (the paper draft)
  trained on jax, so rerun them with `backend=jax` to compare. Throughput, measured
  on an H100: [Throughput](#throughput).
- `mode=dr`: fresh uniform levels at full difficulty, train every
  iteration.
- `mode=dr_easy`: DR over ACCEL's easy generator (control points within
  `generator_scale` of flat) — isolates the curriculum from "train easy".
- `mode=dr_mixed`: uniform levels scaled by a per-level difficulty
  drawn from `[0, 1]`.
- `mode=accel`: replay iterations (`replay_prob`, once the buffer holds
  `num_lanes` levels) train on rank+staleness-prioritized buffer levels,
  refresh their scores and queue their mutations; explore iterations do not
  train and score the queued children plus fresh easy levels (control points
  within `generator_scale` of flat). The buffer keeps the `buffer_size`
  highest-regret levels.
- ACCEL curriculum switches (`mode.*`; defaults reproduce plain ACCEL):
  - `mode.mutation=grow` — after the local edits, scale the level's deviation
    from flat by `U(1, grow_max)` (default 1.3). Plain symmetric edits
    smooth as often as they roughen; plain ACCEL's buffer stayed at the easy
    generator's roughness (~0.03 m) for the whole run.
  - `mode.score=learnability` — level score `p(1-p)` (SFL), `p` = fraction of
    the lane's envs that never fell and ended >= `min_travel` m out,
    instead of positive value loss.
  - `mode.generator=mixed` — explore iterations draw fresh levels from the
    `dr_mixed` distribution instead of the easy generator.
  - Logged: `buffer_roughness_mean`/`_p90`, `buffer_score_roughness_spearman`
    (rank correlation of level score vs roughness), `children_roughness`,
    `train_success`, `learnability_mean`.
  - Mode options `mode=accel_grow` (grow) and `mode=accel_sfl` (grow +
    learnability). `level_space/grid.yaml` composes 16 edits for both
    (8 otherwise), locally and via `brax_ued.sbatch` alike.
- `mode=paired`: PAIRED (port of `feat/paired-implementation`'s design
  onto the brax student). A learned adversary (`modules/generator.py`,
  latent seed -> sigmoid-squashed Gaussian over the genome) proposes
  `num_lanes` levels; protagonist and antagonist (two brax students) both
  train on them; regret = antagonist max return - protagonist mean return
  per level, clipped at 0, is the adversary's reward (single-step PPO, 5
  epochs, HiEnt `gen_entropy_coef=0.1`, no BC distillation yet).
  `env_steps`/`num_timesteps` count protagonist steps only; the antagonist
  doubles the real simulation cost (`sim_steps` counts both).
- `mode=plr`: robust PLR (PLR⊥) - ACCEL's buffer, rank/staleness replay
  and score refresh, but no mutation; explore iterations score fresh
  uniform (DR) levels.
- `robot=go1`: Playground's Go1 instead of Wojtek, see
  [Go1](#go1).
- Eval every `eval_every` iterations: deterministic policy on `eval_levels`
  fresh uniform levels (2 episodes each, in chunks of at most `tiles²`
  levels) at difficulties 0.25/0.5/0.75/1.0 — `eval/d<d>/success_rate` (no
  fall and >= `min_travel` m from spawn), fall rate, travel, forward vs
  commanded speed.
- Curriculum composition (`level_space=param` only), every iteration over
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
- WCSS: `brax_ued.sbatch` array (tasks 0-2 dr, 3-5 accel, seeds 0-2); assets
  must be built beforehand since tasks start together. The job interface
  is env vars (`MODES`, `SPACE`, `ROBOT`, `BACKEND`, `TAGS`, `PREFIX`,
  `NUM_TIMESTEPS`, `NSEEDS`, `SEED0`, `SRC`), turned into Hydra overrides:
  `MODES` entries are `mode` options with an optional `-s<scale>` suffix
  (`mode.generator_scale=<scale>`), `SPACE` the `level_space` option,
  `TAGS="a b"` → `tags=[a,b]`.
- The arena must match the run's `arena` config (`hfield.arena.check_arena`:
  `pad_radius`, height scale, level grid, tiles, gap); the error names the
  `just build-accel-arena arena...` command for it. Training and the
  builder read the same `arena` group, so the default needs no override.

## Throughput {#throughput}

H100, 8192 envs (256 lanes x 32), `scripts/profile_brax_ued.py`; full
numbers and the old loop's profile in `experiments/tiled-arena/report.md`.

- v10 loop (jax, per-env hfield swap): 12.9k env steps/s, ~9.7k end to
  end. Physics is 98% of an iteration; the per-env swap itself costs
  nothing (a shared level on jax is the same 12.9k).
- Tiled arena on warp: 263k env steps/s; a
  training iteration (2.46M steps, 15 unroll + PPO update pairs) ~10.8 s
  = ~227k steps/s; `evaluate` 17-23 s (jax: 106 s).
- 32M steps of `mode=accel` in ~5 min wall clock, ~2 min of it
  compilation (`t7_tiled_wojtek_accel_seed0`).
- Warp is not jax: a jax-trained v10 policy scores 1.5-3 points higher on
  terrain under warp and stands 7-11 mm higher; compare runs within one
  backend. On jax the tiled arena reproduces v10's final eval inside its
  95% CIs.
- Whole-device GPU memory: ~16 GB on warp (JAX preallocation off), so
  larger batches fit.

## Level spaces — `hfield/level_space.py` {#level-spaces}

`brax_ued.py`, `eval_brax.py` and `render_video.py` take
`level_space={grid,param}` (`conf/level_space/`); every
mode samples, mutates and renders levels through it.

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
  - ACCEL: `mutate` edits `level_space.n_edits` (2) parameters by
    ±`level_space.edit_scale` (0.1); `mode.mutation=grow` also scales the
    genome away from `flat`.
  - PAIRED: the generator emits the 8 parameters; a random seed is appended.
- WCSS: `slurm/brax_ued.sbatch` / `slurm/eval_brax.sbatch` take
  `SPACE=param`, `PREFIX` (run-name prefix, `v10`), `ROBOT`,
  `NUM_TIMESTEPS` (train), `NSEEDS` / `SEED0` (array task = mode x
  NSEEDS + seed offset; default seeds 0-2, extra seeds e.g. `NSEEDS=2
  SEED0=3`) and `SRC` (`<dir>/src` of a frozen copy with the
  built `assets/` and `conf/` beside it - `wojtek_env.paths` and
  `config.schema.CONF_DIR` resolve them from `src/..` - so queued jobs
  keep the submitted code and configs).
- `robot_scale` (`ParamSpace`, eval suite): every length range (amplitude,
  wavelength, stairs, gaps, blocks, gap depth/period) times the robot's size
  relative to Wojtek; slopes unchanged. Envelopes stay in Wojtek units.

## Go1 — `robots/go1.py` {#go1}

Second robot for the benchmark: open serial chain (Go1) vs Wojtek's closed
chain. `robot=go1` in every entry point with a `robot` group, `ROBOT=go1` on WCSS
(run prefix `v9_go1`).

- Playground `Go1JoystickRoughTerrain` (feet-only collisions, its reward,
  blind actor), on `backend` like Wojtek (warp: Playground's own
  rough-terrain Go1 trains on warp; on this arena not yet compared with
  jax), with:
  - the tiled ACCEL arena, written by `scene_xml` (no asset files): one
    `hfield` geom `floor` (geom 0, so Playground's friction DR hits it),
    sized by `HFieldConfig`; `reset` spawns on `spawn_xy`, the tile centre
    `FunctionalEnv` sets;
  - forward commands: Wojtek's `vx (0.2, 0.6)`, `vy (-0.2, 0.2)` times
    `GO1_SCALE`, `wz (-0.4, 0.4)`;
  - Wojtek's fall rule at Go1's size: trunk below 0.6 x 0.258 m above the
    terrain under it, or gravity z > -0.4;
  - critic gets a clean 5x5 height scan (Wojtek's grid x `GO1_SCALE`).
- `GO1_SCALE = 2.5` (standing trunk 0.258 m vs Wojtek 0.10 m) scales the
  arena (`hfield_config`, `reach` included), level lengths, commands and
  `min_travel`.
- Physics DR: Playground's `go1/randomize.py`.
- Needs `mujoco_menagerie` under Playground's `external_deps` (meshes);
  `ensure_menagerie_exists()` clones it, by hand on WCSS (no AVX on login).

## Final eval — `scripts/eval_brax.py` {#eval-brax}

`just eval checkpoint=<run>.pkl name=<run>_final_eval out_dir=<dir>`
(`conf/eval.yaml`; pass the run's `robot`, `level_space`, `arena`).
Loads a `brax_ued.py` checkpoint, deterministic
policy, no physics DR, no `no_progress` cut. Success = no fall and >=
`min_travel` m from spawn after `level_steps`. Levels run in chunks of
`min(chunk_envs / episodes, tiles²)` levels, level i of a chunk on tile i;
`backend` as in training (score a checkpoint on the backend it trained
on).

- `random/d<d>`: `num_levels` (2048) genome levels, the same genomes at
  each difficulty 0.25/0.5/0.75/1.0.
- `suite/<terrain>_<k>`: held-out structured terrains no method trains on,
  written straight into a tile, radial so heading does not matter:
  stairs up/down (2k cm steps every 0.3 m), slopes up/down (5k°), trenches
  (10 cm deep, 5k cm wide, every 0.5 m), waves (±2k cm, 0.6 m), random
  blocks (±2k cm, 0.3 m cells); k = 1..4. Heights clip at ±`amplitude`.
- **CVaR** (`cvar<alpha>_success`, alpha 0.1 / 0.25): every level runs in two
  independent passes; the worst-alpha levels ranked on one pass are scored
  on the other, both ways, averaged (cross-fitting). Ranking and scoring
  on the same episodes is biased low by selection noise: with 2000 levels
  of true success 0.5 and 4 episodes each, same-pass CVaR25 reads 0.19,
  cross-fitted 0.49.
- `level_space=param` adds `param/uniform` (2048 levels drawn from the
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
- Writes `<name>.json` (metrics) and `<name>.npz` to `out_dir`, and the
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
- `CpuPhysics`: one `MjModel` copy (of the tiled arena) + `MjData` per
  thread (`sched_getaffinity` threads), contiguous env slices;
  `set_arena` takes each unroll's arena `hfield_data`, which every thread
  copies in once. Per env and control
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
  `success_rate`; `<variant>/elapsed_s`. One pass (`episodes` per level),
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
  levels x episodes) to `out_dir`; wandb tags `sim2sim`, robot.
- `just sim2sim checkpoint=<pkl> name=<name> out_dir=<dir>`
  (`conf/sim2sim.yaml`; always `param` levels on the jax env).
- WCSS: `slurm/eval_sim2sim.sbatch`, same `MODES`/seed array layout, `SRC`,
  `ROBOT`, `PREFIX` as `eval_brax.sbatch`; `lem-cpu-short`, 16 cores (CPU
  only; Lem CPU nodes share the GPU nodes' AlmaLinux 9, so the venv runs).

## Demo videos — `scripts/render_video.py` {#render-video}

mp4 clips of a `brax_ued.py` checkpoint walking generated levels, for demos
(not an eval: no metrics beyond a printed fall time and travel per clip).

```sh
just render-video checkpoint=artifacts/checkpoints/<run>.pkl \
  'levels=[0.5:<seed>,0.75:<seed>,1.0:<seed>]'
```

- Env, network and level space are rebuilt from the `robot`,
  `level_space`, `arena` groups and `backend` (`conf/render.yaml`; defaults =
  v10: `param` levels, `arena=v10`): pass the ones the run trained with.
- Needs the run's arena built with its geometry (v10: `just
  build-accel-arena`). For Wojtek the script checks the arena's spec (`pad_radius`, height scale, tile layout) first and fails with
  the command filled in; an amplitude mismatch would otherwise run
  silently.
- `levels`: list of `D:SEED`, each `space.scaled(default_rng(SEED), 1, D)`,
  one clip each, named `<checkpoint stem>_d<D>_s<SEED>.mp4`; `demo.mp4` in
  the same `out_dir` (default `artifacts/videos`) = all clips in order.
- Rollout on MJX (eval env: no physics DR, no `no_progress` cut, pushes on),
  deterministic policy, `command=[VX,VY,WZ]` held every step (default
  0.3 m/s forward, 0.2 rad/s yaw: an arc; on 24 screened d=1 levels the
  robot got 1.1-3.2 m from spawn in 20 s, so it stays on the 3 m arena).
  The command's height entry keeps the reset draw; the resample counter
  is pinned every step (Wojtek `steps_since_cmd` = 0, Go1
  `steps_until_next_cmd` = 2^30) so the env never resamples.
- `seconds` (default 20) must fit in one episode (1000 steps = 20 s for
  Wojtek): a fall or truncation auto-resets the robot to spawn, which the
  clip would show as a jump. Pick levels whose printed fall list is empty;
  screen candidates with this single-env rollout, not a vmapped batch:
  the batch's MJX numerics differ enough to flip falls (`d=1.0` seed 19
  passed batched, fell twice single-env).
- Levels with a strong slope clip at ±`amplitude` (0.3 m) within ~1 m of
  the pad (d=1 slope up to -20°), leaving a flat pit floor; for a demo,
  prefer low-slope levels whose rough/stairs/gaps persist outward.
- The level runs on tile 0 of the arena (the rest flat). Rendering
  replays the recorded `qpos`, shifted to the origin, on a one-tile copy
  of the scene (`hfield.arena.single_tile_spec`; Go1: `scene_xml` of
  `single_tile`), through `mujoco.Renderer`
  (`width`/`height`, default 1280x720, `fps` 30, frames at the
  nearest control step), camera tracking the free-joint body; mp4 via
  `mediapy` (needs `ffmpeg` on `PATH`). `MUJOCO_GL`: `cgl` on macOS,
  `egl` on a GPU node, `osmesa` on a CPU node (recipe defaults to cgl/egl).
- Cost (Apple M-series, CPU jax): ~1 min compile, then ~1 min per 20 s
  clip.
- Team demo (2026-10-15): `v10_wojtek_accel_seed4` (best of the 5 v10
  Wojtek ACCEL seeds by final eval: `random/d1.0` success .423,
  `param/uniform` .412, suite .584, flat .936), level `1.0:17` (rough
  8.5 cm / 0.33 m wavelength, stairs -3.2 cm, gaps 2.3 cm, slope 1.2°):
  no fall, 2.89 m from spawn after 20 s.

## Paper tables — `scripts/paper_tables.py` {#paper-tables}

Builds the workshop paper's tables from `eval_brax.py` outputs
(`<prefix>_<method>_seed<k>_final_eval.{json,npz}`); bare booktabs
`tabular`s; the paper (on Overleaf) owns float, caption and label.

```sh
just paper-tables dump_wandb=summaries.json \
  prefixes.wojtek=v10_wojtek prefixes.go1=v10_go1
just paper-tables eval_dir=<evals> prefixes.wojtek=v10_wojtek \
  prefixes.go1=v10_go1 train_summaries=summaries.json
```

- `prefixes.<robot>=<prefix>` (`conf/paper_tables.yaml`; a robot left
  `null` is skipped): one column group / row group per robot. Known methods get paper names (DR, DR-easy, PLR⊥, ACCEL,
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
  robot x method, from `train_summaries` (training run name ->
  wandb summary); not written without it.
- `dump_wandb=FILE`: fetches the training-run summaries for the prefixes
  from `dominikgalus9-wroc-aw-university-of-science-and-technology/wojtyla`
  and writes that JSON.
- Robot names in headers: `ROBOT_LABEL` (`wojtek` prints as W01-TEK).

## Paper figures — `scripts/paper_figures.py` {#paper-figures}

```sh
just paper-figures dump_history=history.json \
  prefixes.wojtek=v10_wojtek prefixes.go1=v10_go1
just paper-figures history=history.json \
  prefixes.wojtek=v10_wojtek prefixes.go1=v10_go1
JAX_PLATFORMS=cpu just paper-figures terrains=true
```

- `conf/paper_figures.yaml`; output in `out` (default `artifacts/paper/figs`).
- `dump_history`: per finished run, `eval/d{0.5,1.0}/success_rate` and the
  `level/*` composition vs `env_steps` from wandb; rows with a NaN are dropped.
- `curves.pdf`: eval success vs env steps, robot x d panels, seed mean with
  min-max band.
- `composition.pdf`: mean physical value of the magnitude parameters and
  genome distance among training levels, Wojtek units, running mean over 9 points.
- `terrains.pdf` / `robots.pdf` (`scripts/paper_terrains.py`): offscreen MuJoCo
  renders of one level per feature at full strength plus uniform samples, and
  both robots on one shared genome. Needs the built assets (`just build`,
  `just build-accel-arena`); the recipe sets `MUJOCO_GL` as `render-video`
  does, and `arena` (default `v10`) is the arena the levels render on.
- Overleaf: upload the PDFs into a `figs/` folder; `\resultfig` falls back to a
  placeholder while a file is missing.
- `results.csv`: one row per run, every value the tables use.
