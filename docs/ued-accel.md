# UED / ACCEL

The research core: an adversarial curriculum that generates and replays
heightfield levels to train the student. Two halves: the **heightfield levels**
(`src/hfield/`) and the trainer that runs the curriculum, `src/training/`
(entry point `scripts/brax_ued.py`, [below](#brax-ued)).

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
node count. Mutation strength is not here: it lives in `level_space`. The
`arena` config group has one option, `benchmark`
(`conf/arena/benchmark.yaml`). Full field table:
[configuration.md](configuration.md#hfield).

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
refuses an arena built for another layout, with the rebuild command.
Two far corner nodes are pinned to `0`/`1`: MuJoCo rescales a loaded hfield
to span `[0, 1]`, so an all-`0.5` file would compile to `0` and sink the
static ground to `-amplitude` — wrong for `real_pose_ref`'s standing-height
settle (it runs on the static model) and for any rollout without an
`hfield_data` swap.
Run `just build-accel-arena` before any script that loads the arena. The
per-iteration levels are assembled into this arena's `hfield_data` at
rollout time; the static asset just fixes the compile-time envelope.

## DR vs ACCEL on brax PPO — `scripts/brax_ued.py` {#brax-ued}

Code: `src/training/` (`student.py`, `iteration.py` = one outer iteration,
`curricula.py` = the modes below, `train.py` = the loop and logging).

`just train mode=<option> name=<run> [overrides]` (`conf/train.yaml`;
every field: [configuration.md](configuration.md)).

`brax_ued.py` uses brax's PPO pieces with the mujoco_playground recipe
(networks, running obs normalizer, `compute_ppo_loss`, GAE, 20-step unrolls,
4 epochs x 32 minibatches, discount 0.97) and drives `FunctionalEnv` itself,
because `ppo.train` fixes each env's model for the whole run and cannot swap
levels.

- Outer iteration: `num_lanes` levels (at most the arena's `tiles²`, 256)
  written into tiles 0..num_lanes-1 of the shared arena, `envs_per_lane`
  envs reset onto each tile (plus a fresh physics DR draw), stepped
  `level_steps` in 20-step unrolls; when training, a PPO update follows
  every unroll. Lane regret = mean positive GAE advantage over the
  iteration. `tile_leak_frac`: envs that reached a neighbouring tile
  ([tiled arena](environment.md#tiled-arena)).
- Physics backend: `backend` (default `auto`: Wojtek warp on a GPU, jax on
  CPU; the Playground robots always jax, [robots](#robots)). Compare runs
  within one backend
  ([Throughput](#throughput)).
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
    smooth as often as they roughen, so plain ACCEL's buffer stays near the
    easy generator's roughness (~0.03 m on `grid`).
  - `mode.score=learnability` — level score `p(1-p)` (SFL), `p` = fraction of
    the lane's envs that never fell and ended >= `min_travel` m out,
    instead of positive value loss.
  - `mode.generator=mixed` — explore iterations draw fresh levels from the
    `dr_mixed` distribution instead of the easy generator.
  - Logged: `buffer_roughness_mean`/`_p90`, `buffer_score_roughness_spearman`
    (rank correlation of level score vs roughness), `children_roughness`,
    `train_success`, `learnability_mean`.
  - Mode options `mode=accel_grow` (grow) and `mode=accel_sfl` (grow +
    learnability). `level_space=grid` composes 16 edits for both (8
    otherwise); `param` edits 2 parameters either way.
- `mode=paired` / `paired_uni` / `paired_bi` / `paired_evo`: PAIRED and
  its variants, two students; see [PAIRED](#paired).
- `mode=plr`: robust PLR (PLR⊥) - ACCEL's buffer, rank/staleness replay
  and score refresh, but no mutation; explore iterations score fresh
  uniform (DR) levels.
- `mode=sfl`: PLR⊥ scored by learnability `p(1-p)` instead of positive
  value loss — an approximation of SFL (Rutherford et al. 2024), not the
  original: the buffer persists with rank/staleness replay and explore
  iterations do not train, where SFL periodically rebuilds a top-K buffer
  from a large random pool and trains on a buffer/random mix.
- `robot=go1|barkour|spot|go2`: a Playground quadruped instead of Wojtek,
  see [Playground robots](#robots).
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
- SLURM: `slurm/brax_ued.sbatch`, job interface in the
  [README](../README.md#slurm-job-interface); assets must be built
  beforehand since the array's tasks start together.
- The arena must match the run's `arena` config (`hfield.arena.check_arena`:
  `pad_radius`, height scale, level grid, tiles, gap); the error names the
  `just build-accel-arena arena...` command for it. Training and the
  builder read the same `arena` group, so the default needs no override.

## PAIRED variants {#paired}

Two brax students (protagonist = the checkpointed deliverable, antagonist),
each with its own params, optimizer and running obs normalizer. PAIRED
(Dennis et al. 2020) and the variants of *Stabilizing UED with a Learned
Adversary* (Mediratta et al. 2023).

| `mode=` | level source | students | adversary |
|---|---|---|---|
| `paired` | learned adversary | both train every iteration | trained on regret |
| `paired_uni` | as `paired` | + UniBC | as `paired` |
| `paired_bi` | as `paired` | + BiBC | as `paired` |
| `paired_evo` | ACCEL's buffer / mutation | train on replay iterations | none |

`mode.bc` is a field, so the presets are just `paired` + `bc`; BC on
PAIRED+Evo is `mode=paired_evo mode.bc=uni|bi`. `mode.bc` on a one-student
curriculum (`dr*`, `accel`, `plr`) fails at startup.

- **PAIRED regret** (`training/paired.py:paired_regret`), per level over its
  `envs_per_lane` episodes: `max(antag max return - prot mean return, 0)`.
  Both students run the same levels in separate rollouts (own reset keys
  and physics DR draws), protagonist first.
- **Adversary** (`mode=paired*`): `training/adversary.py` + `training/generator.py`, latent seed ->
  sigmoid-squashed Gaussian over the genome (`param`: the 8 parameters, a
  random seed appended), `num_lanes` levels per iteration; reward = PAIRED
  regret; single-step PPO, 5 epochs, clip 0.2, batch-mean baseline,
  `gen_entropy_coef` (0.1, HiEnt on the teacher), `gen_learning_rate`.
- **Student HiEnt**: `mode.student_entropy_coef` is brax's `entropy_cost`
  for every student of every mode; default `0.01` = the mujoco_playground
  recipe. HiEnt (Mediratta et al. 2023) = raising it on both PAIRED
  students (e.g. `mode.student_entropy_coef=0.05`).
- **UniBC / BiBC** (paper Sec. 4.3): adds `bc_coef * KL(pi_teacher ||
  pi_student)` to the student's PPO loss, teacher stop-gradient, on the
  student's own minibatch observations.
  - `uni`: protagonist <- antagonist. `bi`: also antagonist <- protagonist,
    the protagonist as this iteration's updates left it (it trains first).
  - Teacher = the other student's policy params and its own normalizer
    (`Student.as_teacher`), fixed during the student's iteration.
  - KL (`training/paired.py:policy_kl`): brax's `NormalTanhDistribution`
    turns the policy logits into `loc`, `softplus(raw) + 0.001`; both
    students push the Gaussian through the same tanh, and KL is invariant
    under a shared bijection, so the closed-form diagonal-Gaussian KL of
    the base distributions is the KL of the squashed policies.
  - Applied when `it % bc_every == 0` (`bc_coefs`); `bc_coef` reaches the
    jitted update as a traced scalar (0 on other iterations), so no
    recompile. A student that never distils (`none`; the antagonist under
    `uni`) gets no BC term at all, so `mode=paired` is plain PAIRED.
- **PAIRED+Evo** (`mode=paired_evo`): `accel_iteration` unchanged, with
  `score_levels` -> `run_pair`, so the level score is PAIRED regret over
  both students instead of positive value loss (`mode.score` must stay
  `pvl`).
  - Explore iterations: queued children plus fresh levels from ACCEL's
    explore generator (`mode.generator`), rolled out by both students
    without updates, scored, added to the buffer.
  - Replay iterations (`replay_prob`): rank+staleness-sampled buffer
    levels, both students train (HiEnt + `mode.bc`), scores refreshed,
    mutations (`mode.mutation`) queued and scored on the next explore
    iteration (ACCEL's order).
- **Logged** (all four): `paired_regret_mean`, `prot_return`,
  `antag_return`, `antag_forward_vel`, `prot_bc_kl` / `antag_bc_kl`
  (mean KL(other student || this one) over the iteration's observations,
  after each update; logged under `none` too, as a policy-distance
  diagnostic), `bc_coef` (the protagonist's weight this iteration, 0 when
  off or not training); `gen_loss` on `paired*`; the ACCEL buffer stats
  on `paired_evo`. Student metrics without a prefix are the protagonist's.
- **Cost**: every iteration simulates both students, explore iterations of
  `paired_evo` included: `sim_steps` = 2 x `env_steps`;
  `env_steps`/`num_timesteps` and `student_train_steps` count protagonist
  steps. BC adds one teacher policy forward per minibatch and one KL
  forward per unroll (physics is ~98% of an iteration). Each student jits
  its own update, so compile time roughly doubles (CPU smoke, 2 x 16 envs:
  ~2.5 min to the first logged iteration).

## Throughput {#throughput}

Wojtek, one H100, 8192 envs (256 lanes x 32), 300-step levels;
`just profile` (`scripts/profile_brax_ued.py`, `diagnostics/throughput.py`)
measures it.

| backend | env steps/s | training iteration | `evaluate` | final eval |
|---|---|---|---|---|
| warp | 263k | ~10.8 s (2.46M steps) = ~227k steps/s | 17-23 s | ~8.5 min |
| jax | 12.9k | ~9.7k steps/s end to end | 106 s | ~55 min |

- Physics is ~98 % of an iteration on jax; DR draw, PPO update, regret and
  level rendering together < 5 %.
- 32M steps of `mode=accel` on warp: ~5 min wall clock, ~2 min of it
  compilation. A 300M-step run: ~0.5 GPU-h (PAIRED variants simulate two
  students: ~1 GPU-h). Playground robots: not profiled.
- Whole-device GPU memory on warp: ~16 GB (JAX preallocation off).
- Warp is not jax: one checkpoint scores 1.5-3 points higher success on
  terrain under warp (flat unchanged) and stands 7-11 mm higher on every
  tile. Train, eval and compare within one backend.

## Level spaces — `hfield/level_space.py` {#level-spaces}

`brax_ued.py`, `eval_brax.py` and `render_video.py` take
`level_space={param,grid}` (`conf/level_space/`, default `param`); every
mode samples, mutates and renders levels through it.

- `param` (the benchmark space): 8 terrain parameters in [0, 1] plus a
  noise seed (last entry, never mutated). `PARAMS` maps them to physical
  ranges sized to the ~0.10 m robot: noise amplitude 0–0.10 m and
  wavelength 0.2–1.0 m, radial slope −20…+18°, radial stairs −0.08…+0.05 m
  high and 0.15–0.60 m wide, gaps 0–0.06 m wide (0.15 m deep, every 0.6 m),
  blocks 0–0.06 m high and 0.15–0.60 m wide. Features are radial (or seeded
  fields) so heading does not matter.
  - Uphill slope, step-up, gap and block maxima sit at the 50 %-success
    envelope (`axis/*/envelope`) of policies trained on a wider box (slope
    ±20°, stairs ±0.08 m, gaps 0–0.25 m, blocks 0–0.10 m), where the best
    methods solved ~8 % of uniform levels. Features compound, so uniform
    levels range from trivial to unsolved.
  - `flat` = genome of zero slope/step, so it is off 0.5 on asymmetric ranges.
  - DR: uniform genome; `scaled(d)` shrinks it toward `flat` (no features)
    by d; `easy(s)` = `scaled(s)`; DR-mixed draws d per level.
  - ACCEL: `mutate` edits `level_space.n_edits` (2) parameters by
    ±`level_space.edit_scale` (0.1); `mode.mutation=grow` also scales the
    genome away from `flat`.
  - PAIRED: the generator emits the 8 parameters; a random seed is appended.
- `grid`: a `coarse_n x coarse_n` control-point genome of i.i.d. noise, so
  levels at one scale are statistically alike: per-level success
  correlates only 0.02–0.17 between eval passes, and ACCEL lands within
  seed noise of DR-easy.
- `robot_scale` (`ParamSpace`, eval suite): every length range (amplitude,
  wavelength, stairs, gaps, blocks, gap depth/period) times the robot's size
  relative to Wojtek; slopes unchanged. Envelopes stay in Wojtek units.

## Playground robots — `robots/` {#robots}

Benchmarks for other morphologies: open serial chains (Go1, Go2, Barkour,
Spot) vs Wojtek's closed chain. `robot=<name>` in every entry point with a
`robot` group, `ROBOT=<name>` in `slurm/`.

| `robot=` | class | model | `SCALE` | standing trunk (m) | DR |
|---|---|---|---|---|---|
| `go1` | `Go1Terrain` | Playground `go1_mjx_feetonly.xml` | 2.5 | 0.258 | Playground `go1/randomize.py` |
| `barkour` | `BarkourTerrain` | menagerie `google_barkour_vb/barkour_vb_mjx.xml` | 2.4 | 0.235 | none (Playground has none) |
| `spot` | `SpotTerrain` | Playground `spot_mjx_feetonly.xml` | 4.3 | 0.433 | none (Playground has none) |
| `go2` | `Go2Terrain` | menagerie `unitree_go2/go2_mjx.xml`, rewritten feet-only | 2.5 | 0.254 | Go1's `randomize.py` |

Shared (`robots/terrain.py` `TerrainRobot`, a mixin over the Playground env;
a robot file keeps its constants, scene and what differs):

- Playground's own reward, obs and PD gains (Go2: menagerie's), feet-only
  collisions, blind actor.
- jax backend only (`playground_backend`): `backend=auto` resolves to jax,
  `backend=warp` is refused. On warp, foot contacts with the tiled arena's
  hfield blow up more often the farther the tile is from the arena origin
  (random actions on the 16 x 16 benchmark arena: Go1 1.5 % of steps on the
  inner tiles to 5 % on the outer ones, Go2 to 3.4 %, Barkour to 0.5 %;
  0 % on a 2 x 2 arena and on jax; ccd iterations/tolerance, `base_z` and
  amplitude do not change it). Every blow-up ends the episode, so the
  policy learns to stand still. Go1 on jax runs at ~410k env steps/s on an
  H100 (warp: ~490k).
- The tiled ACCEL arena written by `arena_scene` (no asset files): one
  `hfield` geom `floor` (geom 0, so a friction DR keyed on geom 0 hits it),
  sized by `HFieldConfig`; `reset` spawns on `spawn_xy`, the tile centre
  `FunctionalEnv` sets.
- Commands: the forward slice of each robot's own Playground command box
  (`native_command_range`): vx uniform from 0 to the robot's max, vy and
  wz uniform over their native ranges, no standing commands;
  `sample_command` redraws from the same box on the env's own schedule.

  | robot | vx (m/s) | vy (m/s) | wz (rad/s) | source |
  |---|---|---|---|---|
  | Go1, Go2 | 0-1.5 | +-0.8 | +-1.2 | `go1/joystick.py` `command_config.a` |
  | Barkour | 0-1.5 | +-0.8 | +-0.7 | `barkour/joystick.py` `lin_vel_x/y`, `ang_vel_yaw` |
  | Spot | 0-1.0 | +-0.8 | +-1.0 | `spot/joystick.py` `command_config` |

  vx starts at 0: with only fast commands (Wojtek's vx 0.2-0.6 x `SCALE`,
  Go1 0.5-1.5 m/s) Go1 still stands still at 200M steps on flat levels;
  from 0 it tracks the command by ~170M.
- Eval commands (`make_env(..., eval_travel=)`, used by brax_ued's eval,
  eval_brax and eval_sim2sim): vx only from `EVAL_SPEED_MARGIN` (1.2) x
  threshold / episode time up to the max, so every eval command can reach
  the success threshold (`min_travel` x `SCALE` in `level_steps`): Go1, Go2
  0.5-1.5, Barkour 0.48-1.5, Spot 0.86-1.0 m/s at the benchmark's 1 m and
  300 steps. Training draws the full forward slice, so train success tops
  out below 1.
- Wojtek's fall rule at the robot's size: trunk below 0.6 x `STANDING_HEIGHT`
  above the terrain under it (`terrain_height`, read off `hfield_data`), or
  gravity z > -0.4 in the trunk frame.
- The critic gets a clean 5x5 height scan (Wojtek's grid x `SCALE`) appended
  to `privileged_state`.
- `SCALE` = standing trunk height / Wojtek's 0.10 m, rounded to 0.1. The
  trunk height is measured: the settled `qpos[2]` over flat ground under zero
  action (the PD pose), which sags below the kinematic height.
  `tests/robots/test_terrain_robots.py` re-measures it per robot. `SCALE`
  scales the arena (`hfield_config`, `reach` included), level lengths and
  `min_travel`; slopes do not scale.
- Needs `mujoco_menagerie` under Playground's `external_deps` (meshes);
  `ensure_menagerie_exists()` clones it at the pinned commit
  ([README](../README.md#2-assets)); on a cluster whose login node cannot
  import mujoco, clone it on a compute node or by hand.
- `eval_sim2sim`'s `full_collision` variant exists for Go1 and Spot only
  (`FULL_COLLISION`); Barkour has no full-collision model and Go2's is not
  wired, so those two refuse it with an error.

Per robot:

- **Go1**: Playground `Go1JoystickRoughTerrain`, spawning 0.35 m above the
  tile.
- **Barkour vB**: Playground `barkour.Joystick` (18 dof, `Kp` 35, 12 joints
  clipped to its limits, perturbation kicks always on). Its env builds from a
  fixed menagerie scene and returns one obs vector (a 15-step history, 465
  values), so `BarkourTerrain` builds the arena scene itself, adds the foot
  contact sensors in XML (Playground adds them with `MjSpec`), uses the
  vector as the actor obs and `[obs, height scan]` as the critic obs, and
  carries a copy of `step` whose termination is the shared fall rule plus
  Playground's joint-limit check: its own `torso z < 0.18` is a world height,
  wrong on a pit or raised tile. Commands are redrawn every 500 steps.
- **Spot**: Playground `spot.Joystick` (`Kp` 300, `Kd` 1, 3-step joint-error
  history, feet position obs). `get_gravity` there is the up vector, so the
  fall rule negates it. Playground's `early_termination` flag and `gravity z <
  0.85` termination are replaced by the shared rule. Spot's standing trunk is
  0.433 m, scale 4.3, `reach` 43 m.
- **Go2**: Playground has no Go2 env: `Go2Terrain` is `Go1Terrain` (Go1's
  joystick reward and obs) on the menagerie model with the model's own PD
  gains (`kp` 50, `kv` 0.5, joint damping 2); Go1's `Kp` 35 sags the 15 kg
  Go2 to 0.219 m.
  `go2_mjx.xml` collides every link and lacks the names Go1's env reads, so
  `feet_only_model` rewrites it (each substitution must match exactly once):
  trunk `base` -> `trunk`, foot sites `FL_foot` -> `FL`, only the four foot
  spheres collide (no hfield-sphere `margin` in MJX), the `upvector`,
  `local_linvel`, per-foot `*_global_linvel` and `*_floor_found` sensors are
  added, the model's own keyframe is dropped. Go1's randomization fits
  (hfield = geom 0, trunk = body 1, 12 joints = dofs 6:).

## Final eval — `scripts/eval_brax.py` {#eval-brax}

Code: `src/evaluation/final_eval.py` (sets, metrics), `suite.py`, `stats.py`,
`rollout.py`.

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
  rng/key, so it does not shift the other sets' levels and keys.
- 95% bootstrap CIs over levels for mean and CVaR; `level_success_corr`
  between passes; suite CVaR is over the 28 terrain×strength cells.
- Writes `<name>.json` (metrics) and `<name>.npz` to `out_dir`, and the
  metrics to the wandb run summary. npz arrays (pass x level x episode
  unless noted):
  - `random_d<d>_fell`, `random_d<d>_travel`; `random_levels` (levels x
    genome): the d=1 base genomes of the `random/d*` sets.
  - `suite_success` (28 terrain x strength cells).
  - `param` only: `param_uniform_fell`, `param_uniform_travel`,
    `param_uniform_levels` (levels x genome).
- SLURM: `slurm/eval_brax.sbatch` (same task layout as `brax_ued.sbatch`).

## Sim2sim — `scripts/eval_sim2sim.py` {#sim2sim}

Code: `src/evaluation/sim2sim.py` (sets, metrics), `cpu_physics.py`,
`physics_variants.py`.

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
  | `half_dt` | timestep 0.004 -> 0.002 s, 2 CPU steps per training substep (policy rate unchanged); solref time constants below the training 2·dt floor pinned at 0.008 s | integration-step sensitivity only |
  | `full_collision` | Wojtek: the source model's per-link `*_collision` meshes collide with the terrain (`build_model` disables them; spec recompiled for convex hulls). Go1 and Spot: Playground's `go1_mjx_fullcollisions.xml` / `spot_mjx.xml`; Barkour, Go2: refused | contacts the MJX model drops (shins, thighs, trunk) |
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
- Bridge check (Wojtek ACCEL checkpoint, 16 flat / 64 uniform / 16
  per suite cell, 4 episodes, same levels and keys): `mjx_match` vs MJX
  (eval_brax rollout) success flat .922 / .922, suite 1-2 .778 / .777,
  param/uniform drop +.02 (95% CI -.016….059). One control step from the
  same state differs by up to 1.6 rad/s in qvel at the training solver
  (2 iterations) but 0.01 at 50 iterations: the gap is the unconverged
  solver, not the bridge.
- Go1 does **not** match on heightfields, and the bridge is not the cause.
  32 flat / 64 uniform / 32 per suite cell, 4 episodes, MJX vs
  `mjx_match` success:

  | Checkpoint | flat | param/uniform | suite 1-2 |
  |---|---|---|---|
  | Go1 DR-easy | .92 / .08 | .25 / .01 | .78 / .06 |
  | Go1 ACCEL | .71 / .05 | .11 / .00 | .58 / .03 |

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
- SLURM: `slurm/eval_sim2sim.sbatch`, same task layout and variables as
  `eval_brax.sbatch`; CPU partition, 16 cores (the CPU nodes must run the
  OS the venv was built on).

## Demo videos — `scripts/render_video.py` {#render-video}

Code: `src/rendering/video.py`, `scene.py`.

mp4 clips of a `brax_ued.py` checkpoint walking generated levels, for demos
(not an eval: no metrics beyond a printed fall time and travel per clip).

```sh
just render-video checkpoint=artifacts/checkpoints/<run>.pkl \
  'levels=[0.5:<seed>,0.75:<seed>,1.0:<seed>]'
```

- Env, network and level space are rebuilt from the `robot`,
  `level_space`, `arena` groups and `backend` (`conf/render.yaml`; defaults
  `param`, `arena=benchmark`): pass the ones the run trained with.
- Needs the run's arena built with its geometry (`just
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
  is pinned every step (Wojtek `steps_since_cmd` = 0, Go1/Go2
  `steps_until_next_cmd` = 2^30, Barkour/Spot `step` = 0, which also stops
  Barkour's kicks) so the env never resamples.
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
  of the scene (`hfield.arena.single_tile_spec`; Playground robots: `scene_xml` of
  `single_tile`), through `mujoco.Renderer`
  (`width`/`height`, default 1280x720, `fps` 30, frames at the
  nearest control step), camera tracking the free-joint body; mp4 via
  `mediapy` (needs `ffmpeg` on `PATH`). `MUJOCO_GL`: `cgl` on macOS,
  `egl` on a GPU node, `osmesa` on a CPU node (recipe defaults to cgl/egl).
- Cost (Apple M-series, CPU jax): ~1 min compile, then ~1 min per 20 s
  clip.

## Paper tables — `scripts/paper_tables.py` {#paper-tables}

Code: `src/paper/tables.py`, `runs.py` (loading, solvable subset), `latex.py`.

Builds the workshop paper's tables from `eval_brax.py` outputs
(`<prefix>_<method>_seed<k>_final_eval.{json,npz}`); bare booktabs
`tabular`s; the paper (on Overleaf) owns float, caption and label.

```sh
just paper-tables dump_wandb=artifacts/paper/summaries.json
just paper-tables eval_dir=<evals> \
  train_summaries=artifacts/paper/summaries.json
```

- `prefixes.<robot>=<prefix>` (`conf/paper_tables.yaml`, default
  `ued_<robot>`; a robot set to `null` is skipped): one column group / row
  group per robot. Every `conf/mode` option has a paper name
  (`paper/labels.py`), in this order: DR, DR-easy, DR-mixed, PLR⊥, SFL,
  ACCEL, ACCEL-grow, ACCEL-SFL, PAIRED, PAIRED+UniBC, PAIRED+BiBC,
  PAIRED+Evo; other methods follow under their own name; missing ones are
  skipped.
- `main.tex`: success (%) per method x robot: `param/uniform` mean, its
  CVaR25, solvable subset, `suite/success_rate`, `flat/success_rate`;
  mean ± std over seeds, best per column in bold, `--` if absent.
- Solvable subset: `param/uniform` levels that at least one run of the robot
  (any method) passes on >= 1 episode of pass A; each run is scored on pass
  B on them and vice versa, averaged (cross-fitted, like CVaR). Needs
  `param_uniform_travel` (`level_space=param` evals); otherwise it warns
  and leaves `--`.
- `envelopes.tex`: the 7 `axis/<feature>/envelope` per robot x method, in
  Wojtek units (cm, deg).
- `compute.tex`: GPU-hours (`final/elapsed_h`), sim and student steps per
  robot x method, from `train_summaries` (training run name ->
  wandb summary); not written without it.
- `dump_wandb=FILE`: fetches the training-run summaries for the prefixes
  from the wandb project `ENTITY/PROJECT` (`paper/runs.py`) and writes
  that JSON.
- Robot names in headers: `ROBOT_LABEL` (`wojtek` prints as W01-TEK).

## Paper figures — `scripts/paper_figures.py` {#paper-figures}

Code: `src/paper/figures.py`, `terrains.py`.

```sh
just paper-figures dump_history=artifacts/paper/history.json
just paper-figures history=artifacts/paper/history.json
JAX_PLATFORMS=cpu just paper-figures terrains=true
```

- `conf/paper_figures.yaml`; output in `out` (default `artifacts/paper/figs`).
- `dump_history`: per finished run, `eval/d{0.5,1.0}/success_rate` and the
  `level/*` composition vs `env_steps` from wandb; rows with a NaN are dropped.
- `curves.pdf`: eval success vs env steps, robot x d panels, seed mean with
  min-max band.
- `composition.pdf`: mean physical value of the magnitude parameters and
  genome distance among training levels, Wojtek units, running mean over 9 points.
- `terrains.pdf` / `robots.pdf` (`paper/terrains.py`): offscreen MuJoCo
  renders of one level per feature at full strength plus uniform samples, and
  both robots on one shared genome. Needs the built assets (`just build`,
  `just build-accel-arena`); the recipe sets `MUJOCO_GL` as `render-video`
  does, and `arena` (default `benchmark`) is the arena the levels render on.
- Overleaf: upload the PDFs into a `figs/` folder; `\resultfig` falls back to a
  placeholder while a file is missing.
- `results.csv`: one row per run, every value the tables use.

## Domination comparison — `scripts/compare_domination.py` {#domination}

Code: `src/evaluation/domination.py` (loading, pairing check, solve rates,
histogram, stats, matrix), `src/paper/domination.py` (figures). Design by
Piotr Wojdas (pairing check, binomial
noise reference, bootstrap CI, single-hue colour ramp).

```sh
just domination eval_dir=<evals> method_a=accel method_b=dr
```

- Why: success rate and CVaR are marginals; two methods can match on both
  and still differ per level (one better everywhere, or each winning on
  disjoint levels). The comparison pairs them level by level.
- Definition: per level θ of the set, `solve_rate(θ)` = fraction of
  episodes with no fall and travel >= `min_travel` x robot scale (eval_brax's
  success rule) over both passes (2 x 4 = 8 episodes), then the mean over
  the method's seeds (5 seeds: values k/40).
  H(x, y) = number of levels with A's rate in bin x and B's in bin y.
- Inputs: `eval_dir` of eval_brax npz dumps
  `<prefix>_<method>_seed<k>_final_eval.npz`, `prefixes.<robot>` as in
  `paper_tables`; methods and seeds come from the file names.
  `level_set` picks the per-level set: `param_uniform` (default, needs
  `level_space=param` evals) or `random_d<d>`; a dump without it fails
  naming the sets it has.
- Pairing: every dump of a robot must hold the same level genomes
  (`param_uniform_levels`, `random_levels` for `random_d*`); otherwise it
  refuses (rerun eval_brax with the same seed / `num_levels` /
  `level_space`).
- Gaps are reported, not dropped: a method lacking seeds another method
  of the robot has, a method other robots have, and a robot without both
  `method_a` and `method_b` (its heatmap is skipped) warn and go to the
  JSON (`missing_seeds`, `missing_methods`, `pair: null`). No robot with
  the pair is an error.
- Writes to `out` (default `artifacts/paper/domination`):
  - `domination.json`: per robot `num_levels`, per method seeds,
    `episodes_per_level` (all seeds), `mean_solve_rate`; the pair's
    `a_better` / `b_better` / `ties` (fractions of levels, ties = equal
    rate), `mean_diff` (A - B) with `mean_diff_ci` (95 % bootstrap over
    levels), `mean_abs_diff`, `noise_abs_diff`, `counts` (bins x bins level
    counts, row = A's bin, column = B's bin); `matrix`.
  - `domination_<a>_vs_<b>.pdf`: one heatmap per robot in a row, plus
    `domination_<a>_vs_<b>_<robot>.{pdf,png}`.
  - `domination_matrix.pdf`: per robot, methods x methods, cell = fraction
    of levels where the row method's rate beats the column's
    (`row_better`; the diagonal is 0, `M[a,b] + M[b,a] + ties = 1`).
- Reading the heatmap: x = A's solve rate, y = B's, `bins` (10) bins of
  10 %, the last one closed at 100 %; cells show the fraction of levels
  (sum 1) on one log scale for every robot, from a single level of the
  largest set to all levels, empty cells white. Mass
  below the diagonal = levels A solves more often (A dominates), above = B;
  the bottom-left / top-right corners are levels both fail / both solve.
  Each panel's title has the `a_better` / `b_better` / tie fractions,
  `mean_diff` with its CI and `mean_abs_diff` against the noise reference.
- Noise reference (`noise_abs_diff`): the mean |A - B| two identical
  policies would show from episode sampling alone. Per level, both rates
  are drawn as Binomial(n, p) / n at p = (A + B) / 2 with each method's
  episode count n (seeds x passes x episodes); the exact expectation,
  averaged over levels. At p = 0.5 it is 0.20 for 8 episodes, 0.09 for
  40 (5 seeds). `mean_abs_diff` near it means the per-level
  spread is sampling noise; it ignores seed-to-seed policy variance, so it
  is a lower bound on the spread of two equally good methods.
