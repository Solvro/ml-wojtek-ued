# Training

The brax-PPO student, the 14 curricula (`conf/mode/`), PAIRED, and the
measured throughput.

## Trainer

Entry point `scripts/brax_ued.py`; code in `src/training/` (`student.py`,
`iteration.py` = one outer iteration, `curricula.py` = the modes below,
`train.py` = the loop and logging).

`just train mode=<option> name=<run> [overrides]` (`conf/train.yaml`;
every field: [configuration.md](configuration.md)).

`brax_ued.py` uses brax's PPO pieces with the mujoco_playground recipe
(networks, running obs normalizer, `compute_ppo_loss`, GAE, 20-step unrolls,
4 epochs x 32 minibatches, discount 0.97; the `ppo` and `network` config
fields, defaults = this recipe) and drives `FunctionalEnv` itself,
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
  CPU; the Playground robots always jax, [robots](robots.md)). Compare runs
  within one backend
  ([Throughput](#throughput)).
- `mode=dr`: fresh uniform levels at full difficulty, train every
  iteration.
- `mode=dr_easy`: DR over ACCEL's easy generator (control points within
  `generator_scale` of flat) — isolates the curriculum from "train easy".
- `mode=dr_mixed`: uniform levels scaled by a per-level difficulty
  drawn from `[0, 1]`.
- `mode=terrain_cl`: the game-inspired terrain curriculum of Rudin et al.
  2022 (legged_gym / Isaac Lab `terrain_levels`), per lane because all
  `envs_per_lane` envs of a lane share its tile (`training/terrain_cl.py`).
  - Column: each lane draws one uniform level (own noise seed) at the start
    of the run and keeps it.
  - Row `r` in `0..10`: the lane runs its column shrunk toward `flat` by
    `d = r / 10` (`LevelSpace.shrink`, the `random/d<d>` formula). Initial
    rows uniform in `0..5` (legged_gym `max_init_terrain_level`).
  - Every iteration trains (as DR); afterwards each lane moves: up one row
    when >= 50 % of its envs succeeded (no fall, >= `min_travel` m, as
    `train_success`), else down one row when its mean travel < 0.5 x its
    mean commanded distance; clipped at row 0. A lane promoted past row 10
    goes to a uniformly random row.
  - Commanded distance (`iteration.commanded_motion`): the displacement of
    a robot tracking the iteration's commands exactly, body-frame
    `(vx, vy)` integrated along the heading the `wz` commands turn, over
    every step (resets included). A constant straight command gives
    legged_gym's `|command_xy| x episode_length_s`. Travel is a
    displacement, so a path length (`sum |command_xy| x dt`) would demote
    perfect trackers: with the redrawn turning commands a perfect
    tracker's mean displacement is ~0.43 of it on Spot, ~0.56 on Go1/Go2.
  - Logged: `terrain_cl/row_mean`, `/row_p10`, `/row_p90` (rows the
    iteration ran), `/promote_frac`, `/demote_frac`, `/top_reset_frac`,
    plus the level composition.
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
  - `mode.score=lp` — learning progress, signed (`training/progress.py`),
    after LP-ACRL (Li et al. 2026, arXiv 2601.17428:
    `LP = R_c(ζ) - R_c-1(ζ)`, the change in expected episodic reward
    between consecutive stages), adapted to ACCEL's buffer:
    - `R` per visit of a level = mean over its `envs_per_lane` envs of the
      undiscounted return (full task reward) of the env's first episode on
      the level: reward summed from the level's first step through its
      first termination, else over all `H` = resolved `level_steps` steps
      (Spot 780, Wojtek 300). `lane_episode_return`, `LaneTotals` in
      `training/iteration.py`.
    - A terminated env (fall, divergence, Wojtek's `no_progress` cut)
      auto-resets onto its spawn on the same tile and keeps stepping for
      the rest of the level; those steps still train PPO and still count
      in `lane_returns` (PAIRED regret), but not in `R`. `H` stays below
      the env's 1000-step episode length, so no truncation inside a level.
    - Score = `R(this visit) - R(previous visit of the level)`: negative
      on forgetting. A level's first visit scores 0; LP is defined from
      its second visit on. A mutated child is a new level, so its first
      visit also scores 0: it does not inherit its parent's `R`, since a
      child-vs-parent difference mixes the change in difficulty the
      mutation made with the student's learning progress.
    - The buffer stores each level's last `R` (every buffered mode tracks
      it; only `lp` scores by it). Sampling, replay, mutation and the
      generator are plain ACCEL's. Negative scores rank last and are
      evicted first; at the start the buffer fills with 0-score fresh
      levels (eviction only past `buffer_size`) and replays them.
  - `mode.generator=mixed` — explore iterations draw fresh levels from the
    `dr_mixed` distribution instead of the easy generator.
  - Logged: `buffer_roughness_mean`/`_p90`, `buffer_score_roughness_spearman`
    (rank correlation of level score vs roughness), `children_roughness`,
    `train_success`, `learnability_mean`.
  - Mode options `mode=accel_grow` (grow), `mode=accel_sfl` (grow +
    learnability) and `mode=accel_lp` (plain ACCEL + `lp`).
    `level_space=grid` composes 16 edits for the two `grow` modes (8
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
  see [Playground robots](robots.md).
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
  [README](slurm.md#job-variables); assets must be built
  beforehand since the array's tasks start together.
- The arena must match the run's `arena` config (`hfield.arena.check_arena`:
  `pad_radius`, height scale, level grid, tiles, gap); the error names the
  `just build-accel-arena arena...` command for it. Training and the
  builder read the same `arena` group, so the default needs no override.

## PAIRED

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

## Throughput

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
  students: ~1 GPU-h).
- Playground robots run on jax: Go1 ~410k env steps/s on flat levels and
  ~335k in `mode=accel` on one H100, so a 1B-step run takes ~1 GPU-h.
- Whole-device GPU memory on warp: ~16 GB (JAX preallocation off).
- Warp is not jax: one checkpoint scores 1.5-3 points higher success on
  terrain under warp (flat unchanged) and stands 7-11 mm higher on every
  tile. Train, eval and compare within one backend.

