# Evaluation

Final eval, sim2sim and demo videos of a trained checkpoint.

## Final eval

Entry point `scripts/eval_brax.py`; code in `src/evaluation/final_eval.py`
(sets, metrics), `suite.py`, `stats.py`, `rollout.py`.

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

## Sim2sim

Entry point `scripts/eval_sim2sim.py`; code in `src/evaluation/sim2sim.py`
(sets, metrics), `cpu_physics.py`, `physics_variants.py`.

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
- SLURM: `slurm/eval_sim2sim.sbatch`, same task layout and variables
  (`MODES`, `NSEEDS`, `SEED0`, `PREFIX`, `ROBOT`, `SRC`) as
  `eval_brax.sbatch`; CPU partition, 16 cores (the CPU nodes must run the
  OS the venv was built on).

## Demo videos

Entry point `scripts/render_video.py`; code in `src/rendering/video.py`,
`scene.py`.

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

