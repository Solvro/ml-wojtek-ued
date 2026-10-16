# Environment

Covers `src/wojtek_env/` (the extracted env library) and `src/utils/env.py`
(the training-side bridge). The library is **ruff-excluded and hatchling-
packaged** — treat it as a stable dependency; research code consumes it, does
not edit it.

## Upstream sync {#upstream-sync}

- **Synced from w01-tek `e44c1a9`** (`training/wojtek_rl/`, 2026-10-15).
  Previous copy: extracted 2026-08-11.
- Upstream modules are copied as-is into the `envs/`, `terrain/`, `tools/`
  layout; only `wojtek_rl` imports and module-path strings are rewritten.
  `paths.py`, `gymnasium/`, `tools/check_gym.py` and the `__init__.py` files
  are ours.
- Every local delta vs `e44c1a9` (re-apply on the next sync). None is
  inference-relevant: obs layout, action pipeline, PD/ctrl config and network
  inputs are upstream's.

| file | delta | why | scope |
|---|---|---|---|
| every ported module | `wojtek_rl` imports / module-path strings → `wojtek_env.{envs,terrain,tools}` | package layout | neither |
| `envs/env.py` | spawn pose: every leg joint, passive four-bar joints included, from the settled closed-loop pose (`_spawn_joint_qpos`, `_dsecond`, see below); `_settle_height_grid` also returns the joints and runs at every construction (~0.7 s CPU), not only under `real_pose_ref`; `_height_ctrl` / `_pose_ref` stay verbatim | upstream leaves the closure open at reset | training-only (reset state) |
| `envs/env.py` | `_get_reward` also returns `fall_causes`; `step` writes `fall_{min_height,tilt,toggle,base_contact}_at_done` | read by `utils/ppo_helpers.py` | training-only (metrics) |
| `paths.py` | ours: assets under `assets/wojtek_description/`, `"accel"` and `"accel_deploy"` in `TERRAIN_KINDS`, no room/HF/policy constants | repo layout; accel arenas for UED | training-only |
| `envs/__init__.py`, `gymnasium/`, `tools/check_gym.py` | ours: re-exports, Gymnasium wrapper | convenience API | training-only |
- Generated `wojtek_mjx.xml` / `scene_mjx.xml` are byte-identical to upstream
  `ros/src/wojtek_description/mujoco/` at `e44c1a9`.
- Model-free upstream unit tests live in `tests/wojtek_env/` (mirrors the
  package layout); tests that build an MJX env were not ported.

### What the e44c1a9 sync changed for existing experiments

- **No default changed.** Every new feature ships off (`0.0` / `enable=False`).
  At default config and with `movement_overrides()`, obs / reward / qpos match
  the previous copy bit-for-bit over 20 jax-backend CPU steps.
- New `info` keys on every joystick state: `gyro_bias`, `gyro_vib`,
  `gyro_vib_gain`, `last_base_vz`, `last_gyro_xy`, `last_qvel_probe`.
- New `reward/*` metrics (scale 0 by default): `energy_cmd`, `slip_cmd`,
  `probe_sat`, `probe_slip`, `probe_transient`, `probe_free_transient`,
  `base_accel`, `tau_ff_swing`, `tau_ff`, `target_sag`. `scripts/brax_ppo.py`
  logs every env metric, so its wandb runs gain these columns.
- `biped` task registered (`env_biped.py`; closure is `base` + `env_getup`).
- `FunctionalEnv` auto-reset only restores `data`/`obs`, not `info`; the
  per-episode `gyro_bias` draw therefore stays fixed per env slot there (same
  as every other `info` field).

### New opt-in knobs (joystick, `envs/env.py` / `envs/base.py`)

- `obs_noise.gyro_bias` — per-episode uniform gyro bias, actor obs only.
- `obs_noise.gyro_vib` / `gyro_vib_decay` — gyro resonator driven by the
  torque change (`base.gyro_vib_step`, rings at half the control rate);
  actor obs only.
- `tau_ff.enable` / `scale` / `rate_weight` — feed-forward torque head:
  `action_size` becomes 24, the second 12 go to `qfrc_applied` with the same
  latency as the position targets.
- `obs.history_len` — `proprio_history` obs component (RMA-style history).
- Obs components `foot_friction`, `env_factors` (friction, gain / kd / force
  scale, base mass and CoM offset vs the nominal model) — privileged inputs
  for RMA; they read the (possibly DR-swapped) `_mjx_model`.
- Reward: `base_accel` (+ `base_accel_ang_weight`, `base_accel_wz_fade`/
  `_floor`), `energy_cmd`, `slip_cmd`, `target_sag` (+ `target_sag_wz_fade`),
  `tau_ff_swing`, `tau_ff`, `probe_*` diagnostics,
  `feet_landing_wz_fade`/`_floor`, `terrain_gate.flat_pitch_spin_exempt`.

## Tasks and the registry

`wojtek_env/envs/registry.py` maps task name → (env class, default config):

- `joystick` — velocity-command locomotion (`WojtekJoystick`). Default task.
- `getup` — recover to standing (`WojtekGetup`).
- `jump` — jump task (`WojtekJump`).
- `biped` — rear up onto the hind legs and hold (`WojtekBiped`); from
  upstream, unused by any script here.

```python
from wojtek_env.envs import registry
env = registry.make_env("joystick", env_overrides={...})
```

Configs are `ml_collections` ConfigDicts built in code. `env_overrides` is a
plain nested dict (as Hydra yaml carries) merged onto the default with
tuple/float coercion — e.g. yaml `vx: [-0.8, 1.8]` lands on a tuple field, and
a scalar default like `action_scale` can be overridden with a per-element
vector.

## Task config knobs off by default {#task-config-knobs-off-by-default}

`WojtekJoystick.default_config()` (`envs/env.py`) defines several
locomotion-quality and sim2real-robustness features that ship
`enable=False`/`scale=0.0`; the config value itself already documents each
one inline. `utils/env_overrides.py`'s `movement_overrides()`
([ppo.md](ppo.md#shared-env-overrides--utilsenv_overridespy)) turns on the
task-independent ones for `scripts/ppo_dr.py`/`accel.py`/`accel_top_k.py`:

- `latency`, `encoder` — substep control latency and joint-encoder
  miscalibration randomization (sim2real robustness; `action_delay=1` is a
  separate always-on fixed delay).
- `symmetry` — left/right mirror augmentation.
- `real_pose_ref` — judge pose/height against the kinematic standing family
  instead of commanded ctrl targets (removes a PD-sag-dependent penalty
  floor; sim2real correctness fix, not just a shaping knob).
- Spawn pose (`WojtekJoystick.reset`, local patch on top of upstream): every
  leg joint, passive four-bar joints included, comes from the settled
  closed-loop pose for the commanded height (`_spawn_joint_qpos`, tabled per
  `DSECOND_TABLE` rung by `_settle_height_grid`). Previously only the
  actuated joints were set, leaving the closure loop open by up to ~0.13 m;
  the solver snapped it shut at 40–70° toggle and the policy hit
  `fall.max_toggle_deg=150` within ~0.35 s. Rest-pose toggle is now ≤ ~26°.
- `movement_overrides` no longer sets `fall.max_toggle_deg` (termination
  off; the `toggle_flat` penalty stays). Even with the fixed spawn and an
  accurate solver, random actions push a leg past 150° in ~35% of 2 s
  episodes, and brax PPO (mujoco_playground recipe) still lost 92% of its
  episodes to the toggle at ~3 s while walking at ~62% of the command.
  At the default `opt.iterations=2` the four-bar closure also stretches
  ~14 mm under random actions (2.3 mm at 10 iterations), which doubles the
  toggle events.
- `action_filter` — EMA smoothing on actions before the PD targets.
- `command.pure_wz_prob`/`pure_vy_prob`/`arc_prob`/`pure_slow_prob`/
  `pure_fast_prob`/`pure_back_prob` — dedicated clean-command training
  modes; the uniform command box under-samples these (spin-in-place,
  strafe, arc-following, slow/fast/backward walking), and the inline
  comments cite measured deadlocks from training without them.
- `no_progress` — probabilistic episode cut when the robot ignores the
  commanded velocity, instead of letting a stalled episode run out the
  clock.
- `gait.air_time_cap`, `fall.max_toggle_deg` — anti-runaway swing-time cap
  and four-bar toggle-angle termination.
- `reward.tracking_relative`/`tracking_product`/`shaping_tracking_gate`/
  `tracking_far_weight`/`orientation_tol_deg` and
  `reward.scales.torque_rate`/`feet_landing`/`stand_feet_down`/
  `torque_limit`/`toggle_flat` — see the inline comments in `env.py` for
  what exploit or dead-gradient region each one closes.

## Terrain-only knobs {#terrain-only-knobs}

These read zero / never fire on the flat scene (`terrain.enable=False`), so
they only make sense once terrain is on. `terrain_overrides()` turns them on
for `scripts/accel.py`/`accel_top_k.py`; `scripts/ppo_dr.py` trains flat and
skips them:

- `height_scan` — body-frame grid of terrain heights ahead of the robot;
  **privileged-only** (`"height_scan_clean"` in `obs.privileged`, i.e. the
  critic) — wojtek has no lidar/depth sensor, so the actor (`obs.state`)
  never gets a `height_scan` reading, corrupted or not. This is the
  asymmetric actor-critic split (see `FunctionalEnv` below): the critic's
  terrain knowledge shapes the advantage/regret signal during training, but
  the deployed actor stays proprioceptive-only, matching real hardware.
  `env.py` raises if `obs.privileged` names `"height_scan_clean"` without
  `height_scan.enable`; `terrain_overrides()` sets both together.
- `reward.terrain_gate`, `reward.scales.flat_pitch` — scale gait shaping and
  a nose-down pitch penalty by local terrain roughness.
- `command.terrain_bias` — bias the resampled command toward steady forward
  walking; the base command distribution is tuned for the flat scene, where
  standing/spinning happens on the spawn's local ground and never meets a
  terrain feature.
- `fall.on_base_contact` — analytic base-on-terrain contact ends the
  episode; without it a robot lying on rough ground can dodge the flat
  `min_height`/tilt fall check.

## Terrain generator

`wojtek_env/terrain/terrain.py` builds one deterministic arena from a seed: a
grid of ~3 m tiles, rows of increasing difficulty, one tile of every terrain
type per row, flat border. Rough/slopes/pit are carved into a single
heightfield; stairs and obstacles are static box geoms on top.

Key facts that matter downstream:

- MuJoCo normalises heightfield data to `[0,1]` on compile; the physical
  surface is `pos_z + data * elevation_z`. The generator stores data normalised
  to its own range and sets `elevation_z`/`pos_z` to reproduce exact heights.
- A dense **lookup grid** stores ground-truth surface height on the same node
  grid as the heightfield, for bilinear terrain-relative height/scandot reads
  (no raycasts — they don't exist on the JAX backend).

Companion modules: `terrain_env.py`, `terrain_wrapper.py` (curriculum wrapper),
`terrain_suite.py` (fixed measurement course). ACCEL does **not** use this
tile-type parametrization for its levels — see
[ued-accel.md](ued-accel.md#level-representation).

## Gymnasium wrapper

`wojtek_env/gymnasium/env.py` — `WojtekGymEnv(gym.Env)` wraps a task env in
brax's `EpisodeWrapper` with jitted reset/step for a standard Gymnasium loop:

```python
from wojtek_env.gymnasium import WojtekGymEnv
env = WojtekGymEnv(task="joystick")
obs, info = env.reset(seed=0)
```

`main.py` is a random-agent MuJoCo-viewer preview built on this.

## Domain randomization

`wojtek_env/envs/randomize.py` `make_domain_randomize(mj_model, dr_cfg=None)`
returns a `domain_randomize(model, rng)` closure, the brax
`randomization_fn` convention: given a batch of rngs it returns a
`(batched_model, in_axes)` pair — `model.tree_replace`d with the randomized
fields stacked on a leading axis, `in_axes` marking which fields are batched
(`0`) vs shared (`None`), ready for `jax.vmap(fn, in_axes=[in_axes, 0])`.

`dr_cfg=None` (every field's `enable: False`) randomizes five fields
unconditionally — floor/terrain friction, base and per-link mass, one
gain/kd actuator scale — the library's baseline domain randomization. Passing
a `dr_cfg` dict (shape mirrors `randomize._DEFAULT_DR`) turns on extra,
independently-gated fields: `com_offset`, `joint_gains` (per-actuator instead
of one shared scale), `dof` (damping/armature/frictionloss), `foot_friction`,
`motor_strength`. `FunctionalEnv(..., dr_config=...)` passes one through;
`brax_ued.py --profile deploy` uses upstream `terrain_blind_v4_4d`'s
([deploy.md](deploy.md#profile)). `scripts/ppo_dr.py` (see [ppo.md](ppo.md#domain-randomization-scriptsppo_drpy))
and the benchmark profile use the baseline five-field form via
`FunctionalEnv(domain_randomize=True)`.

`terrain/terrain_wrapper.py`'s `wrap_for_terrain_brax_training` also accepts a
`randomization_fn` and applies it through mujoco_playground's
`BraxDomainRandomizationVmapWrapper`, but no script currently wires
`randomize.py` through that path — `FunctionalEnv` uses its own swap
mechanism instead (below), consistent with the hfield swap it already does.

## FunctionalEnv — the training bridge

`utils/env.py` `FunctionalEnv` is what the PPO/ACCEL trainers actually use. It
wraps `registry.make_env` + brax `EpisodeWrapper` and exposes pure `reset`/`step`
usable under `jit`/`vmap`.

- `go1_hfield=HFieldConfig(...)` builds Playground's Go1 on that arena
  (`robots/go1.py`) instead of Wojtek; the arena and DR work the same, but
  Go1 reads terrain heights straight from `hfield_data` (no lookup) and
  spawns on `Go1Terrain.spawn_xy`.

- **Split obs.** `observation_size` / every `Obs` carries `actor` (`state`) and
  `critic` (`privileged_state`) separately — asymmetric actor-critic.
- **Auto-reset semantics.** `step` freezes `data`/`obs` to the episode's first
  values on `done` (via `first_data`/`first_obs` stashed at reset), so a
  vmapped batch stays aligned.
- **Shared tiled arena.** `reset(key, hfield_data, dr_model, tile_xy)` and
  `step(..., hfield_data, dr_model)` take the arena's `hfield_data`
  (`hfield.tiles.TileLayout.assemble`: every level of the iteration on its
  own tile) — **one array for every env**, vmapped with `in_axes=None`.
  `tile_xy` is the env's tile centre, where it spawns; `reset` raises if an
  arena comes without it. `_swapped_model` temporarily does
  `base._mjx_model = model.tree_replace({"hfield_data": hfield_data})` around
  the call, then restores it, so a new iteration's levels need no MJX
  recompilation (see below). Details: [tiled arena](#tiled-arena).
- **Domain randomization swap.** `reset`/`step` also take an optional
  `dr_model` — a full `mjx.Model` (from `FunctionalEnv.sample_dr`, above) that
  `_swapped_model` installs as `_mjx_model` outright instead of
  `tree_replace`-ing one field. The two compose: passing both swaps
  `dr_model` in first, then `tree_replace`s `hfield_data` on top
  (`brax_ued.py`, `ppo_dr_terrain.py`).

## Why the arena swap is free {#hfield-mutability}

`mjx.Model` registers a field as a dynamic pytree leaf iff it is a `jax.Array`.

- `hfield_data` → `jax.Array` → real dynamic leaf → swappable at runtime with
  no MJX recompilation.
- `hfield_size` (radii, elevation, base) and `hfield_nrow`/`ncol` → plain
  `numpy.ndarray` / int → static, baked in at compile time; changing them
  needs a new model.

Consequence: fix the arena geometry once (`HFieldConfig`: level grid,
elevation envelope, tile count, gap), and the heightfield *values* become a
cheap, JAX-native "level genome" — which is what makes ACCEL practical at RL
scale.

Two things must follow the swap, or the level only exists on paper:

- **Terrain lookup** — height scan, base height, foot contact/clearance and
  the base-contact/min-height falls read `Arena.lookup`, not `hfield_data`.
  The accel arena's static lookup is flat, so `FunctionalEnv._swapped_model`
  also swaps `base._terrain.lookup` for `base_z + hfield_data * elevation`
  (`HFieldGeometry`, `utils/env.py`) and points every spawn-table entry
  (`origin_xy`) at the env's `tile_xy`. Requires an arena whose lookup is the
  single heightfield's grid (the `accel` arenas); otherwise it fails fast.
- **Warp physics** — MJWarp's collision kernel indexes `hfield_data` without
  a world id, and `mjx` flattens a vmapped `(B, N)` `hfield_data` to
  `(B*N,)`, so a per-env (batched) heightfield on `sim.backend=warp` makes
  every env collide with the **first** env's level. The tiled arena avoids
  it by construction: `hfield_data` is unbatched, so warp is valid and
  `accel_arena_overrides` leaves `sim.backend` at `auto` (warp on a GPU, jax
  on CPU) with upstream's terrain buffers (`naconmax_per_env` 116, `njmax`
  1024). Never vmap `hfield_data` with `in_axes=0`: nothing can detect it
  inside the vmap. `src/scripts/check_hfield_batch.py` checks it per
  backend (flat/pit tiles; `--compare` = jax vs warp; `--bench_envs N`
  times it).
- **Warp memory** — warp allocates its collision scratch (EPA buffers,
  ~0.3 GB each at 8192 envs) outside JAX's pool. Upstream measured a
  mid-run OOM with JAX preallocating and calls turning it off mandatory on
  terrain; `FunctionalEnv` refuses warp unless
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and the justfile, `slurm.sh` and
  `slurm/*.sbatch` export it. Upstream's presets spell it
  `XLA_PYTHON_CLIENT_PREALLOC=false`, which is not a variable JAX reads.
- **Warp CUDA graphs** — mjx's warp FFI (`GraphMode.WARP`) captures one
  CUDA graph per set of buffer addresses, each owning its own collision
  scratch, and caches 32 per callable. Upstream's single jitted `ppo.train`
  step reuses its buffers; brax_ued's separate jitted reset / unroll / eval
  calls pass new ones every call, and the first two sanity runs died with
  a CUDA OOM at the third iteration, with JAX preallocating (job 6109872)
  and without (6110159, 15 of 80 GB in use). `FunctionalEnv` caps the
  cache at `WARP_GRAPH_CACHE` = 4 (`configure_warp`) when it builds a warp
  env; with both settings the run completed (job 6110213, ~16 GB).

## Tiled arena {#tiled-arena}

`hfield/tiles.py` `TileLayout(HFieldConfig)`; the `accel` / `accel_deploy`
assets are built with it (`hfield/arena.py`).

- `tiles x tiles` square cells (`HFieldConfig.tiles`, default 16 = 256
  tiles), row-major from `(-half, -half)`. Tile i holds level i, its
  `grid_size x grid_size` patch centred in the cell; spawns at the centre,
  on the level's flat pad (`pad_radius`, ± `pad_jitter` 0.15 m).
- Between neighbouring patches: a flat `gap` (the neutral plane, height 0),
  `reach - patch_radius` rounded up to an even number of cells, at least 2.
  `reach` (default 10 m) bounds how far from its tile centre a robot reads
  terrain within one level, so it cannot read a neighbour's level. Budget
  (`utils.env.TravelBounds`): spawn jitter + `steps x ctrl_dt x` max command
  speed + sensing (feet 0.35 m, or the height scan's far corner 1.48 m when
  any obs reads it; the critic does). Deploy, 300 steps: 0.15 + 6 x 1.30
  (vx 1.2, vy 0.5) + 1.48 = 9.43 m; benchmark (fast walk 1.2 m/s): 8.93 m.
  Benchmark: gap 7 m, 4161² nodes (208 m, 69 MB); deploy: gap 4 m, 5121²
  nodes (256 m, 105 MB); Go1: `reach` x 2.5, gap 17.5 m.
- `TileLayout.check_reach`: every script that puts several levels on the
  arena (`brax_ued`, `eval_brax`, `eval_sim2sim`, `eval_levels`,
  `accel_top_k`, `ppo_dr_terrain`) fails at startup when its steps per
  level exceed the budget; training counts the critic's scan, eval only the
  deterministic policy's obs and feet. 1000-step levels need ~28 m: not
  supported (`eval_sim` runs one level, neighbours flat).
- `TileLayout.leaked(xy, tile_xy, sensing)`: base past `patch_radius + gap
  - sensing` (Chebyshev) from the own centre = reads another tile.
  `brax_ued.py` logs `tile_leak_frac` (envs that got there during the
  iteration, with the critic's sensing).
- vs the old per-level arena: inside the patch identical; past the patch
  edge the robot meets the flat gap where it met the flat apron before. The
  lookup there now reads that flat ground (the old lookup clamped to the
  patch edge's heights, which the physics did not have).
- One model update per iteration: `assemble` (jitted, on device) writes the
  iteration's rendered levels into the arena; unused tiles stay flat.
- Float32 positions differ per tile, which perturbs a standing robot
  slightly (chaotic spawn settle): ~0.5 mm in the 2x2-tile test arena
  (`tests/utils/test_env.py`), far below a level's features.
- CPU rendering (`render_video.py`, `paper_terrains.py`, `eval_sim.py`)
  uses `hfield.arena.single_tile_spec`: the scene cut to one tile with the
  level at the origin; the renderer meshes the whole hfield, which for the
  full arena is millions of cells.

## Assets {#assets}

`wojtek_env/paths.py` centralises repo-relative paths. Generated arena file sets
(`terrain_paths(kind)`) are `scene` XML, `hfield` binary, `spec` JSON, `lookup`
NPZ, one set per kind in `TERRAIN_KINDS = (train, eval, eval_deep, test, accel,
accel_deploy)`; `tests/utils/test_env.py` writes a small tiled arena as `test`.

They **must** live next to the robot XML: the scene includes the robot by
relative path and MuJoCo resolves `<hfield file=>` against the robot's mesh
directory — an arena written elsewhere won't compile. All are gitignored and
built by the tools below.

## Build / check tools

`wojtek_env/tools/` (run via `python -m wojtek_env.tools.<name>`):

- `build_model` — `wojtek.xml` → `wojtek_mjx.xml` + `scene_mjx.xml` (`just build`).
- `build_terrain` — write a procedural terrain arena's asset set.
- `check_gym`, `check_model_mjx`, `check_terrain` — sanity-check CLIs.

The ACCEL arena is written by `hfield/arena.py` instead
(`just build-accel-arena`), documented in [ued-accel.md](ued-accel.md#arena).
