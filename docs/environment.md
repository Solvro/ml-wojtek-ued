# Environment

Covers `src/wojtek_env/` (the extracted env library) and `src/utils/env.py`
(the training-side bridge). The library is **ruff-excluded and hatchling-
packaged** — treat it as a stable dependency; research code consumes it, does
not edit it.

## Tasks and the registry

`wojtek_env/envs/registry.py` maps task name → (env class, default config):

- `joystick` — velocity-command locomotion (`WojtekJoystick`). Default task.
- `getup` — recover to standing (`WojtekGetup`).
- `jump` — jump task (`WojtekJump`).

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
([below](#env-overrides)) turns on the task-independent ones:

- `latency`, `encoder` — substep control latency and joint-encoder
  miscalibration randomization (sim2real robustness; `action_delay=1` is a
  separate always-on fixed delay).
- `symmetry` — left/right mirror augmentation.
- `real_pose_ref` — judge pose/height against the kinematic standing family
  instead of commanded ctrl targets (removes a PD-sag-dependent penalty
  floor; sim2real correctness fix, not just a shaping knob).
- Spawn pose (`WojtekJoystick.reset`, local edit to the extracted env): every
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
for the tiled-arena scripts:

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
`motor_strength`. `brax_ued.py` exposes no `dr_cfg`
overrides; it uses the baseline five-field form via
`FunctionalEnv(domain_randomize=True)`, which draws one randomized
`mjx.Model` per parallel env (`FunctionalEnv.sample_dr`) and keeps the
`jax.vmap` axis spec for it.

`terrain/terrain_wrapper.py`'s `wrap_for_terrain_brax_training` also accepts a
`randomization_fn` and applies it through mujoco_playground's
`BraxDomainRandomizationVmapWrapper`, but no script currently wires
`randomize.py` through that path — `FunctionalEnv` uses its own swap
mechanism instead (below), consistent with the hfield swap it already does.

## FunctionalEnv — the training bridge

`utils/env.py` `FunctionalEnv` is what `brax_ued.py` and the eval scripts use. It
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
- **Divergence guard.** A step whose `qpos`, `qvel`, obs or reward is not
  finite, or with any |`qvel`| above `QVEL_DIVERGED` (200 m/s or rad/s;
  random actions on full-difficulty levels peak at ~75), ends the episode
  (`done` 1, reward and per-env metrics 0, then the auto-reset above) and
  sets `info["diverged"]` (brax_ued logs `diverged_per_step`; in eval a
  diverged env scores a fall). MJX blows up now and then on one env; its
  obs would reach brax's running obs normalizer and never wash out (seen:
  joint-velocity stds 100-640000x, then policy collapse; a nan one turns
  every env nan). `info` is not reset, so an env whose `info` went nan
  stays ended until the next `reset`.
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
  (`brax_ued.py`).

## Shared env overrides — `utils/env_overrides.py` {#env-overrides}

`registry.py`'s task defaults leave most locomotion-quality and sim2real
robustness features off (`enable=False`/`scale=0.0`) — see
[task config knobs](#task-config-knobs-off-by-default) for the
full inventory. `brax_ued.py` (and through it `eval_brax.py`, `eval_sim2sim.py`) and
`check_hfield_batch.py` turn the same set on rather than repeating one large
dict:

- `merge_overrides(*dicts)` — deep-merges nested override dicts (a plain
  `{**a, **b}` would clobber a whole nested key, e.g. `command`, if two
  sources both touch it).
- `movement_overrides()` — task-independent: `latency`/`encoder` noise,
  `symmetry` mirror augmentation, `real_pose_ref`, `action_filter`, the
  `command.pure_*_prob` clean-command modes, `no_progress`,
  `gait.air_time_cap`, `fall.max_toggle_deg`, and the `reward` scale/gate
  fields that were `0.0`/`False` in `registry.py`. Used by both
  scripts.
- `terrain_overrides()` — terrain-only (`height_scan`, `reward.terrain_gate`,
  `reward.scales.flat_pitch`, `command.terrain_bias`, `fall.on_base_contact`,
  and the `obs.state`/`obs.privileged` lists extended with
  `height_scan`/`height_scan_clean`). Only for runs on the tiled arena; on a
  flat scene `terrain_gate`/`height_scan` read zero anyway (see
  [terrain-only knobs](#terrain-only-knobs)).
- Enabling `height_scan` changes `actor_obs_dim`/`critic_obs_dim` (+25 each,
  `NX*NY`); this needs no wiring because `FunctionalEnv.observation_size` reads the size
  off a live env probe, not a hardcoded constant. `symmetry` + `height_scan` compose
  correctly out of the box — `wojtek_env.envs.symmetry.obs_mirror` already
  has a registered mirror map for the `height_scan`/`height_scan_clean`
  catalog entries.
- Most of the `movement_overrides()`/`terrain_overrides()` magnitudes
  (`action_filter`, `tracking_far_weight`, `orientation_tol_deg`, the new
  `reward.scales` values, the `command.pure_*_prob` values,
  `fall.max_toggle_deg`, `gait.air_time_cap`) are **untuned estimates**
  reasoned by analogy to sibling terms already set in `registry.py` — grep
  turns up no prior tuning history for these fields in this repo. Treat them
  as a starting point, not a validated config.

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
  `accel_arena_overrides` takes the backend (`brax_ued.py backend=`,
  default `auto`: warp on a GPU, jax on CPU) with a terrain-sized contact
  pool (`naconmax_per_env` 116, over the 84-contact floor of 21
  ground-colliding geoms; `njmax` 1024). Go1 follows the same flag
  (`Go1Terrain(backend=...)`) with Playground's rough-terrain budget: 8
  contacts per env, `njmax` 60. Never vmap `hfield_data` with `in_axes=0`: nothing can detect it
  inside the vmap. `src/scripts/check_hfield_batch.py` checks it per
  backend (flat/pit tiles; `compare=true` = jax vs warp; `bench_envs=N`
  times it).
- **Warp memory** — warp allocates its collision scratch (EPA buffers,
  ~0.3 GB each at 8192 envs) outside JAX's pool. Upstream measured a
  mid-run OOM with JAX preallocating and calls turning it off mandatory on
  terrain; `FunctionalEnv` refuses warp unless
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and the justfile and
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

`hfield/tiles.py` `TileLayout(HFieldConfig)`; the `accel` assets are built
with it (`hfield/arena.py`).

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
  any obs reads it; the critic does). Benchmark, 300 steps (fast walk
  1.2 m/s): 0.15 + 6 x 1.2 + 1.48 = 8.83 m. Gap 7 m, 4161² nodes (208 m,
  69 MB); Go1: `reach` x 2.5, gap 17.5 m.
- `TileLayout.check_reach`: every script that puts several levels on the
  arena (`brax_ued`, `eval_brax`, `eval_sim2sim`) fails at startup when its steps per
  level exceed the budget; training counts the critic's scan, eval only the
  deterministic policy's obs and feet. 1000-step levels need ~28 m: not
  supported.
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
- CPU rendering (`paper_terrains.py`, `render_video.py`)
  uses `hfield.arena.single_tile_spec`: the scene cut to one tile with the
  level at the origin; the renderer meshes the whole hfield, which for the
  full arena is millions of cells.

## Assets {#assets}

`wojtek_env/paths.py` centralises repo-relative paths. Generated arena file sets
(`terrain_paths(kind)`) are `scene` XML, `hfield` binary, `spec` JSON, `lookup`
NPZ, one set per kind in `TERRAIN_KINDS = (train, eval, eval_deep, test,
accel)`; `tests/utils/test_env.py` writes a small tiled arena as `test`.

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
