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
([ppo.md](ppo.md#shared-env-overrides--utilsenv_overridespy)) turns on the
task-independent ones for `scripts/ppo_dr.py`/`accel.py`/`accel_top_k.py`:

- `latency`, `encoder` — substep control latency and joint-encoder
  miscalibration randomization (sim2real robustness; `action_delay=1` is a
  separate always-on fixed delay).
- `symmetry` — left/right mirror augmentation.
- `real_pose_ref` — judge pose/height against the kinematic standing family
  instead of commanded ctrl targets (removes a PD-sag-dependent penalty
  floor; sim2real correctness fix, not just a shaping knob).
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

- `height_scan` — body-frame grid of terrain heights ahead of the robot; the
  critic gets a clean copy, the actor a masked/stale/optionally-corrupted
  one (`height_scan.corrupt`, off by default, simulates real depth-sensor
  noise — also a sim2real feature). Enabling it requires listing
  `"height_scan"`/`"height_scan_clean"` in `obs.state`/`obs.privileged`
  (`env.py` raises if the obs lists name it without `height_scan.enable`);
  `terrain_overrides()` does both together.
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
`motor_strength`. None of this repo's trainer scripts expose `dr_cfg`
overrides yet — `scripts/ppo_dr.py` (see [ppo.md](ppo.md#domain-randomization-scriptsppo_drpy))
uses the baseline five-field form via `FunctionalEnv(domain_randomize=True)`.

`terrain/terrain_wrapper.py`'s `wrap_for_terrain_brax_training` also accepts a
`randomization_fn` and applies it through mujoco_playground's
`BraxDomainRandomizationVmapWrapper`, but no script currently wires
`randomize.py` through that path — `FunctionalEnv` uses its own swap
mechanism instead (below), consistent with the hfield swap it already does.

## FunctionalEnv — the training bridge

`utils/env.py` `FunctionalEnv` is what the PPO/ACCEL trainers actually use. It
wraps `registry.make_env` + brax `EpisodeWrapper` and exposes pure `reset`/`step`
usable under `jit`/`vmap`.

- **Split obs.** `observation_size` / every `Obs` carries `actor` (`state`) and
  `critic` (`privileged_state`) separately — asymmetric actor-critic.
- **Auto-reset semantics.** `step` freezes `data`/`obs` to the episode's first
  values on `done` (via `first_data`/`first_obs` stashed at reset), so a
  vmapped batch stays aligned.
- **Heightfield swap.** Both `reset` and `step` take optional `hfield_data`.
  `_swapped_model` temporarily does
  `base._mjx_model = model.tree_replace({"hfield_data": hfield_data})` around the
  call, then restores it. This is the mechanism ACCEL uses to run a different
  level per rollout **without recompiling MJX** (see below).
- **Domain randomization swap.** `reset`/`step` also take an optional
  `dr_model` — a full `mjx.Model` (from `FunctionalEnv.sample_dr`, above) that
  `_swapped_model` installs as `_mjx_model` outright instead of
  `tree_replace`-ing one field. The two compose: passing both swaps
  `dr_model` in first, then `tree_replace`s `hfield_data` on top. No script
  currently needs both at once (`scripts/ppo_dr.py` has no hfield curriculum,
  `scripts/accel*.py` build `FunctionalEnv` without `domain_randomize=True`).

## Why hfield swapping is free {#hfield-mutability}

`mjx.Model` registers a field as a dynamic pytree leaf iff it is a `jax.Array`.

- `hfield_data` → `jax.Array` → real dynamic leaf → swappable at runtime and
  **vmappable per env with no MJX recompilation**.
- `hfield_size` (radii, elevation, base) → plain `numpy.ndarray` → static
  metadata, baked in at compile time; changing it needs a new model.

Consequence: fix the grid resolution and elevation envelope once (in
`HFieldConfig`), and the heightfield *values* become a cheap, JAX-native,
batchable "level genome" — which is what makes ACCEL practical at RL scale.

## Assets {#assets}

`wojtek_env/paths.py` centralises repo-relative paths. Generated arena file sets
(`terrain_paths(kind)`) are `scene` XML, `hfield` binary, `spec` JSON, `lookup`
NPZ, one set per kind in `TERRAIN_KINDS = (train, eval, eval_deep, test, accel)`.

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
