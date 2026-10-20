# Levels

Heightfield levels the curricula generate, mutate and replay (`src/hfield/`):
the genome, its physical envelope, the arena assets and the level spaces.
The curricula and trainer: [training.md](training.md).

## Level representation

A level is a `coarse_n × coarse_n` array in `[0,1]` (a "genome"), bilinearly
upsampled to the fine MJX heightfield. Chosen over raw full-res edits or the
terrain generator's tile-type params because it is small, smooth by
construction, keeps a flat spawn pad, and edits directly — while reusing
`terrain.py`'s `_bilinear`/pad-taper primitives.

Values are used directly with no per-level min/max stretch: `0.5` is always the
neutral plane, so a small mutation stays physically small across levels — which
is required to keep the fixed elevation envelope (below) valid.

### Envelope: `HFieldConfig`

Plain dataclass (not frozen — `frozen=True` marks the Hydra structured-config
node read-only and blocks CLI overrides like `arena.amplitude=...`);
one place for every knob. `patch_radius`, `cell_size` (fine
resolution), `coarse_n` (genome size), `amplitude` (peak height off the neutral
plane), `base_z`, `pad_radius`/`pad_taper` (flat spawn disc + ramp),
`grid_size` (property) is the fine
node count. Mutation strength is not here: it lives in `level_space`. The
`arena` config group has one option, `benchmark`
(`conf/arena/benchmark.yaml`). Full field table:
[configuration.md](configuration.md#arena).

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
geometry: [environment.md](environment.md#why-the-arena-swap-is-free),
[tiled arena](environment.md#tiled-arena). A level's semantics are
unchanged: genome → `render` → the same `grid_size²` patch, now written into
its tile by `TileLayout.assemble`.

### Arena assets

`hfield/arena.py` writes the static **"accel"** MJCF arena every env loads:
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

## Level spaces

Code: `src/hfield/level_space.py`.

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

