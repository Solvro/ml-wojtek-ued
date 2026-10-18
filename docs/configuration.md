# Configuration

Only the arena builder uses Hydra: `hfield/arena.py` (`just build-accel-arena`)
is `@hydra.main(config_name="arena_config")` over `ArenaConfig`
(`config/arena_config.py`: `hfield_params: HFieldConfig`, `seed`). Every
other script takes argparse flags (`--help`); `brax_ued.py` builds its
`HFieldConfig` from them via `hfield_config(args)`.

```bash
just build-accel-arena hfield_params.amplitude=0.3 hfield_params.coarse_n=13
```

## HFieldConfig {#hfield}

Plain dataclass (`config/hfield_config.py`); the level genome envelope. See
[ued-accel.md](ued-accel.md#level-representation).

| field | default | meaning |
|---|---|---|
| `patch_radius` | `3.0` | m, half-width of the square patch |
| `cell_size` | `0.05` | m, fine hfield resolution |
| `coarse_n` | `9` | control points per axis (genome size) |
| `amplitude` | `0.12` | m, peak height/depth off the neutral plane |
| `difficulty` | `1.0` | runtime terrain-intensity scale in `[0,1]`, applied at render time within the fixed `amplitude` envelope; `brax_ued.py` leaves it at `1.0` and scales levels itself (`LevelSpace.scaled`) |
| `base_z` | `1.0` | m, solid base under the hfield (MJCF needs > 0) |
| `pad_radius` | `0.6` | m, flat spawn disc at the centre |
| `pad_taper` | `0.3` | m, ramp width pad → full terrain |
| `n_edits` | `4` | control points touched per mutation |
| `edit_scale` | `0.1` | max ± change to an edited control point |
| `tiles` | `16` | tiles per side of the shared arena, one level each (256) |
| `reach` | `10.0` | m from its tile centre a robot can read terrain (feet, height scan) within a level; the gap between tiles is `reach - patch_radius` ([tiled arena](environment.md#tiled-arena)) |
| `grid_size` | *(property)* | `round(2·patch_radius/cell_size)+1`, fine node count of one level |

`patch_radius`, `cell_size`, `amplitude`, `base_z`, `tiles`, `reach` set the
compile-time arena (static `hfield_size`, node count); changing them needs a
fresh arena (`just build-accel-arena`). `coarse_n`/`n_edits`/`edit_scale` shape the genome
and mutations and can vary freely between runs.

`patch_radius=3.0` (was `6.0`): at `ctrl_dt=0.02`/`steps_in_episode=200` a
rollout is 4s, so even a fast-command robot covers only a few metres —
most of a 12m-diameter patch was never reached, let alone scored by regret.
`edit_scale=0.1` (was `0.5`): `HFieldManager.mutate`'s intent is a *small*
nudge from an already-curated parent (see the `to_hfield_data` docstring in
[ued-accel.md](ued-accel.md#level-representation)); `0.5` was half the
entire `[0,1]` genome range per edited point, closer to a fresh `generate()`
than an incremental edit.
