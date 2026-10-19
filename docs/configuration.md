# Configuration

Every entry point is a `@hydra.main` over a YAML config in `conf/` (repo
root). Arguments are Hydra overrides (`key=value`, `group=option`); there are
no `--flags`.

```bash
python src/scripts/brax_ued.py robot=go1 mode=accel seed=1 name=x
python src/scripts/eval_brax.py checkpoint=<pkl> name=x out_dir=<dir> robot=go1
just build-accel-arena                     # arena=v10
just render-video checkpoint=<pkl> 'levels=[0.75:3,1.0:17]'
python src/scripts/brax_ued.py --cfg job mode=dr name=x   # print the composed config
```

- Quote list overrides (`'levels=[...]'`, `'tags=[a,b]'`) in zsh, which globs `[...]`.
- **Typed**: each primary's defaults list starts with its schema, a dataclass in
  `src/config/schema.py` registered in Hydra's ConfigStore
  (`<primary>_schema`). YAML holds every value (schema fields are `???`);
  a misspelled key, a wrong type, or a value outside an enum
  (`backend=cuda`, `mode.score=learnabilty`) fails at startup, before any
  jax work. `???` in YAML = required on the command line.
- **Record**: `hydra.job.chdir: false`; `hydra.run.dir` is
  `artifacts/hydra/${name}` (train / eval / sim2sim) or
  `artifacts/hydra/<entry point>/<timestamp>`; its `.hydra/config.yaml` is the
  composed config of the run. Checkpoint / eval / video paths are the
  config's own (`checkpoint_dir`, `out_dir`, `out`), relative to the launch
  directory.
- `conf/` resolves from `src/..` (`config.schema.CONF_DIR`), like the assets:
  a frozen copy (`SRC` on WCSS) needs `conf/` beside `src/`.
- wandb gets the composed config (`config.schema.as_dict`), nested
  (`mode.curriculum`, `robot.name`, ...).

## Entry points

| primary | script | `just` | groups |
|---|---|---|---|
| `train.yaml` | `scripts/brax_ued.py` | `train` | robot, mode, level_space, arena |
| `profile.yaml` | `scripts/profile_brax_ued.py` | `profile` | `train.yaml` with `mode: dr`, `level_space: param`, `name: profile` |
| `eval.yaml` | `scripts/eval_brax.py` | `eval` | robot, level_space, arena |
| `sim2sim.yaml` | `scripts/eval_sim2sim.py` | `sim2sim` | robot, arena |
| `render.yaml` | `scripts/render_video.py` | `render-video` | robot, level_space, arena |
| `arena.yaml` | `python -m hfield.arena` | `build-accel-arena` | arena |
| `check_hfield.yaml` | `scripts/check_hfield_batch.py` | — | arena |
| `paper_tables.yaml` | `scripts/paper_tables.py` | `paper-tables` | — |
| `paper_figures.yaml` | `scripts/paper_figures.py` (+ `paper_terrains.py`) | `paper-figures` | arena |

## Groups

### `robot/` — `RobotConfig`

| option | `name` |
|---|---|
| `wojtek` (default) | `wojtek` |
| `go1` | `go1` — Playground Go1 ([ued-accel.md](ued-accel.md#robots)) |
| `barkour` | `barkour` — Barkour vB |
| `spot` | `spot` — Spot |
| `go2` | `go2` — Unitree Go2 (Go1's env on the Go2 model) |

The robot's size relative to Wojtek is `ROBOT_SCALE` in
`scripts/brax_ued.py` (each Playground robot's `SCALE`, `robots/*.py`: go1
2.5, barkour 2.4, spot 4.3, go2 2.5); it scales
the arena, level lengths, commands and `min_travel`.

### `mode/` — `ModeConfig` (train only, required)

`train.yaml`'s `mode:` block holds the defaults every option starts from;
each `mode/<option>.yaml` sets its switches.

| option | `curriculum` | switches |
|---|---|---|
| `dr` | `dr` | — |
| `dr_easy` | `dr_easy` | — |
| `dr_mixed` | `dr_mixed` | — |
| `accel` | `accel` | — |
| `accel_grow` | `accel` | `mutation: grow` |
| `accel_sfl` | `accel` | `mutation: grow`, `score: learnability` |
| `plr` | `plr` | — |
| `paired` | `paired` | — |

| field | default | meaning |
|---|---|---|
| `curriculum` | — | `dr` / `dr_easy` / `dr_mixed` / `accel` / `plr` / `paired` ([ued-accel.md](ued-accel.md#brax-ued)) |
| `replay_prob` | `0.8` | ACCEL/PLR: replay probability once the buffer holds `num_lanes` levels |
| `buffer_size` | `4000` | ACCEL/PLR: levels kept (highest score) |
| `generator_scale` | `0.1` | easy generator: control points within this of flat (`dr_easy`, ACCEL explore) |
| `mutation` | `local` | `local` edits, or `grow`: also scale the level away from flat by `U(1, grow_max)` |
| `grow_max` | `1.3` | upper bound of the `grow` factor |
| `score` | `pvl` | level score: positive value loss, or `learnability` `p(1-p)` |
| `generator` | `easy` | ACCEL explore levels: `easy` generator or `mixed` (`dr_mixed` draw) |
| `latent_dim` | `16` | PAIRED adversary latent size |
| `gen_learning_rate` | `1e-4` | PAIRED adversary Adam step |
| `gen_entropy_coef` | `0.1` | PAIRED adversary entropy bonus |

### `level_space/` — `LevelSpaceConfig`

| option | `kind` | `n_edits` | `edit_scale` |
|---|---|---|---|
| `grid` (train / eval default) | `grid` | `8` | `0.15` |
| `param` (render default; v10) | `param` | `2` | `0.1` |

`n_edits` / `edit_scale`: ACCEL mutation strength, training only (eval and
render read `kind`). They replace `arena.n_edits` / `arena.edit_scale` in the
training `HFieldConfig`. Spaces: [ued-accel.md](ued-accel.md#level-spaces).

### `arena/` — `HFieldConfig` {#hfield}

`v10` (the only option, default everywhere) is the benchmark arena:
`amplitude: 0.3`, `coarse_n: 13`, `pad_radius: 0.55`, `pad_taper: 0.2`; the
rest are `HFieldConfig` defaults. Values are in Wojtek units; training,
eval, sim2sim, render and the paper figures scale every length by the robot
(`hfield_config` in `scripts/brax_ued.py`). The arena builder writes the
`arena` it is given, and every Wojtek run checks the built arena against its
own `arena` (`hfield.arena.check_arena`), so an override must go to both:

```bash
just build-accel-arena arena.tiles=8
just train mode=dr name=x arena.tiles=8
```

Plain dataclass (`config/hfield_config.py`); the level genome envelope. See
[ued-accel.md](ued-accel.md#level-representation).

| field | dataclass default | v10 | meaning |
|---|---|---|---|
| `patch_radius` | `3.0` | | m, half-width of the square patch |
| `cell_size` | `0.05` | | m, fine hfield resolution |
| `coarse_n` | `9` | `13` | control points per axis (genome size) |
| `amplitude` | `0.12` | `0.3` | m, peak height/depth off the neutral plane |
| `difficulty` | `1.0` | | runtime terrain-intensity scale in `[0,1]`; `brax_ued.py` leaves it at `1.0` and scales levels itself (`LevelSpace.scaled`) |
| `base_z` | `1.0` | | m, solid base under the hfield (MJCF needs > 0) |
| `pad_radius` | `0.6` | `0.55` | m, flat spawn disc at the centre |
| `pad_taper` | `0.3` | `0.2` | m, ramp width pad → full terrain |
| `n_edits` | `4` | | control points touched per mutation (training: `level_space.n_edits`) |
| `edit_scale` | `0.1` | | max ± change to an edited control point (training: `level_space.edit_scale`) |
| `tiles` | `16` | | tiles per side of the shared arena, one level each (256) |
| `reach` | `10.0` | | m from its tile centre a robot can read terrain within a level; the gap between tiles is `reach - patch_radius` ([tiled arena](environment.md#tiled-arena)) |
| `grid_size` | *(property)* | | `round(2·patch_radius/cell_size)+1`, fine node count of one level |

`patch_radius`, `cell_size`, `amplitude`, `base_z`, `pad_radius`, `tiles`,
`reach` set the compile-time arena; changing them needs a fresh
`just build-accel-arena` with the same overrides. `coarse_n` and the edits
shape the genome and can vary between runs.

`patch_radius=3.0` (was `6.0`): at `ctrl_dt=0.02`/`steps_in_episode=200` a
rollout is 4s, so even a fast-command robot covers only a few metres —
most of a 12m-diameter patch was never reached, let alone scored by regret.
`edit_scale=0.1` (was `0.5`): `HFieldManager.mutate`'s intent is a *small*
nudge from an already-curated parent; `0.5` was half the entire `[0,1]`
genome range per edited point, closer to a fresh `generate()` than an
incremental edit.

## Primaries

`???` = required. `name` / `entity` / `project` are the wandb run, entity
(`""` = default) and project.

### `train.yaml` — `TrainConfig`

| field | default | meaning |
|---|---|---|
| `name` | `???` | wandb run name; checkpoint `<checkpoint_dir>/<name>.pkl` |
| `entity` / `project` | `""` / `wojtyla` | wandb |
| `backend` | `auto` | physics: `auto` (warp on a GPU, jax on CPU) / `jax` / `warp` |
| `seed` | `0` | |
| `num_timesteps` | `300000000` | env steps (protagonist only for PAIRED) |
| `num_lanes` | `256` | levels per iteration, ≤ `arena.tiles²` |
| `envs_per_lane` | `32` | envs per level |
| `level_steps` | `300` | steps per iteration, multiple of the 20-step unroll |
| `eval_every` | `10` | iterations between in-training evals |
| `eval_levels` | `512` | uniform levels per eval difficulty |
| `min_travel` | `1.0` | m (Wojtek units) from spawn for a success |
| `checkpoint_dir` | `null` | no checkpoint when null |
| `tags` | `[]` | extra wandb tags (always: `brax-ued`, curriculum, robot) |

### `profile.yaml` — `TrainConfig`

`train.yaml` with `mode: dr`, `level_space: param`, `name: profile`; set
`backend`, `robot`, `num_lanes`, `envs_per_lane`, `level_steps`,
`eval_levels` as in training (`just profile backend=warp`). The printed
record's `backend` is the config value.

### `eval.yaml` — `EvalConfig`

| field | default | meaning |
|---|---|---|
| `checkpoint` | `???` | `brax_ued.py` `.pkl` params |
| `name` | `???` | wandb run name; `<out_dir>/<name>.{json,npz}` |
| `out_dir` | `???` | |
| `entity` / `project` | `""` / `wojtyla` | |
| `backend` | `auto` | score on the backend the run trained on |
| `num_levels` | `2048` | levels per random set |
| `episodes` | `4` | per level per pass |
| `suite_instances` | `64` | levels per suite cell, flat set, axis step |
| `chunk_envs` | `4096` | envs per batch |
| `level_steps` | `300` | |
| `min_travel` | `1.0` | m, Wojtek units |
| `alphas` | `[0.1, 0.25]` | CVaR levels |
| `seed` | `1234` | |
| `tags` | `[]` | extra wandb tags |

### `sim2sim.yaml` — `Sim2SimConfig`

As `eval.yaml` minus `backend` (always jax: `CpuPhysics` steps the jax env's
data layout), `level_space` (always `param`), `alphas` and `tags`;
`num_levels` `256`, `chunk_envs` `1024`.

### `render.yaml` — `RenderConfig`

| field | default | meaning |
|---|---|---|
| `checkpoint` | `???` | |
| `levels` | `???` | `D:SEED` strings, one clip each (`'levels=[0.75:3,1.0:17]'`) |
| `out_dir` | `artifacts/videos` | |
| `seconds` | `20.0` | must fit in one episode |
| `fps` / `width` / `height` | `30` / `1280` / `720` | |
| `command` | `[0.3, 0.0, 0.2]` | held `[VX, VY, WZ]` (m/s, m/s, rad/s) |
| `seed` | `0` | reset key |
| `backend` | `auto` | |

### `arena.yaml` — `ArenaBuildConfig`

`arena` (group, `v10`) and `seed` (`0`). Writes the `accel` arena assets.

### `check_hfield.yaml` — `CheckHFieldConfig`

| field | default | meaning |
|---|---|---|
| `backend` | `auto` | |
| `pit` | `0.05` | m below the flat plane |
| `tiles` | `8` | tiles in use |
| `envs_per_tile` | `2` | |
| `steps` | `100` | zero-action steps |
| `compare` | `false` | run jax and warp, fail on disagreement |
| `tol` | `0.005` | m, `compare` pit-depth tolerance |
| `bench_envs` | `0` | > 0: time env steps on that many envs instead |

### `paper_tables.yaml` / `paper_figures.yaml`

| field | default | meaning |
|---|---|---|
| `prefixes.<robot>` (`wojtek`, `go1`, `barkour`, `spot`, `go2`) | `null` | run-name prefix per robot; robots left null are skipped |
| `eval_dir` | `null` | tables: dir of `eval_brax.py` outputs (required unless `dump_wandb`) |
| `train_summaries` | `null` | tables: JSON from `dump_wandb`, for `compute.tex` |
| `dump_wandb` | `null` | tables: write training summaries JSON, exit |
| `dump_history` | `null` | figures: write wandb histories JSON, exit |
| `history` | `null` | figures: JSON from `dump_history` → `curves.pdf`, `composition.pdf` |
| `terrains` | `false` | figures: render `terrains.pdf` / `robots.pdf` on `arena` |
| `out` | `artifacts/paper/tables` / `artifacts/paper/figs` | |
