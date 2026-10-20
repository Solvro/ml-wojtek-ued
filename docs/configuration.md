# Configuration

Every entry point is a `@hydra.main` over a YAML config in `conf/` (repo
root). Arguments are Hydra overrides (`key=value`, `group=option`); there are
no `--flags`.

```bash
python src/scripts/brax_ued.py robot=go1 mode=accel seed=1 name=x
python src/scripts/eval_brax.py checkpoint=<pkl> name=x out_dir=<dir> robot=go1
just build-accel-arena                     # arena=benchmark
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
  `artifacts/hydra/${run_name:}` (train / eval / sim2sim; a missing
  `name=` fails with "name is required ... pass name=<run name>") or
  `artifacts/hydra/<entry point>/<timestamp>`; its `.hydra/config.yaml` is the
  composed config of the run. Checkpoint / eval / video paths are the
  config's own (`checkpoint_dir`, `out_dir`, `out`), relative to the launch
  directory.
- `HYDRA_FULL_ERROR=1` is exported in `slurm/*.sbatch` (full tracebacks in
  job logs).
- `conf/` resolves from `src/..` (`config.schema.CONF_DIR`), like the assets:
  a frozen copy (`SRC` in `slurm/`) needs `conf/` beside `src/`.
- wandb gets the composed config (`config.schema.as_dict`), nested
  (`mode.curriculum`, `robot.name`, ...).

## Entry points

| primary | script | `just` | groups |
|---|---|---|---|
| `train.yaml` | `scripts/brax_ued.py`, `scripts/train_export.py` | `train`, `train-export` | robot, mode, level_space, arena, network, policy (optional) |
| `export.yaml` | `scripts/export_policy.py` | `export` | `train.yaml` + `checkpoint`, `config` |
| `profile.yaml` | `scripts/profile_brax_ued.py` | `profile` | `train.yaml` with `mode: dr`, `name: profile` |
| `eval.yaml` | `scripts/eval_brax.py` | `eval` | robot, level_space, arena, network |
| `sim2sim.yaml` | `scripts/eval_sim2sim.py` | `sim2sim` | robot, arena, network |
| `render.yaml` | `scripts/render_video.py` | `render-video` | robot, level_space, arena, network |
| `arena.yaml` | `python -m hfield.arena` | `build-accel-arena` | arena |
| `check_hfield.yaml` | `scripts/check_hfield_batch.py` | — | arena |
| `paper_tables.yaml` | `scripts/paper_tables.py` | `paper-tables` | — |
| `paper_figures.yaml` | `scripts/paper_figures.py` (+ `paper/terrains.py`) | `paper-figures` | arena |
| `domination.yaml` | `scripts/compare_domination.py` | `domination` | — |

## Groups

### `robot/` — `RobotConfig`

| option | `name` |
|---|---|
| `wojtek` (default) | `wojtek` |
| `go1` | `go1` — Playground Go1 ([robots.md](robots.md)) |
| `barkour` | `barkour` — Barkour vB |
| `spot` | `spot` — Spot |
| `go2` | `go2` — Unitree Go2 (Go1's env on the Go2 model) |

The robot's size relative to Wojtek is `ROBOT_SCALE` in
`robots/scale.py` (each Playground robot's `SCALE`, `robots/*.py`: go1
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
| `sfl` | `plr` | `score: learnability` |
| `paired` | `paired` | — |
| `paired_uni` | `paired` | `bc: uni` |
| `paired_bi` | `paired` | `bc: bi` |
| `paired_evo` | `paired_evo` | — (`mode.bc=uni\|bi` for BC) |

| field | default | meaning |
|---|---|---|
| `curriculum` | — | `dr` / `dr_easy` / `dr_mixed` / `accel` / `plr` / `paired` / `paired_evo` ([training.md](training.md#trainer)) |
| `replay_prob` | `0.8` | ACCEL/PLR/PAIRED+Evo: replay probability once the buffer holds `num_lanes` levels |
| `buffer_size` | `4000` | ACCEL/PLR/PAIRED+Evo: levels kept (highest score) |
| `generator_scale` | `0.1` | easy generator: control points within this of flat (`dr_easy`, ACCEL explore) |
| `mutation` | `local` | `local` edits, or `grow`: also scale the level away from flat by `U(1, grow_max)` |
| `grow_max` | `1.3` | upper bound of the `grow` factor |
| `score` | `pvl` | level score: positive value loss, or `learnability` `p(1-p)`; `paired_evo` scores PAIRED regret and requires `pvl` |
| `generator` | `easy` | ACCEL explore levels: `easy` generator or `mixed` (`dr_mixed` draw) |
| `latent_dim` | `16` | PAIRED adversary latent size |
| `gen_learning_rate` | `1e-4` | PAIRED adversary Adam step |
| `gen_entropy_coef` | `0.1` | PAIRED adversary entropy bonus |
| `student_entropy_coef` | `0.01` | every student's PPO entropy bonus (brax `entropy_cost`); raise for PAIRED HiEnt |
| `bc` | `none` | PAIRED / PAIRED+Evo distillation: `none`, `uni` (protagonist <- antagonist), `bi` (also antagonist <- protagonist); other curricula fail on non-`none` |
| `bc_coef` | `0.5` | weight of `KL(pi_teacher \|\| pi_student)` in the student loss |
| `bc_every` | `1` | distil on iterations with `iteration % bc_every == 0` (>= 1) |

### `level_space/` — `LevelSpaceConfig`

| option | `kind` | `n_edits` (`local` / `grow`) | `edit_scale` |
|---|---|---|---|
| `param` (default: train, eval, render) | `param` | `2` / `2` | `0.1` |
| `grid` | `grid` | `8` / `16` | `0.15` |

`n_edits` / `edit_scale`: ACCEL mutation strength, training only (eval and
render read `kind`); the only place they exist (`HFieldConfig` is arena
geometry, so `arena.n_edits=...` fails at startup). `n_edits` picks from
`n_edits_by_mutation` by `mode.mutation`, so `mode=accel_grow` / `accel_sfl`
on `level_space=grid` compose to 16 edits; an explicit
`level_space.n_edits=N` overrides it. Spaces:
[levels.md](levels.md#level-spaces).

### `arena/`

Typed by `HFieldConfig`. `benchmark` (the only option, default everywhere):
`amplitude: 0.3`, `coarse_n: 13`, `pad_radius: 0.55`, `pad_taper: 0.2`; the
rest are `HFieldConfig` defaults. Values are in Wojtek units; training,
eval, sim2sim, render and the paper figures scale every length by the robot
(`hfield_config` in `robots/scale.py`). The arena builder writes the
`arena` it is given, and every Wojtek run checks the built arena against its
own `arena` (`hfield.arena.check_arena`), so an override must go to both:

```bash
just build-accel-arena arena.tiles=8
just train mode=dr name=x arena.tiles=8
```

Plain dataclass (`config/hfield_config.py`); the level genome envelope. See
[levels.md](levels.md#level-representation).

| field | dataclass default | `benchmark` | meaning |
|---|---|---|---|
| `patch_radius` | `3.0` | | m, half-width of the square patch |
| `cell_size` | `0.05` | | m, fine hfield resolution |
| `coarse_n` | `9` | `13` | control points per axis (genome size) |
| `amplitude` | `0.12` | `0.3` | m, peak height/depth off the neutral plane |
| `difficulty` | `1.0` | | runtime terrain-intensity scale in `[0,1]`; `brax_ued.py` leaves it at `1.0` and scales levels itself (`LevelSpace.scaled`) |
| `base_z` | `1.0` | | m, solid base under the hfield (MJCF needs > 0) |
| `pad_radius` | `0.6` | `0.55` | m, flat spawn disc at the centre |
| `pad_taper` | `0.3` | `0.2` | m, ramp width pad → full terrain |
| `tiles` | `16` | | tiles per side of the shared arena, one level each (256) |
| `reach` | `10.0` | | m from its tile centre a robot can read terrain within a level; the gap between tiles is `reach - patch_radius` ([tiled arena](environment.md#tiled-arena)) |
| `grid_size` | *(property)* | | `round(2·patch_radius/cell_size)+1`, fine node count of one level |

`patch_radius`, `cell_size`, `amplitude`, `base_z`, `pad_radius`, `tiles`,
`reach` set the compile-time arena; changing them needs a fresh
`just build-accel-arena` with the same overrides. `coarse_n` and the edits
shape the genome and can vary between runs.

- `patch_radius=3.0`: a 300-step (6 s) level takes a robot a few metres
  from spawn, so a larger patch would mostly go unwalked and unscored.
- `level_space.edit_scale` is small (`0.1`-`0.15`): an ACCEL mutation is a
  small nudge from a curated parent, not a fresh `generate()`.

### `network/` — `NetworkConfig`

`default` (the only option; `train`, `export`, `eval`, `sim2sim`, `render`):
the policy / value MLPs of the brax PPO student.

| field | default | meaning |
|---|---|---|
| `policy_hidden` | `[512, 256, 128]` | policy hidden layer sizes; the exported actor |
| `value_hidden` | `[512, 256, 128]` | value (critic) hidden layer sizes |
| `activation` | `elu` | `elu` / `relu` / `tanh` / `swish` |

A checkpoint is only readable with the `network` it was trained with:
eval, sim2sim, render and export build the network from the same fields
(`network.policy_hidden=[...]`, or `policy=<name>` for export).

### `policy/` — optional, `train` and `export`

`conf/policy/<name>.yaml`: a whole policy definition (`# @package _global_`)
selected with `policy=<name>`; `train.yaml` lists the group as `null`, so
benchmark runs never read it. A policy file overrides the `robot`, `mode`
and `level_space` options in its `defaults` and sets any other field in its
body; command-line overrides win. `template.yaml` is the starting point
([export.md](export.md#1-define-the-policy)).

## Primaries

`???` = required. `name` / `entity` / `project` are the wandb run, entity
(`""` = default) and project.

### `train.yaml` — `TrainConfig`

| field | default | meaning |
|---|---|---|
| `name` | `???` | wandb run name; checkpoint `<checkpoint_dir>/<name>.pkl` |
| `entity` / `project` | `""` / `wojtek-benchmark-iclr` | wandb |
| `backend` | `auto` | physics: `auto` (warp on a GPU, jax on CPU; jax for the Playground robots) / `jax` / `warp` (refused for them) |
| `seed` | `0` | |
| `num_timesteps` | `300000000` | env steps (protagonist only for PAIRED) |
| `num_lanes` | `256` | levels per iteration, ≤ `arena.tiles²` |
| `envs_per_lane` | `32` | envs per level |
| `level_steps` | `300` | steps per iteration, multiple of the 20-step unroll |
| `eval_every` | `10` | iterations between in-training evals |
| `eval_levels` | `512` | uniform levels per eval difficulty |
| `min_travel` | `1.0` | m (Wojtek units) from spawn for a success |
| `checkpoint_dir` | `null` | no checkpoint when null; `train-export` requires it |
| `export_dir` | `artifacts/export` | `train-export` writes `<export_dir>/<name>/` |
| `tags` | `[]` | extra wandb tags (always: `brax-ued`, curriculum, robot) |
| `ppo.*` | | brax PPO, see below |

`ppo` (`PPOConfig`; mujoco_playground's locomotion recipe, also logged to
wandb as `ppo.*`):

| field | default | meaning |
|---|---|---|
| `unroll_length` | `20` | steps per PPO update; `level_steps` is a multiple |
| `num_minibatches` | `32` | |
| `num_updates_per_batch` | `4` | epochs over one unroll |
| `discounting` | `0.97` | |
| `gae_lambda` | `0.95` | |
| `learning_rate` | `3e-4` | Adam |
| `clipping_epsilon` | `0.2` | |
| `max_grad_norm` | `1.0` | global-norm clip |

A checkpoint (`<checkpoint_dir>/<name>.pkl`) is followed by
`<name>.train.yaml`: means of the training metrics over the run's last 10
iterations (the export README reads it).

### `export.yaml` — `ExportConfig`

`train.yaml` plus:

| field | default | meaning |
|---|---|---|
| `checkpoint` | `???` | the `.pkl` of a run |
| `config` | `null` | a composed config YAML (a run's `.hydra/config.yaml`, an export's `config.yaml`) whose values replace the composed ones, except `checkpoint_dir`, `export_dir`; it holds the mode, so `mode=` is not needed with it |

The training config comes from `policy=<name>`, from `config=`, or from
overrides ([export.md](export.md#3-export-an-existing-checkpoint)).
`hydra.run.dir` is `artifacts/hydra/export`.

### `profile.yaml` — `TrainConfig`

`train.yaml` with `mode: dr`, `name: profile`; set
`backend`, `robot`, `num_lanes`, `envs_per_lane`, `level_steps`,
`eval_levels` as in training (`just profile backend=warp`). The printed
record's `backend` is the config value.

### `eval.yaml` — `EvalConfig`

| field | default | meaning |
|---|---|---|
| `checkpoint` | `???` | `brax_ued.py` `.pkl` params |
| `name` | `???` | wandb run name; `<out_dir>/<name>.{json,npz}` |
| `out_dir` | `???` | |
| `entity` / `project` | `""` / `wojtek-benchmark-iclr` | |
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

`arena` (group, `benchmark`) and `seed` (`0`). Writes the `accel` arena assets.

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
| `prefixes.<robot>` (`wojtek`, `go1`, `barkour`, `spot`, `go2`) | `ued_<robot>` | run-name prefix per robot; a robot set to `null` is skipped |
| `eval_dir` | `null` | tables: dir of `eval_brax.py` outputs (required unless `dump_wandb`) |
| `train_summaries` | `null` | tables: JSON from `dump_wandb`, for `compute.tex` |
| `dump_wandb` | `null` | tables: write training summaries JSON, exit |
| `dump_history` | `null` | figures: write wandb histories JSON, exit |
| `history` | `null` | figures: JSON from `dump_history` → `curves.pdf`, `composition.pdf` |
| `terrains` | `false` | figures: render `terrains.pdf` / `robots.pdf` on `arena` |
| `out` | `artifacts/paper/tables` / `artifacts/paper/figs` | |

### `domination.yaml` — `DominationConfig`

[results.md](results.md#domination-comparison).

| field | default | meaning |
|---|---|---|
| `prefixes.<robot>` | `ued_<robot>` | run-name prefix per robot, as in `paper_tables`; a robot set to `null` is skipped |
| `eval_dir` | `???` | dir of `eval_brax.py` npz dumps |
| `level_set` | `param_uniform` | per-level npz set: `param_uniform` or `random_d<d>` (`random_d1.0`, ...) |
| `method_a` / `method_b` | `accel` / `dr` | the heatmap's pair: A on x, B on y |
| `min_travel` | `1.0` | m (Wojtek units, scaled by robot) for a solved episode, as eval_brax |
| `bins` | `10` | heatmap bins per axis |
| `out` | `artifacts/paper/domination` | `domination.json` and figures |
