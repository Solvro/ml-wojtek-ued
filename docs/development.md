# Development

Repo-specific workflow. Global stack conventions (uv, ruff, ty, pytest, comment
style, structure rules) live in the user's global `CLAUDE.md` and are not
repeated here.

## Layout rules

- **Research code goes in sibling packages under `src/`** — `config/`,
  `env/`, `robots/`, `hfield/`, `training/`, `evaluation/`, `rendering/`,
  `paper/`, `diagnostics/`, `scripts/` ([overview](overview.md#repo-map)).
  Never add files inside `src/wojtek_env/`; it is the extracted env library
  and is ruff-excluded and hatchling-packaged
  (`packages = ["src/wojtek_env"]`).
- **Namespace packages** — research code has no `__init__.py`. Imports are
  top-level and absolute (`from hfield.tiles import ...`), resolved via
  `PYTHONPATH=src` (exported by the `justfile`).
- **Organize by role**, JAX/functional style: small modules grouped by
  function, not one `helpers` grab-bag. Keep the intra-package dependency
  graph acyclic (leaves → composers). Classes only for stateful things
  (students, buffers, robots, contexts).
- **Scripts are thin** — `src/scripts/*.py` parse the Hydra config and call
  one function of a package; logic never lives there. The `just` recipes
  and `slurm/` call the scripts by path, so their names stay put.
- **Run names** — `<prefix>_<robot>_<mode>_seed<N>` (`slurm/`, prefix
  `ued`), evals `<run>_final_eval`; the paper scripts parse
  `<prefix>_<method>_seed<k>` with `prefixes.<robot>` (default
  `ued_<robot>`). Change all three together.

## Python version

This project pins **3.11–3.12** (`requires-python = ">=3.11,<3.13"`), a
deliberate deviation from the global "3.13 only" rule — forced by the
`jax==0.9.2` / `mujoco-mjx==3.10.0` / `warp-lang==1.13.0` set. See
[setup.md](setup.md#pinned-dependencies-do-not-bump-blindly) before touching any
of those pins.

## Tooling

`justfile` recipes:

| recipe | does |
|---|---|
| `just` | list all recipes |
| `just sync` | `uv sync --extra dev` |
| `just install` | `uv lock && uv sync --extra dev` |
| `just fmt` | `ruff format` + `ruff check --fix` on `src` |
| `just lint` | `ruff check` + `ruff format --check` on `src` |
| `just build` | build the MJX model XMLs |
| `just build-accel-arena *args` | write the `accel` terrain arena (`conf/arena.yaml`) |
| `just train *args` | `scripts/brax_ued.py` (`conf/train.yaml`) |
| `just profile *args` | `scripts/profile_brax_ued.py` (`conf/profile.yaml`) |
| `just eval *args` | `scripts/eval_brax.py` (`conf/eval.yaml`) |
| `just sim2sim *args` | `scripts/eval_sim2sim.py` (`conf/sim2sim.yaml`) |
| `just render-video *args` | `scripts/render_video.py` (`conf/render.yaml`) |
| `just paper-tables *args` | `scripts/paper_tables.py` (`conf/paper_tables.yaml`) |
| `just paper-figures *args` | `scripts/paper_figures.py` (`conf/paper_figures.yaml`) |
| `just domination *args` | `scripts/compare_domination.py` (`conf/domination.yaml`) |

`*args` are Hydra overrides ([configuration.md](configuration.md)).

`just lint` covers ruff only; run `ty check` (static types) manually per the
global convention. Tests live in `tests/`, mirroring `src/`
(`tests/evaluation/test_sim2sim.py` for `evaluation/sim2sim.py`);
`uv run pytest -q` (pyproject sets `pythonpath = ["src"]`).
- `@pytest.mark.slow` = end-to-end CPU training runs (the PAIRED variant
  smoke runs in `tests/training/test_train.py`, ~3-6 min each, sharing a
  jax compile cache), deselected by default;
  `JAX_PLATFORMS=cpu uv run pytest -q -m slow` runs them. Run them after
  changing the training loop (`src/training/`).
- ruff runs at 80 columns with `E501` and `C901` (complexity <= 8) on top of
  its defaults.
- ty cannot see mujoco's C-extension members; a `[[tool.ty.overrides]]`
  entry treats `mujoco` as `Any` for the files it lists (add new files that
  use the bindings heavily there instead of per-line ignores).

## Adding to the training loop

- New per-iteration metric → return it from `run_levels`
  (`training/iteration.py`) or a curriculum iteration
  (`training/curricula.py`), or add it in `training/train.py`; it flows
  straight into `run.log`.
- New level operator → extend `HFieldManager` (`hfield/manager.py`) or a
  `LevelSpace` (`hfield/level_space.py`), keeping methods pure-given-an-rng.
- New env task → register it in `wojtek_env/envs/registry.py` (that is the one
  library file you edit, since it is the extension point).
- Changing the level envelope or arena layout
  (`patch_radius`/`cell_size`/`amplitude`/`base_z`/`pad_radius`/`tiles`/`reach`)
  → rebuild the arena with the same override as the run, e.g.
  `just build-accel-arena arena.tiles=8` for `just train ... arena.tiles=8`;
  genome-only fields (`coarse_n`, `level_space.*`) need no rebuild. A new
  benchmark arena = a new `conf/arena/<name>.yaml`.
- New curriculum → a `conf/mode/<name>.yaml` and a paper label in
  `paper/labels.py` (`tests/paper/test_labels.py` fails without it).
- New config field → add it to the schema dataclass in `config/schema.py`
  (field `= MISSING`) and its value to the YAML in `conf/`; the composition
  tests (`tests/config/test_schema.py`) fail on a field left unset.
- New entry point → a schema dataclass in `SCHEMAS`, a `conf/<name>.yaml`
  whose defaults list starts with `<name>_schema`, and
  `@hydra.main(config_path=CONF_DIR, config_name=..., version_base="1.3")`
  converting with `to_dataclass`.

## Gotchas

- **Rebuild assets** after cloning or after any envelope change — the MJX model
  and arenas are gitignored and generated ([setup.md](setup.md#build-generated-assets)).
- **Hydra property gap** — `HFieldConfig.grid_size` is a property; the DictConfig
  doesn't expose it, so every entry point converts the composed config to its
  schema dataclass (`config.schema.to_dataclass`) before use.
- **Overriding a required value** — merging `???` onto a set value keeps the
  set value, so a primary that includes another (`profile.yaml` →
  `train.yaml`) cannot make an inherited field required again.
- **`FunctionalEnv` obs is split** actor/critic — `env.functional.Obs.actor`
  (`state`) and `Obs.critic` (`privileged_state`) are consumed separately.
