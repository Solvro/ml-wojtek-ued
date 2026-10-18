# Development

Repo-specific workflow. Global stack conventions (uv, ruff, ty, pytest, comment
style, structure rules) live in the user's global `CLAUDE.md` and are not
repeated here.

## Layout rules

- **Research code goes in sibling packages under `src/`** — `config/`,
  `modules/`, `utils/`, `hfield/`, `robots/`, `scripts/`. Never add files inside
  `src/wojtek_env/`; it is the extracted env library and is ruff-excluded and
  hatchling-packaged (`packages = ["src/wojtek_env"]`).
- **Namespace packages** — research code has no `__init__.py`. Imports are
  top-level and absolute (`from hfield.tiles import ...`), resolved via
  `PYTHONPATH=src` (exported by the `justfile`).
- **Organize by role**, JAX/functional style: small modules grouped by
  function, not one `helpers` grab-bag. Keep the intra-package dependency
  graph acyclic (leaves → composers).

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
| `just build-accel-arena` | write the `accel` terrain arena |
| `just render-video *args` | mp4 clips of a `brax_ued.py` checkpoint (`scripts/render_video.py`) |

`just lint` covers ruff only; run `ty check` (static types) manually per the
global convention. Tests live in `tests/`, mirroring `src/`
(`tests/scripts/test_eval_sim2sim.py`); `uv run pytest -q` (pyproject sets
`pythonpath = ["src"]`).
- ty cannot see mujoco's C-extension members; a `[[tool.ty.overrides]]`
  entry treats `mujoco` as `Any` for the files it lists (add new files that
  use the bindings heavily there instead of per-line ignores).

## Adding to the training loop

- New per-iteration metric → add it to the `log` dict in
  `scripts/brax_ued.py`; it flows straight into `run.log`.
- New level operator → extend `HFieldManager` (`hfield/manager.py`) or a
  `LevelSpace` (`hfield/level_space.py`), keeping methods pure-given-an-rng.
- New env task → register it in `wojtek_env/envs/registry.py` (that is the one
  library file you edit, since it is the extension point).
- Changing the level envelope or arena layout
  (`patch_radius`/`cell_size`/`amplitude`/`base_z`/`tiles`/`reach`)
  → rebuild the arena with the same override, e.g.
  `just build-accel-arena hfield_params.amplitude=0.30`; genome-only fields
  (`coarse_n`/`n_edits`/`edit_scale`) need no rebuild.

## Gotchas

- **Rebuild assets** after cloning or after any envelope change — the MJX model
  and arenas are gitignored and generated ([setup.md](setup.md#build-generated-assets)).
- **Hydra property gap** — `HFieldConfig.grid_size` is a property; the DictConfig
  doesn't expose it, so `hfield/arena.py` materialises the config with
  `OmegaConf.to_object`. Do the same if you read `grid_size` off Hydra config.
- **`FunctionalEnv` obs is split** actor/critic — `utils.env.Obs.actor`
  (`state`) and `Obs.critic` (`privileged_state`) are consumed separately.
