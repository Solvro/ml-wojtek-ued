# Results pipeline

From eval dumps and wandb runs to the paper's tables and figures.

## Paper tables

Entry point `scripts/paper_tables.py`; code in `src/paper/tables.py`,
`runs.py` (loading, solvable subset), `latex.py`.

Builds the paper's tables from `eval_brax.py` outputs
(`<prefix>_<method>_seed<k>_final_eval.{json,npz}`); bare booktabs
`tabular`s; the paper (on Overleaf) owns float, caption and label.

```sh
just paper-tables dump_wandb=artifacts/paper/summaries.json
just paper-tables eval_dir=<evals> \
  train_summaries=artifacts/paper/summaries.json
```

- `prefixes.<robot>=<prefix>` (`conf/paper_tables.yaml`, default
  `ued_<robot>`; a robot set to `null` is skipped): one column group / row
  group per robot. Every `conf/mode` option has a paper name
  (`paper/labels.py`), in this order: DR, DR-easy, DR-mixed, PLR⊥, SFL,
  ACCEL, ACCEL-grow, ACCEL-SFL, PAIRED, PAIRED+UniBC, PAIRED+BiBC,
  PAIRED+Evo; other methods follow under their own name; missing ones are
  skipped.
- `main.tex`: success (%) per method x robot: `random/d0.25` mean, its
  CVaR25, `suite/success_rate`, `flat/success_rate`; mean ± std over
  seeds, best per column in bold, `--` if absent.
- `uniform.tex` (appendix): `param/uniform` mean, its CVaR25 and the
  solvable subset per robot x method (why not the main table:
  [Domination comparison](#domination-comparison)).
- Solvable subset: `param/uniform` levels that at least one run of the robot
  (any method) passes on >= 1 episode of pass A; each run is scored on pass
  B on them and vice versa, averaged (cross-fitted, like CVaR). Needs
  `param_uniform_travel` (`level_space=param` evals); otherwise it warns
  and leaves `--`.
- `envelopes.tex`: the 7 `axis/<feature>/envelope` per robot x method, in
  Wojtek units (cm, deg).
- `compute.tex`: GPU-hours (`final/elapsed_h`), sim and student steps per
  robot x method, from `train_summaries` (training run name ->
  wandb summary); not written without it.
- `results.csv`: one row per run, every value the tables use.
- `dump_wandb=FILE`: fetches the training-run summaries for the prefixes
  from the wandb project `ENTITY/PROJECT` (`paper/runs.py`) and writes
  that JSON.
- Robot names in headers: `ROBOT_LABEL` (`wojtek` prints as W01-TEK).

## Paper figures

Entry point `scripts/paper_figures.py`; code in `src/paper/figures.py`,
`terrains.py`.

```sh
just paper-figures dump_history=artifacts/paper/history.json
just paper-figures history=artifacts/paper/history.json
JAX_PLATFORMS=cpu just paper-figures terrains=true
```

- `conf/paper_figures.yaml`; output in `out` (default `artifacts/paper/figs`).
- `dump_history`: per finished run, `eval/d{0.5,1.0}/success_rate` and the
  `level/*` composition vs `env_steps` from wandb; rows with a NaN are dropped.
- `curves.pdf`: eval success vs env steps, robot x d panels, seed mean with
  min-max band.
- `composition.pdf`: mean physical value of the magnitude parameters and
  genome distance among training levels, Wojtek units, running mean over 9 points.
- `terrains.pdf` / `robots.pdf` (`paper/terrains.py`): offscreen MuJoCo
  renders of one level per feature at full strength plus uniform samples, and
  all five robots on one shared genome. Needs the built assets (`just build`,
  `just build-accel-arena`); the recipe sets `MUJOCO_GL` as `render-video`
  does, and `arena` (default `benchmark`) is the arena the levels render on.
- Overleaf: upload the PDFs into a `figs/` folder; `\resultfig` falls back to a
  placeholder while a file is missing.

## Domination comparison

Entry point `scripts/compare_domination.py`; code in
`src/evaluation/domination.py` (loading, pairing check, solve rates,
histogram, stats, matrix) and `src/paper/domination.py` (figures). Design by
Piotr Wojdas (pairing check, binomial noise reference, bootstrap CI,
single-hue colour ramp).

```sh
just domination eval_dir=<evals> method_a=accel method_b=dr
```

- Why: success rate and CVaR are marginals; two methods can match on both
  and still differ per level (one better everywhere, or each winning on
  disjoint levels). The comparison pairs them level by level.
- Definition: per level θ of the set, `solve_rate(θ)` = fraction of
  episodes with no fall and travel >= the robot's threshold
  (`robots.scale.min_travel`, eval_brax's success rule) over both passes (2 x 4 = 8 episodes), then the mean over
  the method's seeds (5 seeds: values k/40).
  H(x, y) = number of levels with A's rate in bin x and B's in bin y.
- Inputs: `eval_dir` of eval_brax npz dumps
  `<prefix>_<method>_seed<k>_final_eval.npz`, `prefixes.<robot>` as in
  `paper_tables`; methods and seeds come from the file names.
  `level_set` picks the per-level set: `random_d0.25` (default) or another
  `random_d<d>`, or `param_uniform` (needs `level_space=param` evals); a
  dump without it fails naming the sets it has.
  - The paper's main comparison is `random_d0.25`; `param_uniform` goes to
    the appendix (`out=artifacts/paper/domination_param_uniform`): the
    Playground robots solve few `param_uniform` levels (sub-cell gap pits,
    [levels.md](levels.md#level-spaces)), so most levels tie at 0.
- Pairing: every dump of a robot must hold the same level genomes
  (`param_uniform_levels`, `random_levels` for `random_d*`); otherwise it
  refuses (rerun eval_brax with the same seed / `num_levels` /
  `level_space`).
- Gaps are reported, not dropped: a method lacking seeds another method
  of the robot has, a method other robots have, and a robot without both
  `method_a` and `method_b` (its heatmap is skipped) warn and go to the
  JSON (`missing_seeds`, `missing_methods`, `pair: null`). No robot with
  the pair is an error.
- Writes to `out` (default `artifacts/paper/domination`):
  - `domination.json`: per robot `num_levels`, per method seeds,
    `episodes_per_level` (all seeds), `mean_solve_rate`; the pair's
    `a_better` / `b_better` / `ties` (fractions of levels, ties = equal
    rate), `mean_diff` (A - B) with `mean_diff_ci` (95 % bootstrap over
    levels), `mean_abs_diff`, `noise_abs_diff`, `counts` (bins x bins level
    counts, row = A's bin, column = B's bin); `matrix`.
  - `domination_<a>_vs_<b>.pdf`: one heatmap per robot in a row, plus
    `domination_<a>_vs_<b>_<robot>.{pdf,png}`.
  - `domination_matrix.pdf`: per robot, methods x methods, cell = fraction
    of levels where the row method's rate beats the column's
    (`row_better`; the diagonal is 0, `M[a,b] + M[b,a] + ties = 1`).
- Reading the heatmap: x = A's solve rate, y = B's, `bins` (10) bins of
  10 %, the last one closed at 100 %; cells show the fraction of levels
  (sum 1) on one log scale for every robot, from a single level of the
  largest set to all levels, empty cells white. Mass
  below the diagonal = levels A solves more often (A dominates), above = B;
  the bottom-left / top-right corners are levels both fail / both solve.
  Each panel's title has the `a_better` / `b_better` / tie fractions,
  `mean_diff` with its CI and `mean_abs_diff` against the noise reference.
- Noise reference (`noise_abs_diff`): the mean |A - B| two identical
  policies would show from episode sampling alone. Per level, both rates
  are drawn as Binomial(n, p) / n at p = (A + B) / 2 with each method's
  episode count n (seeds x passes x episodes); the exact expectation,
  averaged over levels. At p = 0.5 it is 0.20 for 8 episodes, 0.09 for
  40 (5 seeds). `mean_abs_diff` near it means the per-level
  spread is sampling noise; it ignores seed-to-seed policy variance, so it
  is a lower bound on the spread of two equally good methods.
