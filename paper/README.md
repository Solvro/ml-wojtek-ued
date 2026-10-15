# Paper draft

Workshop paper: UED benchmark on Wojtek (closed chain) and Go1 (open chain).

## Build

```sh
cd paper && tectonic main.tex      # or: just paper (from the repo root)
```

- `tectonic` fetches the LaTeX packages it needs on first run.
- `latexmk -pdf main.tex` also works.

## Layout

- `main.tex` — document class + layout; the only venue-specific file.
  Swap the `\documentclass`/`geometry` block for the venue template.
- `macros.tex` — packages and macros (`\todo`, `\placeholder`,
  `\resulttable`, `\plr`, `\cvar`).
- `sections/*.tex` — intro, related, benchmark, methods, protocol, results,
  discussion.
- `appendix.tex` — pilot (v8) numbers, level rendering, hyperparameters,
  compute.
- `refs.bib` — only verified entries (arXiv id or venue checked).

## Generated tables

`tables/{main,envelopes,compute}.tex` are written by
`src/scripts/paper_tables.py` (`docs/ued-accel.md#paper-tables`);
`tables/sim2sim.tex` has no generator yet. `tables/` is gitignored.

```sh
PYTHONPATH=src uv run python src/scripts/paper_tables.py \
  --eval_dir <evals> --prefix wojtek=v10_wojtek go1=v10_go1 \
  --train_summaries summaries.json --out paper/tables
```

- Each file holds a **bare `tabular`**; the paper owns the `table` float,
  caption and label (`tab:main`, `tab:envelopes`, `tab:compute`,
  `tab:sim2sim`).
- A table wider than the text is shrunk to `\linewidth`.
- A missing file renders a placeholder box, so the paper builds before
  results exist.

## Status

- v10 results, figures and the conclusion are TODO; every open item is a
  red `[TODO: ...]` in the PDF (`rg -n '\\todo' paper`).
- Pilot (v8) numbers in the appendix come from the v8 final eval.
