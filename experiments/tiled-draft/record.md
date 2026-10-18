# tiled-draft

- Question: do DR and ACCEL on the shared tiled arena (warp backend), with
  the paper-draft code (PR #32), reach the v10 paper results for Wojtek and
  Go1?
- Code: branch `jgilewicz/draft-arena`, frozen copy `$PDW/copies/td-<sha>`
  (`SRC`/`COPY`), assets built by `setup.sbatch`.
- Phase 0, sanity (`sanity.sbatch`): eval_brax defaults on
  `v10_wojtek_accel_seed4` and `v10_go1_accel_seed0`, jax and warp.
  Pass: jax inside the v10 eval's 95% CIs (`random/d1.0`,
  `param/uniform`); warp within a few points of jax, no diverged/leaked
  envs. Fail on Go1 warp: train Go1 on `BACKEND=jax`.
- Phase 1, training: `slurm/brax_ued.sbatch`, `MODES="dr accel"`,
  `SPACE=param`, `PREFIX=td_<robot>`, `TAGS=tiled-draft`, seeds 0-2,
  300M steps, `BACKEND=warp`; final eval `slurm/eval_brax.sbatch` (same
  layout, `TAGS=tiled-draft-eval`).

## Jobs

| job | what | submitted |
|---|---|---|
