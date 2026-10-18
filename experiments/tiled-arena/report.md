# Tiled arena: closing the throughput gap to upstream

Measured at commit 8a52ca2 (main before the return to the draft state);
its job scripts and the deploy-profile rows are in that commit's tree. The
tiled arena itself was then ported onto the draft code unchanged.

Question: where does brax_ued's ~9x throughput gap to w01-tek's trainer
(~9.7k vs 90-117k env steps/s on one H100) come from, and does a shared
tiled arena on the warp backend close it without changing what a lane sees?

## Answer

- **The gap is the physics backend, nothing else.** At 8192 envs, MJX's jax
  backend steps 12.9k env steps/s with or without the per-env heightfield
  swap; warp steps 260-280k on the same terrain (21x). The per-env model
  swap, DR draw, PPO update, regret and level rendering together are <5% of
  an iteration on jax.
- **Tiled arena + warp: ~227k steps/s per training iteration** (benchmark
  profile, 8192 envs), ~23x v10. A 32M-step sanity run took 5 min
  wall-clock including compilation and two evals (v10: ~55 min).
- **Level semantics unchanged**: on the jax backend the tiled arena
  reproduces `v10_wojtek_accel_seed4`'s final eval inside the original 95%
  CIs (`random/d1.0` .414 vs .423 [.412, .435]); the sanity run tracks the
  v10 curve.
- **Warp is not jax**: the same v10 policy scores 1.5-3 points higher on
  terrain under warp, and a standing robot sits 7-11 mm higher. Compare
  runs only within one backend.

## 1. Profile of the old loop (job 6109828, code 4b91bce)

`profile_old_path.py`, H100, 8192 envs (256 lanes x 32), param levels.
`cur_jax` = v10 loop (per-env `hfield_data` + DR model swap, jax);
`shared_*` = one level baked into the model for every env.

| profile | variant | env steps/s (DR off / on) | 20-step unroll (s) | PPO update (s) | eval call, 100 steps (s) | peak GPU (GB) |
|---|---|---|---|---|---|---|
| benchmark | cur_jax | 12.9k / 12.9k | 12.74 | 0.073 | 33.4 | 23.4 |
| benchmark | shared_jax | 12.9k / 12.9k | 12.75 | 0.074 | 33.3 | 22.7 |
| benchmark | shared_warp | 278k / 269k | 0.60 | 0.082 | 2.9 | 9.0 |
| deploy | cur_jax | 12.9k / 12.9k | 12.74 | 0.075 | 33.4 | 24.9 |
| deploy | shared_jax | 12.9k / 12.9k | 12.74 | 0.074 | 33.4 | 22.7 |
| deploy | shared_warp | 267k / 260k | 0.59 | 0.081 | 2.9 | 9.0 |

- Per-env swap vs shared level on jax: identical (<0.5%).
- Per iteration on jax, physics is 98% of the time (15 unrolls x 12.7 s vs
  15 x 0.08 s of updates, ~1 s of DR draw + render + reset).
- Eval (4 difficulties x 1024 envs x 300 steps) took ~106 s on jax, 0.55
  of an iteration: 5% amortized at `eval_every` 10, 27% at deploy-ab's
  `--eval_every 2`.

## 2. New path throughput (8192 envs, H100)

`profile_brax_ued.py` (jobs 6109870 jax, 6110212 warp), 300-step levels.
Env steps/s with DR on; loop (est.) = sum of the timed pieces; `run_levels`
= one timed real iteration after a warm-up one; evaluate = one `evaluate`
call (4 difficulties x 512 levels x 2 episodes); peak = JAX's allocator.

| profile | backend | env steps/s | loop (est.) | `run_levels` | evaluate (s) | JAX peak (GB) |
|---|---|---|---|---|---|---|
| benchmark | jax | 12.9k | 12.7k | 9.7k | 106 | 23.3 |
| benchmark | warp | 263k | 229k | 134k | 17 | 9.5 |
| deploy | warp | 257k | 221k | 147k | 23 | 9.6 |

- Old path (jax, per-env swap) vs new (warp, tiled), same 8192 envs:
  env stepping 12.9k → 263k (benchmark) / 257k (deploy); per iteration
  12.7k → 229k / 221k; evaluate 106 s → 17 / 23 s.
- GPU memory, whole device (wandb system metrics): sanity run 15.6 GB
  (no preallocation); v10 76 GB (JAX's 75% preallocation, not need).

- Training run (job 6110213, benchmark, ACCEL): steady-state iteration
  (2.46M steps, 15 unroll+update pairs) 10.8 s = **~227k steps/s**; an eval
  (4 difficulties, 2 chunks of 256 levels x 2 episodes) ~25 s. Wall clock
  for 32M steps 303 s, of which ~2 min compilation.
- v10 (jax, per-env swap): 9.7k steps/s end-to-end.

## 3. Correctness

- CPU tests (`tests/utils/test_env.py`, `tests/hfield/test_tiles.py`):
  moving a level and its env key to another tile, next to other levels,
  leaves the height scan, base z, tile-relative xy and dones unchanged
  (within float32 noise of the tile offset, ~0.5 mm); a pit tile lowers
  only its own robot by the pit depth with no fall (lookup and physics
  agree); the gap and unused tiles stay flat; level i lands exactly on
  tile i's nodes.
- `check_hfield_batch.py --compare` (16 tiles x 4 envs, flat/pit 5 cm,
  jobs 6109870, 6110158): every tile reads its own level on both backends
  (pit depth jax 0.051 m, warp 0.050 m, no dones). Base z differs by
  backend: warp stands 7-11 mm higher than jax on every tile, flat and
  pit alike, so it is the backend, not the arena.
- `tile_leak_frac` (envs that reached another lane's tile) was 0 in every
  iteration of the sanity run. Gap = `reach - patch_radius`
  (benchmark 5 m, deploy 2 m; `reach` 8 m = 1.3 m/s max command x 6 s).

## 4. v10 re-eval on the tiled arena (job 6109871)

`v10_wojtek_accel_seed4`, `eval_brax.py` defaults (2048 levels x 4
episodes x 2 passes), same seeds as the original final eval.

| metric | original (jax, per-env swap) [95% CI] | tiled, jax | tiled, warp |
|---|---|---|---|
| random/d0.25 success | .909 [.905, .914] | .913 | .915 |
| random/d0.5 success | .837 [.830, .843] | .842 | .857 |
| random/d0.75 success | .645 [.635, .654] | .642 | .665 |
| random/d1.0 success | .423 [.412, .435] | .414 | .446 |
| random/d1.0 CVaR25 | .243 [.219, .262] | .237 | .265 |
| param/uniform success | .412 [.401, .423] | .415 | .439 |
| param/uniform CVaR25 | .236 [.211, .249] | .222 | .248 |
| flat success | .936 [.914, .957] | .936 | .934 |
| suite success | .584 | .583 | .600 |
| runtime | 58 min | 55 min | 8.5 min |

- jax on the tiled arena: 11 of 12 success/CVaR numbers inside the
  original CIs (random/d0.25 CVaR25 .914 vs upper .912).
- warp: higher on every terrain set, flat unchanged.

## 5. Sanity training (job 6110213, `t7_tiled_wojtek_accel_seed0`)

v10_wojtek_accel's flags, seed 0, warp, 32M steps; v10 = the 5 finished
`v10_wojtek_accel_seed*` runs (jax) at the same steps.

| env steps | falls/step new | v10 range | train success new | v10 range |
|---|---|---|---|---|
| 2.5M | .015 | .008-.018 | 0 | 0 |
| 9.8M | .011 | .010-.048 | <.001 | 0 |
| 17.2M | .009 | .008-.014 | .003 | 0-.278 |
| 22.1M | .007 | .006-.012 | .520 | 0-.747 |
| 27.0M | .005 | .006-.009 | .813 | .278-.813 |
| 31.9M | .003 | .004-.006 | .879 | .711-.856 |

- Eval at 27M: d0.25 .817 (v10 .535-.837), d1.0 .219 (v10 .110-.226).
- Inside the v10 band up to 27M, slightly better than all five v10 seeds
  at 31.9M (fewer falls, higher success), in the direction of the warp eval
  shift above.

## 6. Failures on the way

- Jobs 6109872 (JAX preallocating 75%) and 6110159 (preallocation off, 15
  of 80 GB in use): warp CUDA OOM at iteration 2 on a 304 MB EPA scratch
  allocation. mjx's warp FFI captures one CUDA graph per set of buffer
  addresses (`GraphMode.WARP`), each owning its collision scratch, and
  caches 32 per callable; brax_ued's separate jitted calls pass new
  addresses every call. Capping the cache at 4 (`utils/env.py`
  `configure_warp`) with preallocation off completed (job 6110213). That
  preallocation must be off is upstream's finding, not isolated here;
  upstream's presets spell the variable `XLA_PYTHON_CLIENT_PREALLOC`, which
  JAX does not read.

## Caveats

- One seed for the sanity run; 32M steps only.
- The warp/jax gap (policy success, stance height) is measured, not
  explained.
- `run_levels` in the profile (134k) is below the sanity run's steady
  iterations (227k), likely because the profile interleaves other warp
  calls whose new buffers evict the capped CUDA-graph cache (re-capture);
  not measured.
- GPU-h used: ~3.7 of the 6 budgeted (including the two OOM'd runs).
