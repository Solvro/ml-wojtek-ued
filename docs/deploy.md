# Deploying a policy to the robot

Train with `brax_ued.py --profile deploy`, export with `just export`, run it
with w01-tek's ROS stack — in the MuJoCo sim first.

## Upstream contract {#contract}

- The robot runs w01-tek's numpy runtime
  (`ros/src/wojtek_policy/wojtek_policy/policy.py`, copied verbatim to
  `src/deploy/runtime.py` @ `e44c1a9`). What it evaluates:
  - MLP, SiLU on hidden layers, `tanh` on the first `action_size` outputs
    (= `tanh(loc)` of brax's tanh-normal: the deterministic action);
  - `(obs - norm_mean) / norm_std`, no clipping;
  - actor obs components from `gyro`, `gravity`, `joint_pos`, `joint_vel`,
    `last_act`, `command` only (`phase` is refused); joystick task only;
  - `filt = action_filter * filt + (1 - action_filter) * action`,
    `targets = clip(anchor_ctrl + filt[:12] * action_scale, target_low, target_high)`;
    a live 4th command dim (height) re-anchors with `height_table`;
  - servo kp/kd/max_torque from the contract `pd` block; `cmd_vel` clipped
    to `command_low/high`.
- Files (a directory): `policy.npz` (`norm_mean`, `norm_std`,
  `hidden_{i}_kernel/bias`) + `policy_meta.json` (schema 2, built by
  `src/deploy/contract.py`, a port of upstream `deploy_contract.py`).
- `contract.py` classifies every env-config key as CONSUMED (shapes the
  contract: `ctrl_dt`, `action_scale`, `action_filter`,
  `abduction_ctrl_limit`, `knee_target_max`, `max_torque`, `pd_kp`, `pd_kd`,
  `obs`, `command`, `real_pose_ref`, `tau_ff`) or TRAINING_ONLY; an
  unclassified key refuses the export.

## `--profile deploy` {#profile}

`brax_ued.py --profile {benchmark,deploy}`, default `benchmark` (the v10
setup, unchanged).

- Env: `utils/env_overrides.deploy_overrides()` = the resolved `task.env` of
  upstream `+experiment=terrain_blind_v4_4d` @ `e44c1a9`, verbatim: kp 40 /
  kd 1.6 / max_torque 32, `action_scale` [0.25, 0.5, 1.35], IMU in the actor
  obs, no `phase`, 4-D command vx [-0.8, 1.2] vy [-0.5, 0.5] wz [-1.5, 1.5]
  height [0.10, 0.18], latency + encoder + symmetry on, its reward, push and
  `no_progress` settings. `deploy_dr_config()` = its `dr` block
  (motor_strength, joint_gains, foot_friction). `obs_noise.gyro_bias` /
  `gyro_vib` stay 0.0 as upstream.
- Network: upstream `train.py build_ppo_params` — Playground's Go1
  `network_factory` (policy and value 512-256-128, `state` / `privileged_state`
  keys), brax defaults SiLU and tanh-normal.
- Arena `accel_deploy`: 12x12 m (`patch_radius` 6.0), `coarse_n` 25 (keeps
  the ~0.5 m feature size), `cell_size` 0.05 (241x241 grid); the rest as
  benchmark. vx up to 1.2 m/s leaves the 6x6 m v10 patch in ~2.5 s of a 6 s
  level. Build: `just build-deploy-arena` (its own files,
  `*_accel_deploy*`; v10's `accel` files stay). `--robot go1` is refused.
- Deviations from upstream, all TRAINING_ONLY (`tests/deploy` checks the
  rest is identical):
  - `terrain`: upstream's tile-curriculum block (flat row, feature spawns,
    type caps, `pad_radius` 0.3) is replaced by our accel arena block
    (`enable`, `arena: accel_deploy`, `pad_radius` 0.55);
  - `sim.backend` = `jax` (per-env hfield swaps; warp collides every env
    with env 0's level);
  - our env-side patches: spawn-pose closure fix, fall-cause metrics
    ([environment.md](environment.md#upstream-sync));
  - training loop: ACCEL/DR levels on `FunctionalEnv`, `level_steps` 300
    episodes, brax_ued's PPO loop and hyperparameters (Playground's Go1
    values, as upstream; upstream trains 8192 envs x 1000-step episodes via
    `ppo.train`).
- Training-side variants: `--env_override <file.json>` merges a JSON object
  onto the profile's env overrides; only TRAINING_ONLY top-level keys are
  accepted (`contract.TRAINING_ONLY_KEYS`), anything that shapes the
  contract (`obs`, `command`, `action_scale`, `pd_*`, ...) is refused at
  parse time. The merged config lands in the sidecar, so export and evals
  rebuild the same env. `--clipping_epsilon` (default 0.2) sets PPO's clip.
  `tests/scripts/test_env_override.py` keeps every
  `experiments/deploy-ab/arms/*.json` at upstream's CONSUMED keys and
  network.
- Parity test: `tests/deploy/test_upstream_parity.py` compares the deploy
  profile's resolved env config (every CONSUMED key, every TRAINING_ONLY key
  but `terrain`/`sim`), `dr` and network against
  `tests/deploy/upstream_terrain_blind_v4_4d.json`, a snapshot taken with
  upstream's own code; `test_export_policy.py` checks the exported contract
  equals upstream's. Regenerate after an upstream bump (read-only on
  w01-tek):

  ```sh
  uv run python -m deploy.upstream_snapshot --w01-tek ../w01-tek
  ```

## Sidecar `<run>.json` {#sidecar}

`brax_ued.py` writes it beside `<run>.pkl` at launch (both profiles).
`export_policy.py`, `eval_brax.py`, `eval_sim2sim.py` and `render_video.py`
rebuild env and network from it (`brax_ued.load_run`), no geometry flags.

- Keys: `profile`, `robot`, `task`, `arena`, `hfield` (`HFieldConfig`),
  `level_space`, `env_overrides` (resolved, without `sim.num_envs`),
  `dr_config`, `network` (factory kwargs, `activation` by name), `args`
  (every `brax_ued.py` flag), `obs_size`, `action_size`, `git_commit`,
  `git_dirty`.
- `git_commit` on WCSS frozen copies (no `.git`): read from `GIT_COMMIT`
  beside `src/` (`git rev-parse HEAD > <copy>/GIT_COMMIT`); missing → `null`
  and a warning.
- `load_run` refuses a sidecar whose `obs_size` differs from the checkpoint's
  normalizer.
- Pre-sidecar checkpoints: `just write-sidecar --checkpoint <run>.pkl <the
  run's brax_ued.py flags>` (flags from its wandb config); marks
  `backfilled: true`, `git_commit` = the backfill's.

## Export — `scripts/export_policy.py` {#export}

```sh
just export --checkpoint artifacts/brax_ued/<run>.pkl [--out <dir>]
```

- Default out: `<run>_deploy/` beside the .pkl. CPU (`JAX_PLATFORMS=cpu`).
- Env rebuilt from the sidecar on the jax backend, one env, terrain off (as
  upstream `build_env`: sim and terrain are training-only), so no arena files
  are needed.
- Refuses (before writing) a checkpoint whose actor observes `phase`, whose
  MLP is not SiLU, or whose head is not tanh-normal — every benchmark-profile
  checkpoint — and any non-Wojtek robot.
- Validations, as upstream: numpy forward vs jitted brax inference on 32
  random obs (< 1e-4); `deploy.runtime.WojtekPolicy` on the written files vs
  an env-derived reference over a 32-step closed loop (< 1e-3).

## Upstream checkpoints and warm start {#warm-start}

- `just import-upstream --checkpoint <w01-tek run>/checkpoints/<step> --out
  <name>.pkl --mode accel --name <name> --profile deploy [brax_ued flags]`
  (`scripts/import_upstream.py`): a w01-tek `train.py` checkpoint (brax
  orbax + `ppo_network_config.json`, `run.json` two levels up) becomes a
  brax_ued `.pkl` (normalizer, policy, value) + sidecar for the given
  flags, with `imported_from` (checkpoint, run name, env steps).
- Refused before writing: any CONSUMED env key differing from the run's
  `run.json`, actor or critic obs size, action size, network kwargs,
  policy/value parameter shapes, normalizer keys. A critic mismatch is not
  papered over by re-initialising the value net.
- The import exports like any deploy checkpoint (`just export`); for
  w01-tek's `terrain_blind_v4_4d` run of experiments/deploy-ab the
  `policy.npz` is identical to upstream's own export and `policy_meta.json`
  differs only in `checkpoint`/`run_name`.
- `brax_ued.py --init_from <name>.pkl`: the student starts from that
  checkpoint's normalizer, policy and value (fresh optimizer state);
  refused if its sidecar's network or any parameter shape differs. The
  sidecar records `init_from`.

## Running it {#run}

w01-tek `ros/` (see its `ros/README.md`); `policy:=` takes a local directory
holding `policy.npz` + `policy_meta.json`.

1. Sim first, from `./dev.sh` in w01-tek:
   `ros2 launch wojtek_pc sim.launch.py policy:=<export dir>` — the plant,
   ros2_control, `real_io_node` and `policy_node` on MuJoCo; the console
   reads its command box from the same contract.
2. Robot: `ros2 launch wojtek_bringup robot.launch.py policy:=<dir on the
   RPi>` (or `ros2 run wojtek_bringup robot`). The launch feeds the
   contract's `pd` block (kp 40, kd 1.6, max_torque 32) into the servo
   xacro. The stack comes up disarmed: zero, stand up, arm by hand.
- The actor observes the IMU (`gyro`, `gravity`): the robot's IMU path must
  be live. Upstream's `terrain_blind_v4_4d.yaml` notes the CANdle-hat BMI
  has to be wired into the policy node before this family runs on hardware.

## WCSS

- `PROFILE=deploy sbatch slurm/brax_ued.sbatch` (with `MODES`, `NSEEDS`,
  `PREFIX` as usual); the frozen copy needs `just build-deploy-arena` run
  inside it.
- Export on any machine with the repo: copy `<run>.pkl` + `<run>.json`.
