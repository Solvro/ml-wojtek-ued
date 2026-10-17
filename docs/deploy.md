# Deploying a policy to the robot

Train with `brax_ued.py --profile deploy` (upstream `terrain_blind_v4_4d`)
or `--profile robot` (the policy the robot runs), export with `just
export`, run it with w01-tek's ROS stack — in the MuJoCo sim first.

## Current robot policy {#current-policy}

- Policy: `experiments/deploy-accel-ft` arm F, seed 1
  ([report](../experiments/deploy-accel-ft/report.md)).
  - ACCEL: `--mode accel --score learnability --generator mixed --mutation
    grow`, `--learning_rate 1e-4`, 300M steps.
  - Fine-tune of the imported upstream `terrain_blind_v4_4d` 1.5B base
    (`--init_from`, [warm start](#warm-start)), deploy profile, warp, tiled
    deploy arena.
- Export dir: `dft2_deploy_accel_sfl_lr1e-4_seed1_deploy/`
  - local (gitignored, in the `jgilewicz/accel-ft` worktree):
    `artifacts/deploy_accel_ft/dft2_deploy_accel_sfl_lr1e-4_seed1_deploy/`
    (checkpoint `.pkl` + sidecar beside it)
  - WCSS: `$PDW/artifacts/deploy_accel_ft/`
- Native deploy eval, d1.0, 3-seed pool, %, 95% CI (bootstrap over seeds x
  levels; base: levels), difference paired by level:

  | metric | base 1.5B | F | F − base |
  |---|---|---|---|
  | cmd success | 41.6 [40.7, 42.5] | 47.9 | +6.3 [+5.7, +6.8] |
  | success | 32.1 [31.1, 33.1] | 38.8 | +6.7 [+5.9, +7.4] |

  - every seed beats the base; flat −0.7, d0.25 +0.4, d1.0 fall rate −4.9
    points (guards passed).
- Contract: `policy_meta.json` equals the base export's (and upstream's)
  except `run_name` / `checkpoint` — CONSUMED keys, network and runtime
  unchanged. Export checks: numpy vs brax inference 1.35e-5, robot runtime
  vs env reference 2.57e-4.
- Attribution: the DR control at the same lr (arm E, `--mode dr
  --learning_rate 1e-4`) ties F (47.9 / 38.8). The gain comes from the lower
  fine-tune lr plus hard training levels, not from ACCEL alone; lr 3e-4
  (arms A, C) degrades the base.
- Run it: [Running it](#run).

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

`brax_ued.py --profile {benchmark,deploy,robot}`, default `benchmark` (the
v10 setup, unchanged); `robot`: [below](#robot-profile).

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
- Arena `accel_deploy`: 12x12 m tiles (`patch_radius` 6.0, 4 m gaps, 16x16
  tiles; [tiled arena](environment.md#tiled-arena)), `coarse_n` 25 (keeps
  the ~0.5 m feature size), `cell_size` 0.05 (241x241 grid); the rest as
  benchmark. vx up to 1.2 m/s leaves the 6x6 m v10 patch in ~2.5 s of a 6 s
  level. Build: `just build-deploy-arena` (its own files,
  `*_accel_deploy*`; v10's `accel` files stay). `--robot go1` is refused.
- Deviations from upstream, all TRAINING_ONLY (`tests/deploy` checks the
  rest is identical):
  - `terrain`: upstream's tile-curriculum block (flat row, feature spawns,
    type caps, `pad_radius` 0.3) is replaced by our accel arena block
    (`enable`, `arena: accel_deploy`, `pad_radius` 0.55);
  - `sim.num_envs` (set per run); backend (`auto`: warp on a GPU) and warp
    buffers are upstream's, since every env shares the tiled arena;
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
  rebuild the same env. `--clipping_epsilon` (default 0.2) sets PPO's clip,
  `--learning_rate` (default 3e-4) Adam's step size.
  `tests/scripts/test_env_override.py` keeps every
  `experiments/deploy-ab/arms/*.json` at upstream's CONSUMED keys and
  network.
- Parity test: `tests/deploy/test_upstream_parity.py` compares each robot
  profile's (`deploy`, [`robot`](#robot-profile)) resolved env config (every
  CONSUMED key, every TRAINING_ONLY key but `terrain`/`sim`), `dr` and
  network against `tests/deploy/upstream_<profile>.json`, a snapshot of the
  profile's upstream `train.py` overrides (`brax_ued.UPSTREAM_RUNS`) taken
  with upstream's own code; `test_export_policy.py` checks the exported
  contract equals upstream's, and that w01-tek's `policy.py` (when a
  checkout is at `../w01-tek` or `$W01_TEK`) gives the vendored runtime's
  targets and `tau_ff` bit for bit. Regenerate after an upstream bump
  (read-only on w01-tek):

  ```sh
  uv run python -m deploy.upstream_snapshot --w01-tek ../w01-tek --profile deploy
  uv run python -m deploy.upstream_snapshot --w01-tek ../w01-tek --profile robot
  ```

## `--profile robot` {#robot-profile}

The contract of the policy the robot's RPi loads:
`wojtek_flat_quiet_baseacc_s0` (the drop-in
`ros/deploy/rpi/wojtek-robot-local.conf` launches `policy:=/home/rpi/policy`;
its comment names that run). Not the policy `policy_source.py` pins
(`wojtek-quiet-locomotion@553795b` = `wojtek_flat_quiet_v10_s0`, kp 20,
w01-tek c843245): the drop-in's explicit `policy:=` overrides the pin, so
the robot team has to make sure the policy they copy to `/home/rpi/policy`
is the one loaded.

- Env: `utils/env_overrides.robot_overrides()` = the resolved `task.env` of
  upstream `train.py` with `brax_ued.UPSTREAM_RUNS["robot"]` @ `e44c1a9`:
  `+experiment=flat_quiet_v6 ++task.env.pd_kp=40.0 ++task.env.pd_kd=0.8
  ++task.env.reward.scales.base_accel=-0.02
  ++task.env.reward.base_accel_wz_fade=0.1
  ++task.env.reward.base_accel_wz_floor=0.35`. `dr` = `flat_quiet_v6`'s,
  the same block as deploy's. Network as deploy (upstream `build_ppo_params`).
- Upstream trains it with RND (`rnd.enable`, the vendored `ppo_rnd`
  trainer); brax_ued has no RND, so fine-tunes run without it
  (training-only).
- No preset records `baseacc_s0`'s recipe. Reconstruction (coordinator /
  user decision, 2026-10-16): `flat_quiet_v6` at the robot's servo plant,
  plus the `base_accel` dose w01-tek 6ff736c measured "on the kp40-sticky
  keeper"; upstream's run was a fine-tune of that keeper (restore chain
  unknown), ours trains from scratch for the preset's 2B.
- CONSUMED keys of the contract:

  | key | value | source |
  |---|---|---|
  | `pd_kp` / `pd_kd` / `max_torque` | 40 / 0.8 / 12 | established: w01-tek 833b9ca (`origin/michal/baseacc-default-policy`) |
  | `tau_ff` | enable, scale 3.0 (24 actions) | established: 833b9ca ("tau_ff +-3") |
  | `action_filter` | 0.0 | established: 833b9ca ("no action filter") |
  | `obs` (actor) | `joint_pos`, `joint_vel`, `last_act`, `command` (40) | established: 833b9ca ("IMU-blind obs") |
  | `command` | 4-D: vx [-0.8, 1.2], vy [-0.5, 0.5], wz [-1, 1], height 0.125 fixed | 4-D established (833b9ca); box inferred from the flat_quiet family |
  | `action_scale` | 0.25 on every joint | inferred: flat_quiet family; `flat_grace_v1` (3616f81) = `flat_quiet_v6` + kp/kd "at the deployed keeper's servo contract" |
  | `abduction_ctrl_limit` / `knee_target_max` | 0.44 / 3.15 | inferred: flat_quiet family, `flat_probe_v1` (kp 40 / kd 0.8 keeper plant) |
  | `ctrl_dt` / `real_pose_ref` | 0.02 / false | inferred: flat_quiet family (env defaults) |

  - the inferred rows have to be checked against the deployed
    `/home/rpi/policy/policy_meta.json` before an export is loaded on the
    robot (nobody here can read it)
- Deviations from upstream, all TRAINING_ONLY (`tests/deploy` checks the
  rest): `terrain` = the `accel_deploy` tiled arena block (upstream trains
  on the flat floor); `sim` = upstream's backend with
  `terrain_blind_v4_4d`'s contact buffers (`naconmax_per_env` 116, `njmax`
  1024; flat upstream: 32 / 320); our env patches and training loop as
  deploy. Arena, command speeds and `just build-deploy-arena` as deploy.
- v10-comparable eval: `--command_box 0.2 0.6 -0.2 0.2 -0.4 0.4 0.125`
  (the height is a policy input, trained at 0.125 only).
- experiments/robot-ft: an upstream-trained 2B base on this profile, then
  T8's ACCEL SFL lr 1e-4 recipe and DR lr 1e-4 (300M, 3 seeds each): no arm
  beats the base, both lose flat success; no export
  ([report](../experiments/robot-ft/report.md)). `experiments/robot-ft/
  to_upstream_run.py` turns a robot-profile `.pkl` into a w01-tek run dir
  for upstream's battery (`run.sh report`).

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
  <name>.pkl --mode accel --name <name> --profile {deploy,robot} [brax_ued
  flags]`
  (`scripts/import_upstream.py`): a w01-tek `train.py` checkpoint (brax
  orbax + `ppo_network_config.json`, `run.json` two levels up) becomes a
  brax_ued `.pkl` (normalizer, policy, value) + sidecar for the given
  flags, with `imported_from` (checkpoint, run name, env steps).
- The profile has to be the one of the checkpoint's upstream run
  (`UPSTREAM_RUNS`). Refused before writing: any CONSUMED env key differing
  from the run's `run.json`, actor or critic obs size, action size, network kwargs,
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
- Fine-tuning the 1.5B upstream base (experiments/deploy-accel-ft, 300M
  steps, warp): lr 3e-4 degrades it; `--learning_rate 1e-4 --mode accel
  --score learnability --generator mixed --mutation grow` beats it on
  d1.0 success (+6.7 points) and command-conditioned success (+6.3), as
  does DR at lr 1e-4; the plain easy-generator ACCEL buffer stays next to
  flat and does not ([current policy](#current-policy)).

## Running it {#run}

Read from w01-tek `e44c1a9` (read-only reference): `ros/README.md`,
`ros/deploy.sh`, `ros/docker/compose.yaml`,
`ros/deploy/rpi/wojtek-robot-local.conf`, the bringup / sim launch files,
`wojtek_policy/policy_source.py`. `policy:=` takes a local directory holding
`policy.npz` + `policy_meta.json`, or a Hugging Face `org/name[@rev]`
answered from the policy store.

1. Sim: make the export dir visible to the dev container, which mounts only
   `ros/policies` (→ `/ros2_ws/policies`, `WOJTEK_POLICY_STORE`):

   ```sh
   cp -r <export dir> <w01-tek>/ros/policies/dft2_deploy_accel_sfl_lr1e-4_seed1_deploy
   ```

   - `./deploy.sh` mirrors `ros/policies/` to the RPi's
     `~/wojtek_ws/policies/` with `--delete` (and skips the sync only when
     `ros/policies` is absent), so creating it on a fresh clone prunes the
     robot's store down to what this PC holds; the pin is re-resolved
     first only when `HF_ORGANIZATION` is set
2. Sim first, in the container (`./dev.sh` in `ros/`):

   ```sh
   ros2 launch wojtek_pc sim.launch.py policy:=/ros2_ws/policies/dft2_deploy_accel_sfl_lr1e-4_seed1_deploy
   ```

   - the plant (MuJoCo), ros2_control, `real_io_node` and `policy_node`; the
     console reads its command box from the same contract
   - `./sim.sh policy:=<dir>` from the host does the same (every
     `name:=value` passes through); `hw:=mock` runs the node graph only
   - the launch prints `>> policy dft2_deploy_accel_sfl_lr1e-4_seed1 from
     local:<dir>; servo settings kp=40 kd=1.6 max_torque=32`
3. Robot: upstream runs the RPi stack only through `wojtek-robot.service`
   (RT-pinned; "there's no non-RT way to run it" apart from the `--dry-run`
   bench). The drop-in `deploy.sh --provision` installs
   (`deploy/rpi/wojtek-robot-local.conf`) launches `robot.launch.py
   boot_pose:=folded ... policy:=/home/rpi/policy`, and that file documents
   deploying a local policy by hand:

   ```sh
   rsync -a <export dir>/ rpi@10.42.0.2:policies/dft2_deploy_accel_sfl_lr1e-4_seed1_deploy/
   ssh rpi@10.42.0.2 'ln -sfn /home/rpi/policies/dft2_deploy_accel_sfl_lr1e-4_seed1_deploy /home/rpi/policy'
   ```

   - then, robot lying folded: `ros2 run wojtek_bringup robot` from
     `./dev.sh` starts the service for the session (`systemctl start`,
     stopped on Ctrl-C); if the service already runs (boot autostart),
     `ssh rpi@10.42.0.2 'sudo systemctl restart wojtek-robot.service'`
     instead — `start` leaves a running stack on the old policy
   - servo kp / kd / max_torque come from the contract's `pd` block (kp 40,
     kd 1.6, max_torque 32; `tau_ff` off, so the drive limit stays 32)
   - `deploy.sh --policy` takes a Hugging Face reference only (it rejects a
     local directory); it writes `~/wojtek_ws/policy_override`, which only
     sets the launch default, so the drop-in's explicit `policy:=` wins
     (read from `policy_source.active_policy`, not stated upstream).
     `ros2 run wojtek_bringup robot --policy` does not reach the service
   - `use_imu` stays `true` (default): `policy_node` holds (no targets)
     while an IMU-observing policy's IMU data is stale; `use_imu:=false` is
     bench only. Upstream's `terrain_blind_v4_4d.yaml` says the CANdle-hat
     BMI has to be wired into the policy node; the real xacro now wires an
     I2C IMU (Adafruit 5543, since 2026-07-17) by default. Confirm
     `/imu_sensor_broadcaster/imu` publishes before arming

### First robot tests {#first-tests}

- Arming is a person's decision at the robot, never a script's: the stack
  comes up DISARMED, and arming is when torque reaches the motors.
  - pose the robot folded before the service starts (`boot_pose:=folded`
    sets the encoder offsets at start), or re-zero
  - `ros2 service call /wojtek/zero std_srvs/srv/Trigger`, then
    `/wojtek/stand_up` (Trigger), then
    `/wojtek/arm std_srvs/srv/SetBool "{data: true}"`
  - wind down: disarm, then `/wojtek/lie_down` (Trigger)
- Bench first, no torque: `ros2 run wojtek_bringup robot --dry-run --policy
  /home/rpi/policies/dft2_deploy_accel_sfl_lr1e-4_seed1_deploy` launches
  `robot.launch.py dry_run:=true` on the RPi without RT (drives never
  enabled, encoder states still stream, the policy runs). Stop the service
  first (inferred: both open the drives).
- Reduced torque next: `max_torque:=2` is upstream's example override "for
  cautious first tests". `ros2 run wojtek_bringup robot` forwards
  `name:=value` only with `--dry-run` / `--sim`; on the service it has to
  be added to the drop-in's `ExecStart` (inferred, not documented
  upstream; `deploy.sh --provision` reinstalls the drop-in).

## WCSS

- `PROFILE=deploy sbatch slurm/brax_ued.sbatch` (with `MODES`, `NSEEDS`,
  `PREFIX` as usual); the frozen copy needs `just build-deploy-arena` run
  inside it.
- Export on any machine with the repo: copy `<run>.pkl` + `<run>.json`.
