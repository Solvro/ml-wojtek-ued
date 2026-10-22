# Robots

The benchmark's four MuJoCo Playground quadrupeds (open serial chains) next
to Wojtek (closed-chain legs, [environment.md](environment.md)). Select with
`robot=<name>` in every entry point with a `robot` group, `ROBOT=<name>` in
`slurm/`; sizes and the robot list: [overview.md](overview.md#robots).

| `robot=` | class | model | `SCALE` | standing trunk (m) | DR |
|---|---|---|---|---|---|
| `go1` | `Go1Terrain` | Playground `go1_mjx_feetonly.xml` | 2.5 | 0.258 | Playground `go1/randomize.py` |
| `barkour` | `BarkourTerrain` | menagerie `google_barkour_vb/barkour_vb_mjx.xml` | 2.5 | 0.253 | none (Playground has none) |
| `spot` | `SpotTerrain` | Playground `spot_mjx_feetonly.xml` | 4.3 | 0.433 | none (Playground has none) |
| `go2` | `Go2Terrain` | menagerie `unitree_go2/go2_mjx.xml`, rewritten feet-only | 2.7 | 0.270 | Go1's `randomize.py` |

Shared (`robots/terrain.py` `TerrainRobot`, a mixin over the Playground env;
a robot file keeps its constants, scene and what differs):

- Playground's own reward and obs, each robot's own PD gains (table
  below), feet-only collisions, blind actor.
- jax backend only (`playground_backend`): `backend=auto` resolves to jax,
  `backend=warp` is refused because foot contacts with the tiled arena's
  hfield diverge on warp far from the origin.
- The tiled ACCEL arena written by `arena_scene` (no asset files): one
  `hfield` geom `floor` (geom 0, so a friction DR keyed on geom 0 hits it),
  sized by `HFieldConfig`; `reset` spawns on `spawn_xy`, the tile centre
  `FunctionalEnv` sets.
- Commands: the forward slice of each robot's own Playground command box
  (`native_command_range`): vx uniform from 0 to the robot's max, vy and
  wz uniform over their native ranges, no standing commands;
  `sample_command` redraws from the same box on the env's own schedule.

  | robot | vx (m/s) | vy (m/s) | wz (rad/s) | source |
  |---|---|---|---|---|
  | Go1, Go2 | 0-1.5 | +-0.8 | +-1.2 | `go1/joystick.py` `command_config.a` |
  | Barkour | 0-1.5 | +-0.8 | +-0.7 | `barkour/joystick.py` `lin_vel_x/y`, `ang_vel_yaw` |
  | Spot | 0-1.0 | +-0.8 | +-1.0 | `spot/joystick.py` `command_config` |

- Eval commands (`make_env(..., eval_travel=)`, used by brax_ued's eval,
  eval_brax and eval_sim2sim): vx only from `EVAL_SPEED_MARGIN` (1.2) x
  threshold / episode time up to the max, so every eval command can reach
  the success threshold (below) in the robot's level: Go1, Barkour 0.5-1.5,
  Go2 0.54-1.5, Spot 0.33-1.0 m/s at the benchmark's `min_travel=1.0`,
  `level_steps=300`.
  Training draws the full forward slice, so train success tops out below 1.
- Success threshold: `robots.scale.min_travel(cfg.robot, cfg.min_travel)`
  = `min_travel` x the robot's size, the one source for training success,
  the ACCEL / SFL / PLR scores, eval, sim2sim, `domination` and `paper`
  (both rebuild success from raw travel) and the export README.
- Level length: `robots.scale.level_steps(cfg.robot, cfg.level_steps)` =
  `level_steps` x the robot's `EPISODE_SCALE` control steps, the one source
  for training (unrolls per iteration, env-step count), eval, sim2sim and
  the export README; `level_steps` is in Wojtek's steps. `EPISODE_SCALE` is
  1 except Spot's 2.6 (780 steps, 15.6 s): Wojtek's top vx per unit size
  (0.6 m/s / 0.1 m) over Spot's (1.0 m/s / 0.43 m), rounded to 0.1, so the
  4.3 m threshold is the same share of the distance at top speed as for
  Wojtek, Go1 and Barkour. `robot.scale` does not change it.

  | robot | threshold (m) | steps | / distance at eval vx floor | / distance at vx max |
  |---|---|---|---|---|
  | Wojtek | 1.0 | 300 | 0.83 | 0.28 |
  | Go1 | 2.5 | 300 | 0.83 | 0.28 |
  | Go2 | 2.7 | 300 | 0.83 | 0.30 |
  | Barkour | 2.5 | 300 | 0.83 | 0.28 |
  | Spot | 4.3 | 780 | 0.83 | 0.28 |

  Spot redraws its command every 200 steps (Barkour 500, Go1 / Go2 every
  5 s on average); with turning, perfect tracking of eval commands reaches
  the threshold in 62 % of Spot's levels, 70 % of Go1's and Go2's, 95 %
  of Barkour's.
- `forward_vel` (train and eval logs) is measured along the commands' x
  axis: `FunctionalEnv.forward_velocity`, for these robots the
  `local_linvel` velocimeter the reward tracks (`command_frame_velocity`).
  Barkour's velocimeter site `imu_frame` faces the torso's -x, so the root
  body frame would log its forward walking as negative.
- Wojtek's fall rule at the robot's size: trunk below 0.6 x `STANDING_HEIGHT`
  above the terrain under it (`terrain_height`, read off `hfield_data`), or
  gravity z > -0.4 in the trunk frame.
- The critic gets a clean 5x5 height scan (Wojtek's grid x `SCALE`) appended
  to `privileged_state`.
- `SCALE` = standing trunk height / Wojtek's 0.10 m, rounded to 0.1;
  `robot.scale` overrides it (below). The
  trunk height is measured: the settled `qpos[2]` over flat ground under zero
  action (the PD pose), which sags below the kinematic height.
  `tests/robots/test_terrain_robots.py` re-measures it per robot. `SCALE`
  scales the arena (`hfield_config`, `reach` included), level lengths and
  `min_travel`; slopes do not scale.
- Needs `mujoco_menagerie` under Playground's `external_deps` (meshes);
  `ensure_menagerie_exists()` clones it at the pinned commit
  ([setup.md](setup.md#build-generated-assets)); on a cluster whose login node cannot
  import mujoco, clone it on a compute node or by hand.
- `eval_sim2sim`'s `full_collision` variant exists for Go1 and Spot only
  (`FULL_COLLISION`); Barkour has no full-collision model and Go2's is not
  wired, so those two refuse it with an error.

## Overrides: gains and scale

`RobotConfig` (`config/schema.py`) has four fields that default to `null`
(= the robot's own value), so `conf/robot/<name>.yaml` sets only `name`; a
number replaces the robot's own value (`robot.kp=50 robot.scale=2.0`). `kp`
must be > 0, `kd` and `damping` >= 0:

| field | sets | Go1 | Go2 | Barkour | Spot |
|---|---|---|---|---|---|
| `kp` | actuator position gain (`gainprm[0]`, `biasprm[1] = -kp`) | 35 | 100 | 50 | 300 |
| `kd` | actuator velocity gain kv (`biasprm[2] = -kd`) | 0 | 0.707 | 0.5 | 20 |
| `damping` | passive damping of the 12 leg joints (`dof_damping[6:]`; Playground's `Kd` for Go1 / Spot) | 0.5 | 2.828 | 0.024 | 1 |
| `scale` | size relative to Wojtek (`SCALE`) | 2.5 | 2.7 | 2.5 | 4.3 |

- Gains: `PDGains` (`robots/terrain.py`) edits the compiled `MjModel`
  after the robot's own setup and rebuilds the MJX model; all-null leaves
  the model untouched. Exported `kd` (`export/contract.py`) = `kd` +
  `damping`.
- Scale: `robots.scale.robot_scale(cfg.robot)` is the one value read by
  `hfield_config` (arena lengths, `reach`), `min_travel`, the param level
  space (`robot_scale`), the eval suite, the critic's height scan and the
  travel bounds; `make_env` passes it to the robot. Pass the same
  overrides to train, eval, sim2sim, render and export of a run
  (`ROBOT_ARGS` in `slurm/`). Fall rule, spawn height and the level length
  (`EPISODE_SCALE`) do not scale (they are the robot's own).
- Wojtek refuses every non-null field.
- Defaults: Go1 and Spot keep Playground's gains, Barkour its menagerie
  model's (Playground's env would set kp 35, damping 0.5239), Go2 its own
  `GAINS` (`robots/go2.py`, below).
- Record: training writes `<name>.robot.yaml` beside the checkpoint (`name`,
  the compiled `kp` / `kd` / `damping`, `scale`, `min_travel_m`,
  `level_steps` (resolved);
  `robots/record.py`). eval and sim2sim rebuild it from their own config
  and refuse a mismatch, naming the fields; without the file they warn and
  run.

## Per robot

- **Go1**: Playground `Go1JoystickRoughTerrain`, spawning 0.35 m above the
  tile.
- **Barkour vB**: Playground `barkour.Joystick` (18 dof, 12 joints
  clipped to its limits, perturbation kicks always on). Its env builds from a
  fixed menagerie scene and returns one obs vector (a 15-step history, 465
  values), so `BarkourTerrain` builds the arena scene itself, adds the foot
  contact sensors in XML (Playground adds them with `MjSpec`), uses the
  vector as the actor obs and `[obs, height scan]` as the critic obs, and
  carries a copy of `step` whose termination is the shared fall rule plus
  Playground's joint-limit check: its own `torso z < 0.18` is a world height,
  wrong on a pit or raised tile. Commands are redrawn every 500 steps.
- **Spot**: Playground `spot.Joystick` (`Kp` 300, `Kd` 1, 3-step joint-error
  history, feet position obs). `get_gravity` there is the up vector, so the
  fall rule negates it. Playground's `early_termination` flag and `gravity z <
  0.85` termination are replaced by the shared rule. Spot's standing trunk is
  0.433 m, scale 4.3, `reach` 43 m. Commands are redrawn every 200 steps;
  levels are 780 steps (`EPISODE_SCALE` 2.6, above).
- **Go2**: Playground has no Go2 env: `Go2Terrain` is `Go1Terrain` (Go1's
  joystick reward and obs) on the menagerie model with `GAINS` kp 100,
  kv 0.707, joint damping 2.828: menagerie's kp 50 doubled, kv and damping
  x sqrt(2), so the damping ratio stays menagerie's. The trunk stands at
  0.270 m (Go1's `Kp` 35 sags it to 0.219 m).
  `go2_mjx.xml` collides every link and lacks the names Go1's env reads, so
  `feet_only_model` rewrites it (each substitution must match exactly once):
  trunk `base` -> `trunk`, foot sites `FL_foot` -> `FL`, only the four foot
  spheres collide (no hfield-sphere `margin` in MJX), the `upvector`,
  `local_linvel`, per-foot `*_global_linvel` and `*_floor_found` sensors are
  added, the model's own keyframe is dropped. Go1's randomization fits
  (hfield = geom 0, trunk = body 1, 12 joints = dofs 6:).

