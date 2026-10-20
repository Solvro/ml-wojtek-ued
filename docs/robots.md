# Robots

The benchmark's four MuJoCo Playground quadrupeds (open serial chains) next
to Wojtek (closed-chain legs, [environment.md](environment.md)). Select with
`robot=<name>` in every entry point with a `robot` group, `ROBOT=<name>` in
`slurm/`; sizes and the robot list: [overview.md](overview.md#robots).

| `robot=` | class | model | `SCALE` | standing trunk (m) | DR |
|---|---|---|---|---|---|
| `go1` | `Go1Terrain` | Playground `go1_mjx_feetonly.xml` | 2.5 | 0.258 | Playground `go1/randomize.py` |
| `barkour` | `BarkourTerrain` | menagerie `google_barkour_vb/barkour_vb_mjx.xml` | 2.4 | 0.235 | none (Playground has none) |
| `spot` | `SpotTerrain` | Playground `spot_mjx_feetonly.xml` | 4.3 | 0.433 | none (Playground has none) |
| `go2` | `Go2Terrain` | menagerie `unitree_go2/go2_mjx.xml`, rewritten feet-only | 2.5 | 0.254 | Go1's `randomize.py` |

Shared (`robots/terrain.py` `TerrainRobot`, a mixin over the Playground env;
a robot file keeps its constants, scene and what differs):

- Playground's own reward, obs and PD gains (Go2: menagerie's), feet-only
  collisions, blind actor.
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
  the success threshold (`min_travel` x `SCALE` in `level_steps`): Go1, Go2
  0.5-1.5, Barkour 0.48-1.5, Spot 0.86-1.0 m/s at the benchmark's 1 m and
  300 steps. Training draws the full forward slice, so train success tops
  out below 1.
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
- `SCALE` = standing trunk height / Wojtek's 0.10 m, rounded to 0.1. The
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

Per robot:

- **Go1**: Playground `Go1JoystickRoughTerrain`, spawning 0.35 m above the
  tile.
- **Barkour vB**: Playground `barkour.Joystick` (18 dof, `Kp` 35, 12 joints
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
  0.433 m, scale 4.3, `reach` 43 m.
- **Go2**: Playground has no Go2 env: `Go2Terrain` is `Go1Terrain` (Go1's
  joystick reward and obs) on the menagerie model with the model's own PD
  gains (`kp` 50, `kv` 0.5, joint damping 2); Go1's `Kp` 35 sags the 15 kg
  Go2 to 0.219 m.
  `go2_mjx.xml` collides every link and lacks the names Go1's env reads, so
  `feet_only_model` rewrites it (each substitution must match exactly once):
  trunk `base` -> `trunk`, foot sites `FL_foot` -> `FL`, only the four foot
  spheres collide (no hfield-sphere `margin` in MJX), the `upvector`,
  `local_linvel`, per-foot `*_global_linvel` and `*_floor_found` sensors are
  added, the model's own keyframe is dropped. Go1's randomization fits
  (hfield = geom 0, trunk = body 1, 12 joints = dofs 6:).

