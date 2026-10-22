import jax
import numpy as np

from config.schema import BC, Curriculum, Score, TrainConfig
from env.factory import check_wojtek_arena, make_env, tile_reset
from hfield.level_space import make_level_space
from hfield.tiles import TileLayout
from robots.scale import hfield_config, level_steps, min_travel, robot_scale
from training.adversary import Adversary
from training.context import KeyStream, TrainContext
from training.curricula import PAIRED
from training.progress import unvisited
from training.student import Student
from training.terrain_cl import TerrainLanes
from training.unroll import make_unroll


def check_config(cfg: TrainConfig) -> None:
    steps = level_steps(cfg.robot, cfg.level_steps)
    if steps % cfg.ppo.unroll_length:
        raise ValueError(
            f"{steps} steps per level (level_steps={cfg.level_steps}, "
            f"robot={cfg.robot.name}) must be a multiple of "
            f"ppo.unroll_length={cfg.ppo.unroll_length}"
        )
    mode = cfg.mode
    if mode.bc != BC.none and mode.curriculum not in PAIRED:
        raise ValueError(
            f"mode.bc={mode.bc} distils between PAIRED's two students; "
            f"mode.curriculum={mode.curriculum} has one. Use mode=paired_uni, "
            "mode=paired_bi or mode=paired_evo with mode.bc"
        )
    if mode.bc_every < 1:
        raise ValueError(f"mode.bc_every={mode.bc_every} must be >= 1")
    if mode.curriculum == Curriculum.paired_evo and mode.score != Score.pvl:
        raise ValueError(
            f"mode.score={mode.score} does not apply to paired_evo, whose "
            "level score is the PAIRED regret; drop the override"
        )


def build(cfg: TrainConfig) -> TrainContext:
    robot, backend = cfg.robot, cfg.backend
    config = hfield_config(robot, cfg.arena)
    space = make_level_space(
        cfg.level_space.kind,
        config,
        n_edits=cfg.level_space.n_edits,
        edit_scale=cfg.level_space.edit_scale,
        robot_scale=robot_scale(robot),
    )
    check_wojtek_arena(robot.name, config)
    layout = TileLayout(config)
    if cfg.num_lanes > layout.n_tiles:
        raise ValueError(
            f"num_lanes={cfg.num_lanes} exceeds the arena's {layout.n_tiles} "
            "tiles, one level each; rebuild it with a larger arena.tiles"
        )
    num_envs = cfg.num_lanes * cfg.envs_per_lane
    env = make_env(robot, backend, config, num_envs=num_envs, train=True)
    steps = level_steps(robot, cfg.level_steps)
    reach = env.travel.reach(steps, env.travel.sensing)
    layout.check_reach(reach, f"{steps} steps per level")
    eval_envs = 2 * min(cfg.eval_levels, layout.n_tiles)
    eval_env = make_env(
        robot,
        backend,
        config,
        num_envs=eval_envs,
        train=False,
        eval_travel=(min_travel(robot, cfg.min_travel), steps),
    )
    next_key = KeyStream(cfg.seed)
    entropy = cfg.mode.student_entropy_coef
    student = Student(env, next_key(), entropy, cfg.ppo, cfg.network)
    paired = cfg.mode.curriculum in PAIRED
    antagonist = (
        Student(env, next_key(), entropy, cfg.ppo, cfg.network)
        if paired
        else None
    )
    adversary = None
    if cfg.mode.curriculum == Curriculum.paired:
        adversary = Adversary(cfg.mode, space.gen_shape, next_key())
    _, dr_in_axes = env.sample_dr(jax.random.key(0), num_envs)
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None, dr_in_axes, 0)))
    rng = np.random.default_rng(cfg.seed)
    terrain = None
    if cfg.mode.curriculum == Curriculum.terrain_cl:
        terrain = TerrainLanes(space, rng, cfg.num_lanes)

    return TrainContext(
        cfg=cfg,
        env=env,
        student=student,
        antagonist=antagonist,
        adversary=adversary,
        space=space,
        layout=layout,
        assemble=jax.jit(layout.assemble),
        rng=rng,
        next_key=next_key,
        reset=reset,
        eval_reset=tile_reset(eval_env),
        unroll=make_unroll(env, student.make_policy, dr_in_axes),
        eval_unroll=make_unroll(eval_env, student.make_policy, None),
        children=np.zeros((0, *space.shape)),
        children_return=unvisited(0),
        terrain=terrain,
    )
