import jax
import jax.numpy as jnp
import numpy as np

from robots.scale import level_steps, min_travel
from training.context import TrainContext

EVAL_DIFFICULTIES = (0.25, 0.5, 0.75, 1.0)


def evaluate(ctx: TrainContext, rng: np.random.Generator) -> dict[str, float]:
    """The deterministic policy on fresh uniform levels per difficulty."""
    cfg, out = ctx.cfg, {}
    per_chunk = min(cfg.eval_levels, ctx.layout.n_tiles)
    threshold = min_travel(cfg.robot, cfg.min_travel)
    for difficulty in EVAL_DIFFICULTIES:
        levels = ctx.space.scaled(rng, cfg.eval_levels, difficulty)
        rendered = jnp.asarray(ctx.space.render(levels))
        chunks = [
            eval_chunk(ctx, rendered[start : start + per_chunk])
            for start in range(0, cfg.eval_levels, per_chunk)
        ]
        fell, travel, forward_vel, cmd_vx = (
            np.concatenate(x) for x in zip(*chunks)
        )
        success = (fell == 0) & (travel >= threshold)
        tag = f"eval/d{difficulty}"
        out[f"{tag}/success_rate"] = float(success.mean())
        out[f"{tag}/fall_rate"] = float(fell.mean())
        out[f"{tag}/mean_travel"] = float(travel.mean())
        out[f"{tag}/forward_vel"] = float(forward_vel.mean())
        out[f"{tag}/cmd_vx"] = float(cmd_vx.mean())
    return out


def eval_chunk(
    ctx: TrainContext, rendered: jax.Array
) -> tuple[np.ndarray, ...]:
    """Two episodes per level, one level per tile: per env fell, travel,
    mean forward and commanded velocity."""
    hfield = ctx.assemble(rendered)
    tile_xy = ctx.layout.tile_xy(rendered.shape[0], 2)
    keys = jax.random.split(ctx.next_key(), tile_xy.shape[0])
    _, state = ctx.eval_reset(keys, hfield, tile_xy)
    start_xy = state.data.qpos[:, :2]
    state, data = ctx.eval_unroll(
        ctx.student.inference_params(),
        state,
        ctx.next_key(),
        hfield,
        None,
        deterministic=True,
        length=level_steps(ctx.cfg.robot, ctx.cfg.level_steps),
    )
    log = data.extras["log"]
    travel = jnp.linalg.norm(log["xy"][:, -1] - start_xy, axis=-1)
    return (
        np.asarray(log["fell"].max(axis=1)),
        np.asarray(travel),
        np.asarray(log["forward_vel"].mean(axis=1)),
        np.asarray(log["cmd_vx"].mean(axis=1)),
    )
