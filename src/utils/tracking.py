import numpy as np

# command-conditioned success: no fall and the command tracked, so stand,
# spin and slow commands (|v_xy| x level time < min_travel) can succeed.
# Relative error with the deploy reward's floors (tracking_rel_floor_lin /
# _ang): an absolute 0.25 m/s, 0.5 rad/s let a stand-still robot pass 26.9%
# of deploy command draws, this lets it pass 12.4%. The linear floor scales
# with the robot like min_travel. Steps within TRACK_SKIP_S of the reset or
# a command change are a start-up transient and not scored.
TRACK_REL_TOL = 0.5
TRACK_FLOOR_LIN = 0.3
TRACK_FLOOR_YAW = 0.7
TRACK_SKIP_S = 1.0

Episodes = dict[str, np.ndarray]


def settled_steps(cmd: np.ndarray, skip: int) -> np.ndarray:
    """(envs, steps): at least `skip` steps since the reset or the last
    command change. cmd: (envs, steps, 3)."""
    steps = np.arange(cmd.shape[1])
    changed = np.any(cmd[:, 1:] != cmd[:, :-1], axis=-1)
    last_change = np.maximum.accumulate(np.where(changed, steps[1:], 0), axis=1)
    last_change = np.concatenate([np.zeros((len(cmd), 1), int), last_change], 1)
    return steps - last_change >= skip


def settled_mean(x: np.ndarray, settled: np.ndarray) -> np.ndarray:
    count = settled.sum(axis=1)
    # no settled step: the episode cannot be judged, nan is never tracked
    total = np.where(settled, x, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def tracking_errors(
    cmd: np.ndarray, vel: np.ndarray, skip: int, lin_floor: float
) -> Episodes:
    """Per env, means over `settled_steps`: absolute (`lin_err` m/s,
    `yaw_err` rad/s) and relative to max(|cmd|, floor) (`lin_rel`,
    `yaw_rel`). cmd, vel: (envs, steps, 3) as (vx, vy, wz), body frame."""
    err = cmd - vel
    lin = np.linalg.norm(err[..., :2], axis=-1)
    yaw = np.abs(err[..., 2])
    lin_ref = np.maximum(np.linalg.norm(cmd[..., :2], axis=-1), lin_floor)
    yaw_ref = np.maximum(np.abs(cmd[..., 2]), TRACK_FLOOR_YAW)
    settled = settled_steps(cmd, skip)
    return {
        "lin_err": settled_mean(lin, settled),
        "yaw_err": settled_mean(yaw, settled),
        "lin_rel": settled_mean(lin / lin_ref, settled),
        "yaw_rel": settled_mean(yaw / yaw_ref, settled),
    }


def tracked(episodes: Episodes) -> np.ndarray:
    return (
        (episodes["fell"] == 0)
        & (episodes["lin_rel"] < TRACK_REL_TOL)
        & (episodes["yaw_rel"] < TRACK_REL_TOL)
    )
