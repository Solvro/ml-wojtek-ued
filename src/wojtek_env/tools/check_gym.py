"""Sanity-check the Gymnasium adapter: build, reset, a few random steps,
one full episode to exercise reset-on-done. Mirrors check_model_mjx.py's
gate pattern -- prints PASS/FAIL, exits nonzero on failure."""

import argparse
import sys

import numpy as np

from wojtek_env.gymnasium import WojtekGymEnv


def check_single(task: str, steps: int) -> None:
    episode_length = 50
    env = WojtekGymEnv(task=task, episode_length=episode_length, seed=0)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), "reset obs outside observation_space"
    for _ in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs), "step obs outside observation_space"
        assert np.isfinite(reward), "non-finite reward"
        if terminated or truncated:
            obs, _ = env.reset()

    # Drive one episode to its truncation deterministically (no resets in
    # between), then confirm step() refuses to continue without reset().
    env.reset(seed=0)
    terminated = truncated = False
    for _ in range(episode_length):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    assert terminated or truncated, "episode never ended within episode_length"
    try:
        env.step(env.action_space.sample())
        raise AssertionError("step() after done should have needed reset() first")
    except RuntimeError:
        pass
    print(f"single [{task}]: obs {obs.shape}, action {env.action_space.shape}  ok")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="joystick", choices=["joystick", "getup", "jump"])
    p.add_argument("--steps", type=int, default=60)
    args = p.parse_args()
    try:
        check_single(args.task, args.steps)
    except Exception as e:  # noqa: BLE001
        print(f"GATE FAIL: {e}")
        sys.exit(1)
    print("GATE PASS")


if __name__ == "__main__":
    main()
