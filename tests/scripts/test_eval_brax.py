import argparse
from typing import Any

import jax
import numpy as np
import pytest

from config.hfield_config import HFieldConfig
from deploy.contract import CONSUMED_KEYS
from scripts import eval_brax
from scripts.brax_ued import parse_args, run_spec
from scripts.eval_brax import eval_spec, rollout, stack_passes, tracking_summary
from scripts.export_policy import build_env


def test_rollout_refuses_levels_shorter_than_the_transient() -> None:
    args = argparse.Namespace(level_steps=50, chunk_envs=8, episodes=2)
    ctx = {"args": args, "dt": 0.02, "spec": {"robot": "wojtek"}}
    with pytest.raises(ValueError, match="--level_steps 50"):
        rollout(ctx, np.zeros((1, 4)), jax.random.key(0))


def episodes(fell: float, lin: float, yaw: float) -> dict[str, np.ndarray]:
    return {
        "fell": np.full((1, 1), fell),
        "travel": np.zeros((1, 1)),
        "lin_err": np.full((1, 1), lin),
        "yaw_err": np.full((1, 1), yaw),
        "lin_rel": np.full((1, 1), lin),
        "yaw_rel": np.full((1, 1), yaw),
    }


def test_tracking_summary_averages_errors_over_standing_episodes() -> None:
    out = tracking_summary("set", [episodes(0, 0.1, 0.2), episodes(1, 9.0, 9.0)])
    assert out["set/cmd_success_rate"] == pytest.approx(0.5)
    assert out["set/track_lin_err"] == pytest.approx(0.1)
    assert out["set/track_yaw_err"] == pytest.approx(0.2)


def test_tracking_summary_without_a_standing_episode_omits_errors() -> None:
    out = tracking_summary("set", [episodes(1, 0.0, 0.0)] * 2)
    assert out == {"set/cmd_success_rate": 0.0}


def test_stack_passes_names_every_key_per_set() -> None:
    raw = stack_passes("random_d1.0", [episodes(0, 0, 0), episodes(1, 0, 0)])
    assert set(raw) == {
        f"random_d1.0_{k}"
        for k in ("fell", "travel", "lin_err", "yaw_err", "lin_rel", "yaw_rel")
    }
    assert raw["random_d1.0_fell"].shape == (2, 1, 1)


def test_suite_reports_cmd_success_per_set_and_their_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rates: list[float] = []

    def fake_two_passes(
        ctx: dict[str, Any], hfields: np.ndarray, key: jax.Array
    ) -> list[dict[str, np.ndarray]]:
        levels = len(hfields)
        tracked_levels = len(rates) % (levels + 1)
        rel = np.where(np.arange(levels) < tracked_levels, 0.0, 1.0)[:, None]
        rates.append(tracked_levels / levels)
        run = {k: v.repeat(levels, axis=0) for k, v in episodes(0, 0, 0).items()}
        run["lin_rel"] = rel
        return [run, run]

    monkeypatch.setattr(eval_brax, "two_passes", fake_two_passes)
    args = argparse.Namespace(
        seed=0,
        num_levels=4,
        suite_instances=4,
        min_travel=1.0,
        alphas=[0.25],
    )
    ctx = {
        "args": args,
        "config": HFieldConfig(),
        "spec": {"robot": "wojtek", "level_space": "grid"},
    }
    metrics, raw = eval_brax.evaluate(ctx)
    suite_rates = rates[len(eval_brax.DIFFICULTIES) :][: len(raw["suite_success"])]
    tags = [
        f"suite/{terrain}_{k}"
        for terrain in eval_brax.SUITE_TERRAINS
        for k in eval_brax.SUITE_STRENGTHS
    ]
    assert [metrics[f"{t}/cmd_success_rate"] for t in tags] == suite_rates
    assert raw["suite_cmd_success"].tolist() == suite_rates
    assert metrics["suite/cmd_success_rate"] == pytest.approx(np.mean(suite_rates))
    assert metrics["suite/success_rate"] == 0.0


V10_BOX = [0.2, 0.6, -0.2, 0.2, -0.4, 0.4, 0.14]


def deploy_spec() -> dict:
    flags = ["--mode", "accel", "--name", "t", "--profile", "deploy"]
    return run_spec(parse_args(flags))


def test_eval_arena_swaps_geometry_only() -> None:
    spec = deploy_spec()
    ours = eval_spec(spec, "benchmark", None)
    assert ours["arena"] == "accel"
    assert ours["env_overrides"]["terrain"]["arena"] == "accel"
    assert ours["hfield"]["patch_radius"] == 3.0
    assert ours["hfield"]["coarse_n"] == 13
    assert spec["arena"] == "accel_deploy", "the checkpoint's spec is untouched"
    assert ours["network"] == spec["network"]


def test_command_box_keeps_every_other_consumed_key() -> None:
    spec = deploy_spec()
    base = build_env(spec)._config.to_dict()
    boxed = build_env(eval_spec(spec, None, V10_BOX))._config.to_dict()
    for key in CONSUMED_KEYS - {"command"}:
        assert boxed[key] == base[key], key


def test_command_box_draws_stay_inside_the_box() -> None:
    env = build_env(eval_spec(deploy_spec(), None, V10_BOX))
    states = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 256))
    cmd = np.asarray(states.info["command"])
    lo = np.array([0.2, -0.2, -0.4, 0.14])
    hi = np.array([0.6, 0.2, 0.4, 0.14])
    assert np.all(cmd >= lo - 1e-6) and np.all(cmd <= hi + 1e-6)
    assert np.all(cmd[:, 0] > 0.0), "no stand or backward draws"


def test_eval_overrides_are_refused_for_go1() -> None:
    spec = run_spec(parse_args(["--mode", "accel", "--name", "t", "--robot", "go1"]))
    with pytest.raises(ValueError, match="Wojtek"):
        eval_spec(spec, "benchmark", None)
