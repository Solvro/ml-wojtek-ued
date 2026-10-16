"""The heightfield-overflow guardrail in check-terrain.

The warning it looks for is printed from MJWarp device code, so producing a
real one needs a CUDA GPU. Everything here works on the string and the
descriptor plumbing instead: the counter, the dup2 capture, and main()'s exit
path when the count is not zero.
"""

import json
import os
import sys

import pytest

from wojtek_env.tools import check_terrain

WARNING = check_terrain.OVERFLOW_WARNING


def test_counter_finds_every_occurrence():
    log = (
        "put_model 1.2s\n"
        f"Warning: {WARNING} at cell 12\n"
        "step 0\n"
        f"Warning: {WARNING} at cell 13\n"
    )
    assert check_terrain.count_overflow_warnings(log) == 2
    assert check_terrain.count_overflow_warnings("clean run, 200 steps\n") == 0


def test_capture_takes_descriptor_writes_and_puts_them_back(capfd):
    """A write straight to fd 1 bypasses sys.stdout, which is the whole reason
    the capture uses dup2 instead of redirect_stdout."""
    with check_terrain.capture_os_stdout(True) as captured:
        os.write(1, f"Warning: {WARNING}\n".encode())
    assert check_terrain.count_overflow_warnings(captured[0]) == 1
    # Re-emitted, so redirecting hid nothing.
    assert WARNING in capfd.readouterr().out


def test_capture_disabled_leaves_stdout_alone(capfd):
    with check_terrain.capture_os_stdout(False) as captured:
        print("ordinary output")
    assert captured[0] == ""
    assert "ordinary output" in capfd.readouterr().out


def test_main_fails_on_any_warning(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    sentinel = tmp_path / "sentinel"

    def fake_check(args):
        return {
            "status": "ok",
            "backend": "warp",
            "steps_per_s": 100.0,
            "env_steps_per_s": 1e5,
            "contacts": {
                "active_max": 40,
                "active_mean": 12.0,
                "hfield_overflow_warnings": 7,
            },
        }

    monkeypatch.setattr(check_terrain, "run_check", fake_check)
    monkeypatch.setattr(
        sys,
        "argv",
        ["check-terrain", "--out", str(out), "--sentinel", str(sentinel)],
    )
    with pytest.raises(SystemExit) as exc:
        check_terrain.main()
    assert exc.value.code == 1
    report = json.loads(out.read_text())
    assert report["status"] == "fail"
    assert "7 heightfield contact overflow" in report["error"]
    assert sentinel.read_text().startswith("FAIL")


def test_main_passes_when_there_are_no_warnings(tmp_path, monkeypatch):
    out = tmp_path / "report.json"
    sentinel = tmp_path / "sentinel"

    def fake_check(args):
        return {
            "status": "ok",
            "backend": "jax",
            "steps_per_s": 100.0,
            "env_steps_per_s": 1e5,
            "contacts": {
                "active_max": 40,
                "active_mean": 12.0,
                "hfield_overflow_warnings": 0,
            },
        }

    monkeypatch.setattr(check_terrain, "run_check", fake_check)
    monkeypatch.setattr(
        sys,
        "argv",
        ["check-terrain", "--out", str(out), "--sentinel", str(sentinel)],
    )
    with pytest.raises(SystemExit) as exc:
        check_terrain.main()
    assert exc.value.code == 0
    assert sentinel.read_text().strip() == "OK"
