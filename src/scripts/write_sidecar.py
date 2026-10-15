"""Backfill the sidecar `<run>.json` of a `brax_ued.py` checkpoint saved
before `brax_ued.py` wrote one.

Pass the run's own `brax_ued.py` flags after `--checkpoint` (its wandb config
has them). The env those flags build supplies the observation sizes, checked
against the checkpoint's normalizer; `git_commit` is the backfill's, and
`backfilled: true` marks it.
"""

import argparse
from pathlib import Path

from scripts.brax_ued import (
    load_run,
    make_env,
    parse_args,
    run_spec,
    sidecar_path,
    write_sidecar,
)


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="brax_ued.py .pkl")
    known, train_flags = p.parse_known_args(argv)
    checkpoint = Path(known.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")
    if sidecar_path(checkpoint).exists():
        raise FileExistsError(
            f"{sidecar_path(checkpoint)} exists; delete it first to rewrite it"
        )
    spec = {**run_spec(parse_args(train_flags)), "backfilled": True}
    env = make_env(spec, num_envs=1, train=False)
    path = write_sidecar(checkpoint, spec, env.observation_size, env.action_size)
    try:
        load_run(checkpoint)
    except ValueError:
        path.unlink()
        raise
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main()
