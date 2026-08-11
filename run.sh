#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
if [[ "$(uname)" == "Linux" ]]; then export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi
case "${1:-}" in
  build) shift; "$PY" -m wojtek_ued.build_model "$@" ;;
  build-terrain) shift; "$PY" -m wojtek_ued.build_terrain "$@" ;;
  check) shift; "$PY" -m wojtek_ued.check_model_mjx "$@" ;;
  check-terrain) shift; "$PY" -m wojtek_ued.check_terrain "$@" ;;
  train) shift; "$PY" -m wojtek_ued.train "$@" ;;
  smoke) shift; JAX_PLATFORMS=cpu "$PY" -m wojtek_ued.train smoke=true wandb.enable=false "$@" ;;
  test)  shift; "$PY" -m pytest tests -q "$@" ;;
  *) echo "usage: run.sh {build|build-terrain|check|check-terrain|train|smoke|test} [args]"; exit 1 ;;
esac
