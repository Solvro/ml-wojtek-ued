#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
if [[ "$(uname)" == "Linux" ]]; then export MUJOCO_GL="${MUJOCO_GL:-egl}"; fi
case "${1:-}" in
  build) shift; "$PY" -m wojtek_env.tools.build_model "$@" ;;
  build-terrain) shift; "$PY" -m wojtek_env.tools.build_terrain "$@" ;;
  check) shift; "$PY" -m wojtek_env.tools.check_model_mjx "$@" ;;
  check-terrain) shift; "$PY" -m wojtek_env.tools.check_terrain "$@" ;;
  check-gym) shift; "$PY" -m wojtek_env.tools.check_gym "$@" ;;
  test)  shift; "$PY" -m pytest tests -q "$@" ;;
  *) echo "usage: run.sh {build|build-terrain|check|check-terrain|check-gym|test} [args]"; exit 1 ;;
esac
