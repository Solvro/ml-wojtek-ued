"""Repo-relative paths and robot constants shared by every module."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]  # src/wojtek_env/paths.py -> repo root
MUJOCO_DIR = PROJECT_DIR / "assets/wojtek_description/mujoco"
SOURCE_XML = MUJOCO_DIR / "wojtek.xml"
ROBOT_XML = MUJOCO_DIR / "wojtek_mjx.xml"
SCENE_XML = MUJOCO_DIR / "scene_mjx.xml"

# Procedural terrain arenas (terrain.py -> build_terrain.py). One file set per
# arena kind: `train` is what policies train on, `eval` the fixed measurement
# course, `test` belongs to the test suite. Every set sits next to the robot
# XML, because the scene includes the robot by relative path and MuJoCo
# resolves <hfield file=> against the robot's mesh directory -- an arena
# written to a temporary directory does not compile.
TERRAIN_KINDS = ("train", "eval", "eval_deep", "test")


def terrain_paths(kind: str = "train") -> dict[str, Path]:
    """The four generated files for one arena kind: scene XML, heightfield
    binary, spec JSON, lookup NPZ. `train` keeps the original unsuffixed
    names, so an arena built before the kinds existed still loads."""
    if kind not in TERRAIN_KINDS:
        raise ValueError(
            f"terrain arena kind must be one of {TERRAIN_KINDS}, got {kind!r}"
        )
    tag = "" if kind == "train" else f"_{kind}"
    return {
        "scene": MUJOCO_DIR / f"scene_terrain{tag}.xml",
        "hfield": MUJOCO_DIR / f"terrain{tag}_hfield.bin",
        "spec": MUJOCO_DIR / f"terrain{tag}_spec.json",
        "lookup": MUJOCO_DIR / f"terrain{tag}_lookup.npz",
    }


# XML declaration order. Actuators and joints follow this order.
LEGS = ("rear_left", "rear_right", "front_right", "front_left")
