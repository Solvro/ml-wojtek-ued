from pathlib import Path

from config.schema import CONF_DIR
from paper.labels import METHODS, method_label, method_order


def test_every_mode_has_a_paper_label() -> None:
    modes = {path.stem for path in (Path(CONF_DIR) / "mode").glob("*.yaml")}
    assert modes == set(METHODS)


def test_unknown_methods_follow_the_known_ones() -> None:
    assert method_order({"zeta", "accel", "dr"}) == ["dr", "accel", "zeta"]
    assert method_label("my_method") == r"my\_method"
