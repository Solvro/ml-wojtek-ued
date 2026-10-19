import numpy as np

from hfield.level_space import PARAMS, LevelSpace, ParamSpace


def level_composition(
    space: LevelSpace, levels: np.ndarray
) -> dict[str, float]:
    """Mean physical parameter values of the levels the student ran on."""
    if not isinstance(space, ParamSpace):
        return {}
    physical = [space.physical(level) for level in levels]
    out = {}
    for p in PARAMS:
        values = np.array([ph[p.name] for ph in physical])
        out[f"level/{p.name}"] = float(values.mean())
        if p.low < 0.0:
            out[f"level/{p.name}_abs"] = float(np.abs(values).mean())
    distance = np.abs(levels[:, : len(PARAMS)] - space.flat)
    out["level/genome_dist"] = float(distance.mean())
    return out
