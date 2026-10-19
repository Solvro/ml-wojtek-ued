import numpy as np

from config.hfield_config import HFieldConfig
from hfield.level_space import make_level_space


def test_grid_mutation_touches_at_most_n_edits_cells() -> None:
    space = make_level_space("grid", HFieldConfig(coarse_n=13), 3, 0.15)
    rng = np.random.default_rng(0)
    level = space.uniform(rng, 1)[0]
    child = space.mutate(level, rng, None)
    assert 0 < np.count_nonzero(child != level) <= 3
    assert np.abs(child - level).max() <= 0.15
