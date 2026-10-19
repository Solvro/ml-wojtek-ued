from typing import Any, NamedTuple

import jax
import mujoco


class HFieldGeometry(NamedTuple):
    nrow: int
    ncol: int
    base_z: float
    elevation: float

    def heights(self, hfield_data: jax.Array) -> jax.Array:
        grid = hfield_data.reshape(self.nrow, self.ncol)
        return self.base_z + grid * self.elevation


def hfield_geometry(base: Any) -> HFieldGeometry | None:
    """The single hfield whose grid is the terrain lookup, else None."""
    if base._terrain is None:
        return None
    m = base.mj_model
    # mujoco is a C extension that ty cannot introspect
    hfield_type = mujoco.mjtGeom.mjGEOM_HFIELD  # ty: ignore[unresolved-attribute]
    geoms = [i for i in range(m.ngeom) if m.geom_type[i] == hfield_type]
    if m.nhfield != 1 or len(geoms) != 1:
        return None
    geometry = HFieldGeometry(
        nrow=int(m.hfield_nrow[0]),
        ncol=int(m.hfield_ncol[0]),
        base_z=float(m.geom_pos[geoms[0], 2]),
        elevation=float(m.hfield_size[0, 2]),
    )
    if base._terrain.lookup.shape != (geometry.nrow, geometry.ncol):
        return None
    return geometry
