from dataclasses import dataclass


@dataclass
class HFieldConfig:
    patch_radius: float = 3.0  # m, half-width of the square patch
    cell_size: float = 0.05  # m, fine hfield resolution
    coarse_n: int = 9  # control points per axis, before upsampling
    amplitude: float = 0.12
    difficulty: float = 1.0  # runtime terrain-intensity scale in [0, 1]
    base_z: float = 1.0  # solid base thickness under the hfield, MJCF requires > 0
    pad_radius: float = 0.6  # flat spawn disc at the patch centre
    pad_taper: float = 0.3  # ramp width from flat pad to full terrain
    n_edits: int = 4  # control points touched per mutation
    edit_scale: float = 0.1  # max +/- change to an edited control point

    @property
    def grid_size(self) -> int:
        return round(2 * self.patch_radius / self.cell_size) + 1
