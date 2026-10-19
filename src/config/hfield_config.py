from dataclasses import dataclass


@dataclass
class HFieldConfig:
    patch_radius: float = 3.0  # m, half-width of the square patch
    cell_size: float = 0.05  # m, fine hfield resolution
    coarse_n: int = 9  # control points per axis, before upsampling
    amplitude: float = 0.12
    difficulty: float = 1.0  # runtime terrain-intensity scale in [0, 1]
    base_z: float = (
        1.0  # solid base thickness under the hfield, MJCF requires > 0
    )
    pad_radius: float = 0.6  # flat spawn disc at the patch centre
    pad_taper: float = 0.3  # ramp width from flat pad to full terrain
    tiles: int = 16  # tiles per side of the shared arena, one level each
    # m readable from the tile centre in one level (sets the tile gap):
    # 0.15 jitter + 1.3 m/s x 6 s + 1.48 m scan = 9.43, plus margin
    reach: float = 10.0

    @property
    def grid_size(self) -> int:
        return round(2 * self.patch_radius / self.cell_size) + 1
