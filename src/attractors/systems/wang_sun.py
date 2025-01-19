import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="wang_sun",
    default_params=np.array([0.2, -0.01, 1.0, -0.4, -1.0, -1.0]),
    param_names=["a", "b", "c", "d", "e", "f"],
    init_coord=np.array([0.5, 0.1, 0.1]),
    reference=(
        "Wang, Z., Sun, Y., van Wyk, B. J., Qi, G. & van Wyk, M. A. "
        "“A 3-D four-wing attractor and its analysis,” Brazilian J. Phys. 39, (2009) 547-553."
    ),
    plot_lims={"xlim": (-4.0, 4.0), "ylim": (-4.0, 4.0), "zlim": (-3.0, 2.0)},
)
def wang_sun(state: Vector, params: Vector) -> Vector:
    """Wang-Sun attractor system."""
    x, y, z = state
    a, b, c, d, e, f = params
    dx = a * x + c * y * z
    dy = b * x + d * y - x * z
    dz = e * z + f * x * y
    return np.array([dx, dy, dz], dtype=np.float64)
