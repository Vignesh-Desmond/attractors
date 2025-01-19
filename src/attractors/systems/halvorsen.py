import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="halvorsen",
    default_params=np.array([1.89]),  # a
    param_names=["a"],
    init_coord=np.array([-1.48, 1.51, 2.04]),
    reference=(
        "J. C. Sprott and J. C. Sprott, Chaos and time-series analysis, Vol. 69 (Citeseer, 2003)"
    ),
    plot_lims={"xlim": (-20.0, 15.0), "ylim": (-12.0, 8.0), "zlim": (-12.0, 8.0)},
)
def halvorsen(state: Vector, params: Vector) -> Vector:
    """Halvorsen attractor system."""
    x, y, z = state
    a = params[0]
    dx = -a * x - 4 * y - 4 * z - y * y
    dy = -a * y - 4 * z - 4 * x - z * z
    dz = -a * z - 4 * x - 4 * y - x * x
    return np.array([dx, dy, dz], dtype=np.float64)
