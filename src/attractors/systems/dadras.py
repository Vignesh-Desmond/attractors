import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="dadras",
    default_params=np.array([3.0, 2.7, 1.7, 2.0, 9.0]),  # a, b, c, d, h
    param_names=["a", "b", "c", "d", "h"],
    init_coord=np.array([5.0, 0.0, -4.0]),
    reference=(
        "Dadras, Sara & Momeni, Hamid. (2009). A novel three-dimensional autonomous chaotic system "
        "generating two, three and four-scroll attractors. Physics Letters A. 373. 3637-3642."
    ),
    plot_lims={"xlim": (-15.0, 15.0), "ylim": (-10.0, 8.0), "zlim": (-12.0, 12.0)},
)
def dadras(state: Vector, params: Vector) -> Vector:
    """Dadras attractor system."""
    x, y, z = state
    a, b, c, d, h = params
    dx = y - a * x + b * y * z
    dy = c * y - x * z + z
    dz = d * x * y - h * z
    return np.array([dx, dy, dz], dtype=np.float64)
