import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="chen",
    default_params=np.array([35.0, 3.0, 28.0]),  # a, b, c
    param_names=["a", "b", "c"],
    init_coord=np.array([-10.0, 0.0, 37.0]),
    reference=(
        'Chen, G. & Ueta, T. "Yet another chaotic attractor," '
        "International Journal of Bifurcation and Chaos 9, 1465 - 1466. [1999]"
    ),
    plot_lims={"xlim": (-30.0, 30.0), "ylim": (-30.0, 30.0), "zlim": (5.0, 45.0)},
)
def chen(state: Vector, params: Vector) -> Vector:
    """Chen attractor system."""
    x, y, z = state
    a, b, c = params
    dx = a * (y - x)
    dy = (c - a) * x - (x * z) + c * y
    dz = x * y - b * z
    return np.array([dx, dy, dz], dtype=np.float64)
