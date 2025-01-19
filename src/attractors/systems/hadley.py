import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="hadley",
    default_params=np.array([0.2, 4.0, 8.0, 1.0]),  # a, b, f, g
    param_names=["a", "b", "f", "g"],
    init_coord=np.array([0.0, 0.0, 1.0]),
    reference=(
        "J. C. Sprott and J. C. Sprott, Chaos and time-series analysis, Vol. 69 (Citeseer, 2003)"
    ),
    plot_lims={"xlim": (-1.0, 3.0), "ylim": (-2.0, 2.0), "zlim": (-2.0, 2.0)},
)
def hadley(state: Vector, params: Vector) -> Vector:
    """Hadley attractor system."""
    x, y, z = state
    a, b, f, g = params
    dx = -y * y - z * z - a * (x - f)
    dy = x * y - b * x * z - y + g
    dz = b * x * y + z * (x - 1)
    return np.array([dx, dy, dz], dtype=np.float64)
