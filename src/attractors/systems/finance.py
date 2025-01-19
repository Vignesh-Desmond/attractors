import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="finance",
    default_params=np.array([1e-05, 0.1, 1.0]),  # a, b, c
    param_names=["a", "b", "c"],
    init_coord=np.array([0.0, -10.0, 0.1]),
    reference=(
        "Cai & Huang (2007). A new finance chaotic attractor. International Journal of "
        "Nonlinear Science. vol 3. pp. 1479-3889."
    ),
    plot_lims={"xlim": (-3.0, 3.0), "ylim": (-15.0, -5.0), "zlim": (-1.5, 1.5)},
)
def finance(state: Vector, params: Vector) -> Vector:
    """Finance attractor system."""
    x, y, z = state
    a, b, c = params
    dx = (1 / b - a) * x + x * y + z
    dy = -b * y - x * x
    dz = -x - c * z
    return np.array([dx, dy, dz], dtype=np.float64)
