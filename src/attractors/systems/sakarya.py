import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="sakarya",
    default_params=np.array([0.4, 0.3]),  # a, b
    param_names=["a", "b"],
    init_coord=np.array([1.0, -1.0, 1.0]),
    reference="NA",
    plot_lims={"xlim": (-35.0, 30.0), "ylim": (-17.0, 15.0), "zlim": (-13.0, 17.0)},
)
def sakarya(state: Vector, params: Vector) -> Vector:
    """Sakarya attractor system."""
    x, y, z = state
    a, b = params
    dx = -x + y + y * z
    dy = -x - y + a * x * z
    dz = z - b * x * y
    return np.array([dx, dy, dz], dtype=np.float64)
