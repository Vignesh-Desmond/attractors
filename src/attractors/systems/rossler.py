import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="rossler",
    default_params=np.array([0.2, 0.2, 5.7]),
    param_names=["a", "b", "c"],
    init_coord=np.array([0.1, 0.0, -0.1]),
    reference=(
        'Rossler, O. E. "An Equation for Continuous Chaos," '
        "Physics Letters A, 57(5), 397-398, 1976."
    ),
    plot_lims={"xlim": (-15.0, 15.0), "ylim": (-15.0, 15.0), "zlim": (-1.0, 20.0)},
)
def rossler(state: Vector, params: Vector) -> Vector:
    """Rossler attractor system."""
    x, y, z = state
    a, b, c = params
    dx = -(y + z)
    dy = x + a * y
    dz = b + z * (x - c)
    return np.array([dx, dy, dz], dtype=np.float64)
