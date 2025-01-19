import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="lotka_volterra",
    default_params=np.array([2.9851, 3.0, 2.0]),
    param_names=["a", "b", "c"],
    init_coord=np.array([1.0, 1.0, 1.0]),
    reference=(
        "J. S. Costello, “Synchronization of chaos in a generalized Lotka-Volterra attractor,” "
        "The Nonlinear Journal, vol. 1, pp. 11 - 17, 1999."
    ),
    plot_lims={"xlim": (0.7, 1.3), "ylim": (0.7, 1.3), "zlim": (0.5, 1.1)},
)
def lotka_volterra(state: Vector, params: Vector) -> Vector:
    """Lotka-Volterra attractor system."""
    x, y, z = state
    a, b, c = params
    dx = x - x * y + c * x * x - a * z * x * x
    dy = -y + x * y
    dz = -b * z + a * z * x * x
    return np.array([dx, dy, dz], dtype=np.float64)
