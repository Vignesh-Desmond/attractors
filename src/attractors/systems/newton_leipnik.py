import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="newton_leipnik",
    default_params=np.array([0.4, 0.175]),  # alpha, beta
    param_names=["alpha", "beta"],
    init_coord=np.array([0.349, 0.0, -0.160]),
    reference=(
        'Leipnik, R. B. & Newton, T. A. "Double strange attractors in rigid body motion '
        'with linear feedback control," Phys. Lett. A86, 63 - 67. (1981)'
    ),
    plot_lims={"xlim": (-0.7, 0.7), "ylim": (-0.4, 0.4), "zlim": (-0.4, 0.6)},
)
def newton_leipnik(state: Vector, params: Vector) -> Vector:
    """Newton-Leipnik attractor system."""
    x, y, z = state
    alpha, beta = params
    dx = -alpha * x + y + 10 * y * z
    dy = -x - 0.4 * y + 5 * x * z
    dz = beta * z - 5 * x * y
    return np.array([dx, dy, dz], dtype=np.float64)
