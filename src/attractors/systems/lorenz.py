import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="lorenz",
    default_params=np.array([10.0, 28.0, 8 / 3]),
    param_names=["sigma", "rho", "beta"],
    init_coord=np.array([0.0, 1.0, 0.0], dtype=np.float64),
    reference=(
        'Lorenz, E. N. "Deterministic Nonperiodic Flow", '
        "Journal of Atmospheric Sciences, 20(2), 130-141, 1963."
    ),
    plot_lims={"xlim": (-20.0, 20.0), "ylim": (-30.0, 30.0), "zlim": (5.0, 45.0)},
)
def lorenz(state: Vector, params: Vector) -> Vector:
    """
    Lorenz attractor system.
    """
    x, y, z = state
    sigma, rho, beta = params
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=np.float64)
