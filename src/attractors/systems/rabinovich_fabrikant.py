import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="rabinovich_fabrikant",
    default_params=np.array([1.1, 0.87]),
    param_names=["alpha", "gamma"],
    init_coord=np.array([-1.0, 0.0, 0.5]),
    reference=(
        "Rabinovich, M. I. and Fabrikant, A. L., “Stochastic self-modulation of waves in "
        "nonequilibrium media”, Soviet Journal of Experimental and Theoretical Physics, "
        "vol. 50, p. 311, 1979."
    ),
    plot_lims={"xlim": (-2.0, 0.0), "ylim": (-0.5, 3.0), "zlim": (0.0, 1.5)},
)
def rabinovich_fabrikant(state: Vector, params: Vector) -> Vector:
    """Rabinovich-Fabrikant attractor system."""
    x, y, z = state
    alpha, gamma = params
    dx = y * (z - 1 + x * x) + gamma * x
    dy = x * (3 * z + 1 - x * x) + gamma * y
    dz = -2 * z * (alpha + x * y)
    return np.array([dx, dy, dz], dtype=np.float64)
