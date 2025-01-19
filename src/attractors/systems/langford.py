import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="langford",
    default_params=np.array([0.95, 0.7, 0.6, 3.5, 0.25, 0.1]),
    param_names=["alpha", "beta", "lmbda", "omega", "rho", "epsilon"],
    init_coord=np.array([0.1, 0.0, 0.0]),
    reference=(
        "W. F. Langford, Numerical studies of torus bifurcations, Numerical methods for "
        "bifurcation problems  (Dortmund, 1983), Internat. Schriftenreihe Numer. Math., vol. 70, "
        "Birkhäuser, Basel, 1984,  pp. 285 - 295."
    ),
    plot_lims={"xlim": (-2.0, 2.0), "ylim": (-2.0, 2.0), "zlim": (-0.5, 2.0)},
)
def langford(state: Vector, params: Vector) -> Vector:
    """Langford attractor system."""
    x, y, z = state
    alpha, beta, lmbda, omega, rho, epsilon = params
    dx = (z - beta) * x - omega * y
    dy = omega * x + (z - beta) * y
    dz = lmbda + alpha * z - (z**3 / 3) - (x**2 + y**2) * (1 + rho * z) + epsilon * z * x**3
    return np.array([dx, dy, dz], dtype=np.float64)
