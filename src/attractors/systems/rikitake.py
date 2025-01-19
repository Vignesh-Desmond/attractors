import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="rikitake",
    default_params=np.array([5.0, 2.0]),
    param_names=["a", "mu"],
    init_coord=np.array([-0.7, 0.0, 5.0]),
    reference=(
        "Rikitake, Tsuneji. “Oscillations of a System of Disk Dynamos.” "
        "Mathematical Proceedings of the Cambridge Philosophical Society, "
        "vol. 54, no. 1, 1958, pp. 89 - 105."
    ),
    plot_lims={"xlim": (-7.0, 6.0), "ylim": (-5.0, 3.0), "zlim": (3.0, 9.0)},
)
def rikitake(state: Vector, params: Vector) -> Vector:
    """Rikitake attractor system."""
    x, y, z = state
    a, mu = params
    dx = -mu * x + z * y
    dy = -mu * y + x * (z - a)
    dz = 1 - x * y
    return np.array([dx, dy, dz], dtype=np.float64)
