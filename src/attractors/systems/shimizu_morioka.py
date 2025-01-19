import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="shimizu_morioka",
    default_params=np.array([0.45, 0.75]),  # a, B
    param_names=["a", "B"],
    init_coord=np.array([-1.0, 2.0, 1.0]),
    reference=(
        "Shimizu, T.; Morioka, N. On the bifurcation of a symmetric limit cycle to an "
        "asymmetric one in a simple model. Phys. Lett. A 1980, 76, 201 - 204."
    ),
    plot_lims={"xlim": (-10.0, 10.0), "ylim": (-10.0, 10.0), "zlim": (-10.0, 10.0)},
)
def shimizu_morioka(state: Vector, params: Vector) -> Vector:
    """Shimizu-Morioka attractor system."""
    x, y, z = state
    a, B = params
    dx = y
    dy = x - B * y - x * z
    dz = -a * z + x * x
    return np.array([dx, dy, dz], dtype=np.float64)
