import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="yu_wang",
    default_params=np.array([10.0, 40.0, 2.0, 2.5]),  # a, b, c, d
    param_names=["a", "b", "c", "d"],
    init_coord=np.array([0.1, 0.0, 15.0]),
    reference=(
        'F. Yu, C. H. Wang, and J. W. Yin, "A 4-D chaos with fully qualified four-wing type," '
        "Acta Physica Sinica, vol. 61, (2012)."
    ),
    plot_lims={"xlim": (-3.0, 3.0), "ylim": (-5.0, 5.0), "zlim": (0.0, 45.0)},
)
def yu_wang(state: Vector, params: Vector) -> Vector:
    """Yu-Wang attractor system."""
    x, y, z = state
    a, b, c, d = params
    dx = a * (y - x)
    dy = b * x - c * x * z
    dz = np.exp(x * y) - d * z
    return np.array([dx, dy, dz], dtype=np.float64)
