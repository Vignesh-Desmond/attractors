import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="chen_lu",
    default_params=np.array([36.0, 3.0, 17.0]),  # a, b, c
    param_names=["a", "b", "c"],
    init_coord=np.array([1.0, 1.0, 30.0]),
    reference=(
        'Lu, Jinhu & Chen, Guanrong. (2002). "A New Chaotic Attractor Coined.".'
        " International Journal of Bifurcation and Chaos. vol. 12. pp-659-661."
    ),
    plot_lims={"xlim": (-30.0, 30.0), "ylim": (-30.0, 30.0), "zlim": (0.0, 30.0)},
)
def chen_lu(state: Vector, params: Vector) -> Vector:
    """Chen-Lu attractor system."""
    x, y, z = state
    a, b, c = params
    dx = a * (y - x)
    dy = -(x * z) + c * y
    dz = x * y - b * z
    return np.array([dx, dy, dz], dtype=np.float64)
