import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="chen_lee",
    default_params=np.array([5.0, -10.0, -0.38]),  # a, b, c
    param_names=["a", "b", "c"],
    init_coord=np.array([1.0, 1.0, 1.0]),
    reference=(
        'Chen HK, Lee CI. "Anti-control of chaos in rigid body motion.", '
        "Chaos, Solitons & Fractals (2004), vol. 21, pp. 957 - 65"
    ),
    plot_lims={"xlim": (-30.0, 30.0), "ylim": (-30.0, 30.0), "zlim": (-1.0, 35.0)},
)
def chen_lee(state: Vector, params: Vector) -> Vector:
    """Chen-Lee attractor system."""
    x, y, z = state
    a, b, c = params
    dx = a * x - y * z
    dy = b * y + x * z
    dz = c * z + x * y / 3
    return np.array([dx, dy, dz], dtype=np.float64)
