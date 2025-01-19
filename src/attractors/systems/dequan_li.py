import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="dequan_li",
    default_params=np.array([40.0, 1.833, 0.16, 0.65, 55.0, 20.0]),  # a, c, d, e, k, f
    param_names=["a", "c", "d", "e", "k", "f"],
    init_coord=np.array([0.01, 0.0, 0.0]),
    reference=(
        'Li, Dequan., "A three-scroll chaotic attractor." Physics Letters A. 372. 387-393. (2008).'
    ),
    plot_lims={
        "xlim": (-200.0, 200.0),
        "ylim": (-200.0, 250.0),
        "zlim": (-50.0, 250.0),
    },
)
def dequan_li(state: Vector, params: Vector) -> Vector:
    """Dequan-Li attractor system."""
    x, y, z = state
    a, c, d, e, k, f = params
    dx = a * (y - x) + d * x * z
    dy = k * x + f * y - x * z
    dz = c * z + x * y - e * x * x
    return np.array([dx, dy, dz], dtype=np.float64)
