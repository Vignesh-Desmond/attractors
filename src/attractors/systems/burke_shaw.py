import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="burke_shaw",
    default_params=np.array([10.0, 4.272]),  # s, v
    param_names=["s", "v"],
    init_coord=np.array([1.0, 0.0, 0.0]),
    reference=(
        'Shaw, Robert. "Strange Attractors, Chaotic Behavior, and Information Flow" '
        "Zeitschrift für Naturforschung A, vol. 36, no. 1, 1981, pp. 80-112."
    ),
    plot_lims={"xlim": (-2.5, 2.5), "ylim": (-2.5, 2.5), "zlim": (-2.0, 2.0)},
)
def burke_shaw(state: Vector, params: Vector) -> Vector:
    """Burke-Shaw attractor system."""
    x, y, z = state
    s, v = params
    dx = -s * (x + y)
    dy = -y - s * x * z
    dz = s * x * y + v
    return np.array([dx, dy, dz], dtype=np.float64)
