import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="nose_hoover",
    default_params=np.array([1.0]),
    param_names=["a"],
    init_coord=np.array([0.1, 0.0, -0.1]),
    reference=(
        'Posch et al. "Canonical dynamics of the Nosé oscillator: Stability, order, '
        'and chaos." Physical Review A, 33(6), 4253-4265, 1986.'
    ),
    plot_lims={"xlim": (-3.0, 1.0), "ylim": (-3.0, 3.0), "zlim": (-3.0, 3.0)},
)
def nose_hoover(state: Vector, params: Vector) -> Vector:
    """Nose-Hoover attractor system."""
    x, y, z = state
    a = params[0]
    dx = a * y
    dy = -x + y * z
    dz = 1 - y * y
    return np.array([dx, dy, dz], dtype=np.float64)
