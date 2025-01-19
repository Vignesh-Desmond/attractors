import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="moore_spiegel",
    default_params=np.array([20.0, 100.0]),
    param_names=["t", "r"],
    init_coord=np.array([0.0, 0.8, 0.0]),
    reference=(
        'Moore, D. W., & Spiegel, E. A. (1966). "A thermally excited nonlinear oscillator," '
        "The Astrophysical Journal, 143, 871-887."
    ),
    plot_lims={"xlim": (-10.0, 10.0), "ylim": (-20.0, 20.0), "zlim": (-250.0, 250.0)},
)
def moore_spiegel(state: Vector, params: Vector) -> Vector:
    """Moore-Spiegel attractor system."""
    x, y, z = state
    t, r = params
    dx = y
    dy = z
    dz = -z - (t - r * (1 - x * x)) * y - t * x
    return np.array([dx, dy, dz], dtype=np.float64)
