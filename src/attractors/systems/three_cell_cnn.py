import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="three_cell_cnn",
    default_params=np.array([1.24, 1.1, 4.4, 3.21]),
    param_names=["p1", "p2", "r", "s"],
    init_coord=np.array([0.1, 0.1, 0.1]),
    reference=(
        "Arena, P., et al. (1998). Bifurcation and Chaos in Noninteger Order Cellular "
        "Neural Networks. International Journal of Bifurcation and Chaos, 8(7), 1527-1539."
    ),
    plot_lims={"xlim": (-1.5, 1.5), "ylim": (-1.5, 1.5), "zlim": (-2.0, 1.5)},
)
def three_cell_cnn(state: Vector, params: Vector) -> Vector:
    """Three-Cell CNN attractor system with nonlinear coupling."""
    x, y, z = state
    p1, p2, r, s = params

    fx = 0.5 * (np.abs(x + 1) - np.abs(x - 1))
    fy = 0.5 * (np.abs(y + 1) - np.abs(y - 1))
    fz = 0.5 * (np.abs(z + 1) - np.abs(z - 1))

    dx = -x + p1 * fx - s * fy - s * fz
    dy = -y - s * fx + p2 * fy - r * fz
    dz = -z - s * fx + r * fy + fz

    return np.array([dx, dy, dz], dtype=np.float64)
