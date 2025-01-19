import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="thomas",
    default_params=np.array([0.208]),  # b
    param_names=["b"],
    init_coord=np.array([0.01, 0.0, 0.0]),
    reference=(
        'Thomas, René. "DETERMINISTIC CHAOS SEEN IN TERMS OF FEEDBACK CIRCUITS: '
        'ANALYSIS, SYNTHESIS, "LABYRINTH CHAOS"." International Journal of '
        "Bifurcation and Chaos 9 (1999): 1889-1905."
    ),
    plot_lims={"xlim": (-2.0, 5.0), "ylim": (-2.0, 4.0), "zlim": (-2.0, 4.0)},
)
def thomas(state: Vector, params: Vector) -> Vector:
    """Thomas attractor system."""
    x, y, z = state
    b = params[0]
    dx = np.sin(y) - b * x
    dy = np.sin(z) - b * y
    dz = np.sin(x) - b * z
    return np.array([dx, dy, dz], dtype=np.float64)
