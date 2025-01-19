import numpy as np

from attractors.systems.registry import SystemRegistry
from attractors.type_defs import Vector


@SystemRegistry.register(
    name="bouali_type_1",
    default_params=np.array([0.02, 0.2, 0.4, 10.0, 0.1, 50.0]),  # k, b, mu, p, q, s
    param_names=["k", "b", "mu", "p", "q", "s"],
    init_coord=np.array([0.012, 3.69, -0.09]),
    reference=(
        "S. Bouali, et al. (2012). Emulating complex business cycles by using an "
        "electronic analogue"
    ),
    plot_lims={"xlim": (-0.05, 0.05), "ylim": (-5.0, 5.0), "zlim": (-0.2, 0.2)},
)
def bouali_type_1(state: Vector, params: Vector) -> Vector:
    """Bouali Type 1 attractor system."""
    x, y, z = state
    k, b, mu, p, q, s = params

    dx = k * y + mu * x * (b - y * y)
    dy = -x + s * z
    dz = p * x - q * y

    return np.array([dx, dy, dz], dtype=np.float64)


@SystemRegistry.register(
    name="bouali_type_2",
    default_params=np.array([4.0, 1.0, 1.4, 2.8, 1.0, 1.0]),  # a, b, c, s, alpha, beta
    param_names=["a", "b", "c", "s", "alpha", "beta"],
    init_coord=np.array([0.1, 3.0, 0.2]),
    reference="Bouali, S. (2012). A novel strange attractor with a stretched loop",
    plot_lims={"xlim": (-8.0, 8.0), "ylim": (-3.0, 13.0), "zlim": (-20.0, 1.5)},
)
def bouali_type_2(state: Vector, params: Vector) -> Vector:
    """Bouali Type 2 attractor system."""
    x, y, z = state
    a, b, c, s, alpha, beta = params

    dx = x * (a - y) + alpha * z
    dy = -y * (b - x * x)
    dz = -x * (c - s * z) - beta * z

    return np.array([dx, dy, dz], dtype=np.float64)


@SystemRegistry.register(
    name="bouali_type_3",
    default_params=np.array([1.0, 0.001, 3.0, 2.2]),  # gamma, mu, alpha, beta
    param_names=["gamma", "mu", "alpha", "beta"],
    init_coord=np.array([1.0, 1.0, 0.0]),
    reference="Bouali, S. (2013). A 3D Strange Attractor with a Distinctive Silhouette",
    plot_lims={"xlim": (-3.0, 3.0), "ylim": (0.0, 3.0), "zlim": (-0.15, 0.15)},
)
def bouali_type_3(state: Vector, params: Vector) -> Vector:
    """Bouali Type 3 attractor system."""
    x, y, z = state
    gamma, mu, alpha, beta = params

    dx = alpha * x * (1 - y) - beta * z
    dy = -gamma * y * (1 - x * x)
    dz = mu * x

    return np.array([dx, dy, dz], dtype=np.float64)
