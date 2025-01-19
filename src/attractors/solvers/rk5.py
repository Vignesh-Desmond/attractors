from attractors.solvers.registry import SolverRegistry
from attractors.type_defs import SystemCallable, Vector


@SolverRegistry.register("rk5")
def rk5(system_func: SystemCallable, state: Vector, params: Vector, dt: float) -> Vector:
    k1 = system_func(state, params)
    k2 = system_func(state + dt * k1 / 4, params)
    k3 = system_func(state + dt * (k1 + k2) / 8, params)
    k4 = system_func(state + dt * k3, params)
    k5 = system_func(state + dt * (k1 - k2 + k4) / 2, params)
    k6 = system_func(state + dt * (k1 - 3 * k3 + 4 * k5) / 2, params)
    result: Vector = state + dt * (k1 + 4 * k5 + k6) / 6
    return result
