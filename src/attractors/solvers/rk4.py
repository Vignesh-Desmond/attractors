from attractors.solvers.registry import SolverRegistry
from attractors.type_defs import SystemCallable, Vector


@SolverRegistry.register("rk4")
def rk4(system_func: SystemCallable, state: Vector, params: Vector, dt: float) -> Vector:
    k1 = system_func(state, params)
    k2 = system_func(state + dt * k1 / 2, params)
    k3 = system_func(state + dt * k2 / 2, params)
    k4 = system_func(state + dt * k3, params)
    result: Vector = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return result
