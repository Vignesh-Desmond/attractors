from attractors.solvers.registry import SolverRegistry
from attractors.type_defs import SystemCallable, Vector


@SolverRegistry.register("euler")
def euler(system_func: SystemCallable, state: Vector, params: Vector, dt: float) -> Vector:
    return state + dt * system_func(state, params)
