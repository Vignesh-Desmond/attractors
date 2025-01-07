from attractors.solvers.registry import SolverRegistry
from attractors.type_defs import ParamVector, StateVector, SystemCallable


@SolverRegistry.register("rk3")
def rk3(
    system_func: SystemCallable, state: StateVector, params: ParamVector, dt: float
) -> StateVector:
    k1 = system_func(state, params)
    k2 = system_func(state + dt * k1 / 2, params)
    k3 = system_func(state - dt * k1 + 2 * dt * k2, params)
    return state + dt * (k1 + 4 * k2 + k3) / 6
