import numpy as np
import pytest
from numba import njit

from attractors import SolverRegistry
from attractors.type_defs import Vector


@pytest.fixture()
def test_state():
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


@pytest.fixture()
def test_params():
    return np.array([0.2], dtype=np.float64)


def nonlinear_system(state: Vector, params: Vector) -> Vector:
    x, y, z = state
    a = params[0]
    dx = y
    dy = -x + a * y
    dz = x * y - z
    return np.array([dx, dy, dz], dtype=np.float64)


class TestSolverRegistry:
    @pytest.mark.parametrize("solver_name", ["euler", "rk2", "rk3", "rk4", "rk5"])
    def test_solver_consistency(self, solver_name, test_state, test_params):
        """Test that both JIT and non-JIT versions give same results"""
        dt = 0.01
        solver = SolverRegistry.get(solver_name)

        jit_func = solver.get_func(jitted=True)
        jit_result = jit_func(njit(nonlinear_system), test_state, test_params, dt)

        nojit_func = solver.get_func(jitted=False)
        nojit_result = nojit_func(nonlinear_system, test_state, test_params, dt)

        np.testing.assert_allclose(jit_result, nojit_result, rtol=1e-14)

    @pytest.mark.parametrize("solver_name", ["euler", "rk2", "rk3", "rk4", "rk5"])
    def test_solver_deterministic(self, solver_name, test_state, test_params):
        """Test solver produces same results for same inputs"""
        solver = SolverRegistry.get(solver_name)
        dt = 0.01
        solver_func = solver.get_func()
        result1 = solver_func(njit(nonlinear_system), test_state, test_params, dt)
        result2 = solver_func(njit(nonlinear_system), test_state, test_params, dt)

        np.testing.assert_array_equal(result1, result2)

    def test_invalid_solver_access(self):
        """Test error handling for invalid solver access"""
        with pytest.raises(KeyError, match="Solver nonexistent_solver not found"):
            SolverRegistry.get("nonexistent_solver")
