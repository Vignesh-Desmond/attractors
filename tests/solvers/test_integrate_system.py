import numpy as np
import pytest
from numba import njit

from attractors.solvers.core import (
    integrate_system,
)
from attractors.solvers.registry import Solver
from attractors.systems.registry import System
from attractors.type_defs import Vector


@pytest.fixture()
def lorenz_system():
    """Create a Lorenz system for testing - using a real attractor from our package"""

    def system_func(state: Vector, params: Vector) -> Vector:
        x, y, z = state
        sigma, rho, beta = params
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return np.array([dx, dy, dz], dtype=np.float64)

    return System(
        func=system_func,
        jitted_func=njit(system_func),
        name="test_lorenz",
        params=np.array([10.0, 28.0, 8 / 3]),
        param_names=["sigma", "rho", "beta"],
        reference="test",
        init_coord=np.array([1.0, 1.0, 1.0]),
    )


@pytest.fixture()
def euler_solver():
    """Simple Euler solver step for testing"""

    def euler_step(system_func, state, params, dt):
        return state + dt * system_func(state, params)

    return Solver(euler_step, njit(euler_step), "euler")


class TestCore:
    def test_integrate_system_wrapper(self, lorenz_system, euler_solver):
        steps, dt = 1000, 0.01

        trajectory, time = integrate_system(lorenz_system, euler_solver, steps, dt, use_jit=False)

        assert trajectory.shape == (steps, 3)
        assert time.shape == (steps,)
        assert trajectory.dtype == np.float64
        assert time.dtype == np.float64
        assert np.allclose(time, np.arange(steps) * dt)

        assert not np.any(np.isnan(trajectory))
        assert not np.any(np.isinf(trajectory))
        assert np.all(np.abs(trajectory[:, 0]) < 50)
        assert np.all(np.abs(trajectory[:, 1]) < 50)
        assert np.all(trajectory[:, 2] > 0)

    def test_error_handling(self, lorenz_system, euler_solver):
        """Test basic error handling"""
        with pytest.raises((ValueError, TypeError)):
            integrate_system(lorenz_system, euler_solver, -100, 0.01)

        with pytest.raises((ValueError, TypeError)):
            integrate_system(lorenz_system, euler_solver, 100, -0.01)

    def test_deterministic_behavior(self, lorenz_system, euler_solver):
        """Test that integration is deterministic for same inputs"""
        steps, dt = 1000, 0.01

        traj1, time1 = integrate_system(lorenz_system, euler_solver, steps, dt)
        traj2, time2 = integrate_system(lorenz_system, euler_solver, steps, dt)

        np.testing.assert_array_equal(traj1, traj2)
        np.testing.assert_array_equal(time1, time2)

    def test_jitted_vs_nonjit_consistency(self, lorenz_system, euler_solver):
        """Test that jitted and non-jitted versions give same results"""
        steps, dt = 1000, 0.01

        traj1, time1 = integrate_system(lorenz_system, euler_solver, steps, dt, use_jit=False)
        traj2, time2 = integrate_system(lorenz_system, euler_solver, steps, dt, use_jit=True)

        np.testing.assert_allclose(traj1, traj2, rtol=1e-14)
        np.testing.assert_allclose(time1, time2, rtol=1e-14)
