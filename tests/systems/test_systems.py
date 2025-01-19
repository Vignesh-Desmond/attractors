import numpy as np
import pytest

from attractors import SolverRegistry, SystemRegistry


class TestSystemRegistry:
    @pytest.mark.parametrize("system_name", SystemRegistry.list_systems())
    def test_system_consistency(self, system_name):
        """Test that both JIT and non-JIT versions of each system give same results"""
        system = SystemRegistry.get(system_name)

        test_state = system.init_coord
        test_params = system.params

        jit_func = system.get_func(jitted=True)
        nojit_func = system.get_func(jitted=False)

        jit_result = jit_func(test_state, test_params)
        nojit_result = nojit_func(test_state, test_params)

        np.testing.assert_allclose(jit_result, nojit_result, rtol=1e-14)

    @pytest.mark.parametrize("system_name", SystemRegistry.list_systems())
    def test_system_deterministic(self, system_name):
        """Test system produces same results for same inputs"""
        system = SystemRegistry.get(system_name)
        test_state = system.init_coord
        test_params = system.params

        system_func = system.get_func()
        result1 = system_func(test_state, test_params)
        result2 = system_func(test_state, test_params)

        np.testing.assert_array_equal(result1, result2)

    @pytest.mark.parametrize("system_name", SystemRegistry.list_systems())
    def test_system_output_validity(self, system_name):
        """Test that system outputs are valid and have correct shape/type"""
        system = SystemRegistry.get(system_name)
        test_state = system.init_coord
        test_params = system.params

        system_func = system.get_func()
        result = system_func(test_state, test_params)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result.dtype == np.float64
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_system_parameter_validation(self):
        """Test parameter validation for systems"""
        system_name = "lorenz"
        system = SystemRegistry.get(system_name)

        invalid_params = np.array([1.0])
        with pytest.raises(ValueError, match="Expected .* parameters"):
            system.set_params(invalid_params)

    def test_system_state_validation(self):
        """Test state vector validation"""
        system_name = "lorenz"
        system = SystemRegistry.get(system_name)

        invalid_state = np.array([1.0, 1.0])
        with pytest.raises(ValueError, match="State vector must have length 3"):
            system.set_init_coord(invalid_state)

    def test_invalid_system_access(self):
        """Test error handling for invalid system access"""
        with pytest.raises(KeyError, match="System nonexistent_system not found"):
            SystemRegistry.get("nonexistent_system")

    @pytest.mark.parametrize("system_name", SystemRegistry.list_systems())
    def test_system_reference_validity(self, system_name):
        """Test that all systems have valid reference information"""
        system = SystemRegistry.get(system_name)
        assert isinstance(system.reference, str)
        assert len(system.reference) > 0 or system.reference == "NA"

    @pytest.mark.parametrize("system_name", SystemRegistry.list_systems())
    def test_system_plot_limits(self, system_name):
        """Test that plot limits are properly defined when present"""
        system = SystemRegistry.get(system_name)
        if system.plot_lims is not None:
            assert "xlim" in system.plot_lims
            assert "ylim" in system.plot_lims
            assert "zlim" in system.plot_lims
            assert all(
                isinstance(lim, tuple) and len(lim) == 2 for lim in system.plot_lims.values()
            )
            assert all(
                isinstance(v, int | float)
                for lim in [
                    system.plot_lims["xlim"],
                    system.plot_lims["ylim"],
                    system.plot_lims["zlim"],
                ]
                for v in lim
            )

    @pytest.mark.parametrize("system_name", SystemRegistry.list_systems())
    def test_system_param_names(self, system_name):
        """Test that parameter names are properly defined"""
        system = SystemRegistry.get(system_name)
        assert isinstance(system.param_names, list)
        assert len(system.param_names) > 0
        assert len(system.param_names) == len(system.params)
        assert all(isinstance(name, str) for name in system.param_names)

    @pytest.mark.parametrize("system_name", SystemRegistry.list_systems())
    def test_system_integration_stability(self, system_name):
        """Test basic integration stability for each system"""
        system = SystemRegistry.get(system_name)
        solver = SolverRegistry.get("rk4")
        solver_func = solver.get_func()

        dt = 0.01
        steps = 1000
        state = system.init_coord.copy()

        if system.plot_lims:
            bounds = [
                max(abs(lim[0]), abs(lim[1])) * 10  # 10x safety factor
                for lim in [
                    system.plot_lims["xlim"],
                    system.plot_lims["ylim"],
                    system.plot_lims["zlim"],
                ]
            ]
            max_bound = max(bounds)
        else:
            max_bound = 1e6  # fallback if no plot_lims

        try:
            for _ in range(steps):
                state = solver_func(system.get_func(), state, system.params, dt)

                assert not np.any(np.isnan(state))
                assert not np.any(np.isinf(state))
                assert np.all(np.abs(state) < max_bound)

        except AssertionError:
            pytest.fail(f"System {system_name} showed numerical instability")
