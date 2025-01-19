import numpy as np
import pytest

from attractors.visualizers.utils.color_mapper import (
    ColorMapper,
    CoordinateColorMapper,
    TimeColorMapper,
    VelocityColorMapper,
)


@pytest.fixture()
def sample_trajectory():
    """Create a simple test trajectory"""
    return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])


class TestColorMapper:
    @pytest.fixture()
    def sample_trajectory(self):
        """Create a simple test trajectory"""
        return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])

    def test_normalize_helper(self):
        """Test the _normalize helper method"""

        class TestMapper(ColorMapper):
            def map(self, trajectory):
                return self._normalize(trajectory[:, 0])

        mapper = TestMapper()
        test_values = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = mapper._normalize(test_values)

        assert np.allclose(normalized, np.array([0.0, 1 / 3, 2 / 3, 1.0]))

    def test_normalize_constant_values(self):
        """Test normalization with constant values"""

        class TestMapper(ColorMapper):
            def map(self, trajectory):
                return self._normalize(trajectory[:, 0])

        mapper = TestMapper()
        test_values = np.array([2.0, 2.0, 2.0, 2.0])
        normalized = mapper._normalize(test_values)

        assert np.allclose(normalized, np.zeros_like(test_values))


class TestTimeColorMapper:
    def test_linear_mapping(self, sample_trajectory):
        """Test that TimeColorMapper creates linear spacing"""
        mapper = TimeColorMapper()
        colors = mapper.map(sample_trajectory)

        expected = np.linspace(0, 1, len(sample_trajectory))
        assert np.allclose(colors, expected)

    def test_different_lengths(self):
        """Test TimeColorMapper with different trajectory lengths"""
        mapper = TimeColorMapper()

        for length in [2, 5, 10]:
            trajectory = np.zeros((length, 3))
            colors = mapper.map(trajectory)
            assert len(colors) == length
            assert colors[0] == 0.0
            assert colors[-1] == 1.0


class TestCoordinateColorMapper:
    def test_invalid_axis(self):
        """Test that invalid axis raises ValueError"""
        with pytest.raises(ValueError, match="Axis must be 0, 1 or 2"):
            CoordinateColorMapper(axis=3)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_axis_mapping(self, axis, sample_trajectory):
        """Test mapping for each axis"""
        mapper = CoordinateColorMapper(axis=axis)
        colors = mapper.map(sample_trajectory)

        assert len(colors) == len(sample_trajectory)
        assert np.all(colors >= 0.0)
        assert np.all(colors <= 1.0)

        expected = np.linspace(0, 1, len(sample_trajectory))
        assert np.allclose(colors, expected)


class TestVelocityColorMapper:
    def test_velocity_calculation(self):
        """Test velocity-based color mapping"""
        trajectory = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])

        mapper = VelocityColorMapper()
        colors = mapper.map(trajectory)

        assert len(colors) == len(trajectory)
        assert np.all(colors >= 0.0)
        assert np.all(colors <= 1.0)

    def test_constant_velocity(self):
        """Test with constant velocity"""
        trajectory = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        mapper = VelocityColorMapper()
        colors = mapper.map(trajectory)

        assert np.allclose(colors, colors[0])

    def test_stationary_trajectory(self):
        """Test with a stationary trajectory"""
        trajectory = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

        mapper = VelocityColorMapper()
        colors = mapper.map(trajectory)

        assert np.allclose(colors, np.zeros_like(colors))
