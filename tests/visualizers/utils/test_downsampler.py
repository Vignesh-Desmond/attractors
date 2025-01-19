import numpy as np
import pytest

from attractors.visualizers.utils.downsampler import CompressionMethod, _downsample_trajectory


class TestDownsampler:
    @pytest.fixture()
    def sample_trajectory(self):
        """Create a sample trajectory for testing"""
        t = np.linspace(0, 10, 1000)
        x = np.sin(t)
        y = np.cos(t)
        z = t / 10
        return np.column_stack([x, y, z])

    def test_input_validation(self, sample_trajectory):
        """Test input validation"""

        with pytest.raises(ValueError, match="Compression must be between 0 and 1"):
            _downsample_trajectory(sample_trajectory, compression=1.5)

        with pytest.raises(ValueError, match="Trajectory must be Nx3 array"):
            _downsample_trajectory(np.array([[1, 2]]))

    def test_no_compression(self, sample_trajectory):
        """Test with no compression"""
        result = _downsample_trajectory(sample_trajectory, compression=0.0)
        assert np.array_equal(result, sample_trajectory)

        result = _downsample_trajectory(
            sample_trajectory, compression=0.0, method=CompressionMethod.NONE
        )
        assert np.array_equal(result, sample_trajectory)

    def test_full_compression_limits(self, sample_trajectory):
        """Test behavior with maximum compression"""
        result = _downsample_trajectory(sample_trajectory, compression=1.0)
        assert len(result) < len(sample_trajectory)
        assert result.shape[1] == 3

    @pytest.mark.parametrize("method", list(CompressionMethod))
    def test_compression_methods(self, sample_trajectory, method):
        """Test all compression methods"""
        compression = 0.5
        result = _downsample_trajectory(sample_trajectory, compression=compression, method=method)

        if method == CompressionMethod.NONE:
            assert len(result) == len(sample_trajectory)
        else:
            expected_length = int(len(sample_trajectory) * (1 - compression))
            max_diff = expected_length * 0.2 if method != CompressionMethod.UNIFORM else 1
            assert abs(len(result) - expected_length) <= max_diff
        assert result.shape[1] == 3

    def test_small_trajectory(self):
        """Test with very small trajectories"""
        tiny_trajectory = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        result = _downsample_trajectory(tiny_trajectory, compression=0.5)
        assert len(result) > 0
        assert result.shape[1] == 3

    def test_deterministic_uniform(self, sample_trajectory):
        """Test that uniform downsampling is deterministic"""
        result1 = _downsample_trajectory(
            sample_trajectory, compression=0.5, method=CompressionMethod.UNIFORM
        )
        result2 = _downsample_trajectory(
            sample_trajectory, compression=0.5, method=CompressionMethod.UNIFORM
        )

        assert np.array_equal(result1, result2)

    def test_velocity_based_sampling(self):
        """Test velocity-based importance sampling"""
        t = np.linspace(0, 10, 1000)
        trajectory = np.column_stack(
            [
                t * np.sin(t),
                t * np.cos(t),
                t,
            ]
        )

        result = _downsample_trajectory(
            trajectory, compression=0.5, method=CompressionMethod.VELOCITY
        )

        assert len(result) < len(trajectory)
        assert result.shape[1] == 3

    def test_curvature_based_sampling(self):
        """Test curvature-based importance sampling"""
        t = np.linspace(0, 10, 1000)
        trajectory = np.column_stack(
            [
                np.sin(t) * t,
                np.cos(t) * t,
                t,
            ]
        )

        result = _downsample_trajectory(
            trajectory, compression=0.5, method=CompressionMethod.CURVATURE
        )

        assert len(result) < len(trajectory)
        assert result.shape[1] == 3
