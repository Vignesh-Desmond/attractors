from enum import Enum

import numpy as np

from attractors.type_defs import Vector
from attractors.utils.logger import setup_logger

logger = setup_logger(name=__name__)


class CompressionMethod(Enum):
    """Compression methods for trajectory downsampling.
    none: No compression
    uniform: Uniform downsampling
    velocity: Velocity-based importance sampling
    curvature: Full curvature-based sampling
    """

    NONE = "none"
    UNIFORM = "uniform"
    VELOCITY = "velocity"
    CURVATURE = "curvature"


def _downsample_trajectory(
    trajectory: Vector,
    compression: float = 0.0,
    method: CompressionMethod = CompressionMethod.VELOCITY,
) -> Vector:
    """
    Downsample trajectory while preserving important features.

    Args:
        trajectory: Input trajectory points (N x 3)
        compression: Compression ratio from 0.0 (no compression) to 1.0 (max compression)
        method: Compression method to use
        min_points: Minimum number of points to retain

    Returns:
        Downsampled trajectory
    """
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("Trajectory must be Nx3 array")

    if not 0.0 <= compression <= 1.0:
        raise ValueError("Compression must be between 0 and 1")

    n_points = len(trajectory)
    if compression <= 0 or method == CompressionMethod.NONE:
        logger.debug("No downsampling applied")
        return trajectory

    target_points = int(n_points * (1 - compression))
    logger.debug("Downsampling trajectory to %d points", target_points)
    if target_points >= n_points:
        return trajectory

    if method in (CompressionMethod.VELOCITY, CompressionMethod.CURVATURE):
        if n_points < 3:
            return trajectory

        def compute_derivatives(pos: Vector) -> tuple[Vector, Vector]:
            vel = np.diff(pos, axis=0)
            acc = np.diff(vel, axis=0)
            return vel, acc

        velocity, acceleration = compute_derivatives(trajectory)

        if method == CompressionMethod.VELOCITY:
            speed_changes = np.linalg.norm(acceleration, axis=1)
            importance = np.pad(speed_changes, (1, 1), mode="edge")
        else:  # curvature
            speed = np.linalg.norm(velocity, axis=1)
            speed = np.maximum(speed, 1e-10)
            curvature = np.linalg.norm(np.cross(velocity[:-1], acceleration), axis=1) / (
                speed[:-1] ** 3
            )
            importance = np.pad(curvature, (1, 1), mode="edge")
            importance = 0.8 * (importance / np.maximum(importance.max(), 1e-10)) + 0.2

        probabilities = importance / np.sum(importance)
        indices = np.random.choice(n_points, size=target_points, replace=False, p=probabilities)
        return trajectory[np.sort(indices)]

    indices = np.linspace(0, n_points - 1, target_points, dtype=int)
    return trajectory[indices]
