from abc import ABC, abstractmethod

import numpy as np

from attractors.type_defs import Vector


class ColorMapper(ABC):
    """Base class for color mapping strategies."""

    @abstractmethod
    def map(self, trajectory: Vector) -> Vector:
        """Map trajectory points to color values in range [0,1]."""

    def _normalize(self, values: Vector) -> Vector:
        """Utility to normalize values to [0,1] range."""
        if values.max() == values.min():
            return np.full_like(values, 0.0, dtype=np.float64)
        return np.divide(
            np.subtract(values, values.min(), dtype=np.float64),
            np.subtract(values.max(), values.min(), dtype=np.float64),
            dtype=np.float64,
        )


class TimeColorMapper(ColorMapper):
    """Maps colors based on time/position in trajectory."""

    def map(self, trajectory: Vector) -> Vector:
        return np.linspace(0, 1, len(trajectory))


class CoordinateColorMapper(ColorMapper):
    """Maps colors based on x, y, or z coordinate values."""

    def __init__(self, axis: int):
        if axis not in (0, 1, 2):
            raise ValueError("Axis must be 0, 1 or 2")
        self.axis = axis

    def map(self, trajectory: Vector) -> Vector:
        return self._normalize(trajectory[:, self.axis])


class VelocityColorMapper(ColorMapper):
    """Maps colors based on point-to-point velocity."""

    def map(self, trajectory: Vector) -> Vector:
        velocities = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        return self._normalize(np.pad(velocities, (0, 1), "edge"))
