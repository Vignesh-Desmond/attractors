from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from matplotlib import pyplot as plt

from attractors.systems.registry import System
from attractors.themes.theme import Theme
from attractors.type_defs import Vector
from attractors.visualizers.utils.color_mapper import (
    ColorMapper,
    CoordinateColorMapper,
    TimeColorMapper,
    VelocityColorMapper,
)
from attractors.visualizers.utils.downsampler import CompressionMethod, _downsample_trajectory


class BasePlotter(ABC):
    VALID_COLOR_OPTIONS = ("time", "x", "y", "z", "velocity")

    def __init__(
        self,
        system: System,
        theme: Theme,
        num_segments: int = 50,
        color_by: str | ColorMapper = "time",
        color_cycles: float = 1.0,
        fig_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.color_mapper: ColorMapper
        self.system = system
        self.theme = theme
        self.num_segments = num_segments
        self.color_cycles = color_cycles
        self.fig_kwargs = fig_kwargs or {}

        if isinstance(color_by, str):
            if color_by not in self.VALID_COLOR_OPTIONS:
                msg = f"color_by must be one of {self.VALID_COLOR_OPTIONS} or a ColorMapper"
                raise ValueError(msg)

            if color_by == "time":
                self.color_mapper = TimeColorMapper()
            elif color_by == "velocity":
                self.color_mapper = VelocityColorMapper()
            else:  # x, y, z
                self.color_mapper = CoordinateColorMapper({"x": 0, "y": 1, "z": 2}[color_by])
        else:
            self.color_mapper = color_by

    def _validate_inputs(
        self, num_segments: int, color_cycles: float, color_by: str | Callable[[Vector], Vector]
    ) -> None:
        if num_segments <= 0:
            raise ValueError("num_segments must be positive")
        if color_cycles <= 0:
            raise ValueError("color_cycles must be positive")
        if isinstance(color_by, str) and color_by not in self.VALID_COLOR_OPTIONS:
            error_message = f"color_by must be one of {self.VALID_COLOR_OPTIONS} or a callable"
            raise ValueError(error_message)

    def _setup_plot(self) -> None:
        self.fig = plt.figure(facecolor=self.theme.background, **self.fig_kwargs)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor(self.theme.background)
        self.ax.set_axis_off()

        for axis in ["x", "y", "z"]:
            if self.system.plot_lims:
                getattr(self.ax, f"set_{axis}lim")(*self.system.plot_lims[f"{axis}lim"])  # type: ignore[literal-required]

    def _get_color_values(self, trajectory: Vector) -> Vector:
        try:
            values = self.color_mapper.map(trajectory)
            return (values * self.color_cycles) % 1.0
        except Exception as e:
            msg = f"Error in color mapping: {e!s}"
            raise ValueError(msg) from e

    def visualize(
        self,
        trajectory: Vector,
        compression: float = 0.0,
        compression_method: CompressionMethod = CompressionMethod.VELOCITY,
        **kwargs: Any,
    ) -> "BasePlotter":
        """Template method that handles downsampling before specific visualization"""
        processed = _downsample_trajectory(trajectory, compression, compression_method)
        return self._visualize(processed, **kwargs)

    @abstractmethod
    def _visualize(self, trajectory: Vector, **kwargs: Any) -> "BasePlotter":
        """Implementation specific visualization logic"""
