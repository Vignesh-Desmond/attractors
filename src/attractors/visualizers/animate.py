import typing
from collections.abc import Callable, Sequence
from typing import Any, TypedDict

import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]
from mpl_toolkits.mplot3d.art3d import Line3D  # type: ignore[import-untyped]

from attractors.type_defs import Vector
from attractors.visualizers.base import BasePlotter


class AnimatedVisualizeKwargs(TypedDict, total=False):
    speed: int
    interval: int
    rotate_view: Callable[[Axes3D], None] | None
    line_kwargs: dict[str, Any] | None
    anim_kwargs: dict[str, Any] | None


class AnimatedPlotter(BasePlotter):
    def _visualize(
        self,
        trajectory: Vector,
        **kwargs: AnimatedVisualizeKwargs,
    ) -> "AnimatedPlotter":
        self.speed = typing.cast(int, kwargs.get("speed", 20))
        self.line_kwargs = typing.cast(dict[str, Any], kwargs.get("line_kwargs", {})) or {}
        self.anim_kwargs = typing.cast(dict[str, Any], kwargs.get("anim_kwargs", {})) or {}
        interval = typing.cast(int, kwargs.get("interval", 1))
        rotate_view = typing.cast(Callable[[Axes3D], None] | None, kwargs.get("rotate_view"))

        self._setup_plot()
        segment_length = len(trajectory) // self.num_segments
        colors = self.theme.colormap(np.linspace(0, 1, self.num_segments))
        lines: list[Line3D] = [
            self.ax.plot([], [], [], "-", c=color, **self.line_kwargs)[0] for color in colors
        ]

        def init() -> list[Line3D]:
            for line in lines:
                line.set_data_3d([], [], [])
            return lines

        def update(frame: int) -> Sequence[Line3D]:
            frame = frame * self.speed
            active_segments = min(frame // segment_length + 1, self.num_segments)

            for i in range(self.num_segments):
                if i < active_segments:
                    start_idx = i * segment_length
                    end_idx = min(start_idx + segment_length, frame + 1)
                    lines[i].set_data_3d(
                        trajectory[start_idx:end_idx, 0],
                        trajectory[start_idx:end_idx, 1],
                        trajectory[start_idx:end_idx, 2],
                    )

                ax: Axes3D = self.ax
                if rotate_view is not None:
                    rotate_view(ax)

            return lines

        self.anim = animation.FuncAnimation(
            self.fig,
            update,
            init_func=init,
            frames=len(trajectory) // self.speed,
            interval=interval,
            **self.anim_kwargs,
        )

        return self
