from typing import Any

from attractors.type_defs import Vector
from attractors.visualizers.base import BasePlotter


class StaticPlotter(BasePlotter):
    def _visualize(
        self,
        trajectory: Vector,
        line_kwargs: dict[str, Any] | None = None,
        segment_overlap: int = 1,
        **kwargs: Any,
    ) -> "StaticPlotter":
        self._setup_plot()

        line_kwargs = line_kwargs or {}

        n = len(trajectory)
        segment_size = n // self.num_segments
        overlap = segment_overlap

        segments = []
        color_values = self._get_color_values(trajectory)
        color_segments = []

        for i in range(self.num_segments):
            start_idx = max(0, i * segment_size - overlap)
            end_idx = min(n, (i + 1) * segment_size + overlap)
            segments.append(trajectory[start_idx:end_idx])
            color_segments.append(color_values[start_idx:end_idx])

        for segment, color_segment in zip(segments, color_segments, strict=False):
            self.ax.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                "-",
                c=self.theme.colormap(color_segment.mean()),
                **line_kwargs,
            )
        return self
