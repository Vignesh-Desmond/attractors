from dataclasses import dataclass

import matplotlib.colors


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    foreground: str
    colors: str | list[str]

    @property
    def colormap(self) -> matplotlib.colors.Colormap:
        if isinstance(self.colors, str):
            return matplotlib.colormaps[self.colors]
        rgb_colors = [
            tuple(int(c.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)) for c in self.colors
        ]
        return matplotlib.colors.LinearSegmentedColormap.from_list(self.name, rgb_colors)
