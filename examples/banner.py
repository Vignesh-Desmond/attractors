from typing import Any

import matplotlib.animation as animation
import numpy as np

from attractors import (
    SolverRegistry,
    StaticPlotter,
    SystemRegistry,
    ThemeManager,
    integrate_system,
)
from attractors.type_defs import Vector


class ThemeCyclingPlotter(StaticPlotter):
    def visualize(
        self,
        trajectory: Vector,
        n_frames: int = 30,
        **kwargs: Any,
    ) -> "ThemeCyclingPlotter":
        super().visualize(trajectory, **kwargs)

        frames = range(n_frames)

        def update_theme(_) -> None:
            self.theme = ThemeManager.random()
            self.fig.set_facecolor(self.theme.background)
            self.ax.set_facecolor(self.theme.background)

            self.ax.text2D(
                0.7,
                0.5,
                "ttractors",
                fontsize=80,
                ha="center",
                va="center",
                transform=self.fig.transFigure,
                color=self.theme.foreground,
                alpha=0.8,
                fontname="Sora",
                weight="light",
                rasterized=False,
            )

            colors = self.theme.colormap(np.linspace(0, 1, self.num_segments))
            for line, color in zip(self.ax.lines, colors, strict=False):
                line.set_color(color)

        self.theme_anim = animation.FuncAnimation(
            self.fig,
            update_theme,
            frames=frames,
            interval=1000,
            save_count=n_frames,
            blit=False,
        )
        return self


theme = ThemeManager.random()
system = SystemRegistry.get("rucklidge")
solver = SolverRegistry.get("rk4")
steps = 1000000
dt = 0.001

trajectory, time = integrate_system(system, solver, steps, dt)

plotter = ThemeCyclingPlotter(
    system,
    theme,
    fig_kwargs={"figsize": (12, 10)},
).visualize(
    trajectory, theme_interval=2000, line_kwargs={"linewidth": 1, "antialiased": True, "alpha": 0.5}
)


plotter.ax.view_init(elev=152.90, azim=40.754)


# to find the limits of the plot
# (so that I can move the ruckllidge attractor to the left of the text)
# def on_xlim_changed(event_ax):
#     print(f"xlim updated: {event_ax.get_xlim()}")


# def on_ylim_changed(event_ax):
#     print(f"ylim updated: {event_ax.get_ylim()}")


# def on_zlim_changed(event_ax):
#     print(f"zlim updated: {event_ax.get_zlim()}")


# plotter.ax.callbacks.connect("xlim_changed", on_xlim_changed)
# plotter.ax.callbacks.connect("ylim_changed", on_ylim_changed)
# plotter.ax.callbacks.connect("zlim_changed", on_zlim_changed)

# use the plot limits obtained from above
plotter.ax.set_xlim(np.float64(-15.641259786104337), np.float64(5.98313413705718))
plotter.ax.set_ylim(np.float64(-9.856908159289528), np.float64(12.055811016180805))
plotter.ax.set_zlim(np.float64(-2.6899039982519675), np.float64(19.078652551063936))

save = plotter.theme_anim.save("attractor.gif", writer="pillow", dpi=300)
