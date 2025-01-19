import matplotlib.pyplot as plt

from attractors import (
    SolverRegistry,
    StaticPlotter,
    SystemRegistry,
    ThemeManager,
    integrate_system,
)

# you cal also use custom themes, or get predefined themes from the ThemeManager
# theme = Theme(
#     name="monochrome",
#     background="#FFFFFF",
#     colors=["#616161", "#7a7a7a", "#2e2e2e", "#1c1c1c"],
#     foreground="#000000",  # Black foreground for contrast with white background
# )
from attractors.utils.logger import set_log_level

set_log_level("DEBUG")  # or set_log_level(logging.DEBUG)

# setup
theme = ThemeManager.random()
system = SystemRegistry.get("rossler")
solver = SolverRegistry.get("rk4")
steps = 1000000
dt = 0.001

# generate trajectory
trajectory, time = integrate_system(system, solver, steps, dt)

# use the StaticPlotter class to create an plot
plotter = StaticPlotter(
    system,
    theme,
    fig_kwargs={"figsize": (16, 9)},
    color_by="time",
).visualize(
    trajectory,
    compression_method="curvature",
    compression=0.8,
    line_kwargs={"linewidth": 1, "antialiased": True, "alpha": 0.5},
)


plotter.ax.get_xaxis().set_visible(False)
plotter.ax.get_yaxis().set_visible(False)


plt.tight_layout(pad=0)


# mng = plt.get_current_fig_manager()
# mng.full_screen_toggle()

plt.show()
