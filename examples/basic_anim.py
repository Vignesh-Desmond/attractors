import matplotlib.pyplot as plt

from attractors import (
    AnimatedPlotter,
    SolverRegistry,
    SystemRegistry,
    ThemeManager,
    integrate_system,
)

# setup
theme = ThemeManager.get("solarized_darcula")
system = SystemRegistry.get("rucklidge")
solver = SolverRegistry.get("rk4")
steps = 1000000
dt = 0.001

# generate trajectory
trajectory, time = integrate_system(system, solver, steps, dt)


def rotate_view(ax) -> None:
    ax.view_init(elev=30, azim=(ax.azim + 10) % 360)


# use the AnimatedPlotter class to create an animation
plotter = AnimatedPlotter(
    system,
    theme,
    fig_kwargs={"figsize": (10, 8)},
).visualize(
    trajectory,
    line_kwargs={"linewidth": 2, "antialiased": True, "alpha": 0.9},
    rotate_view=rotate_view,
    anim_kwargs={"blit": True},
)
plt.show()
