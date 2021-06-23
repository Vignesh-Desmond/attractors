#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
import matplotlib.pyplot as plt
from src.utils.runge_kutta import RK
from src.utils.attractors import ATTRACTOR_PARAMS
from src.utils.colortable import get_continuous_cmap
from src.utils.video import ffmpeg_video

def animate_simulation(attractor, width, height, dpi, bgcolor, palette, sim_time, points, n, integrator, interactive = False, rk2_method = "heun", fps = 60, outf = "output.mp4"):
    
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.axis('off')
    fig.set_facecolor(bgcolor) 
    ax.set_facecolor(bgcolor)

    attr = ATTRACTOR_PARAMS[attractor]
    init_coord = np.array(attr["init_coord"], dtype='double')
    attr_params = dict(zip(attr["params"], attr["default_params"]))
    xlim = attr["xlim"]
    ylim = attr["ylim"]
    zlim = attr["zlim"]
    
    init_coords = [init_coord] + [init_coord + np.random.normal(0, 0.01, 3) for _ in range(n-1)]

    attractor_vects = [RK(xyz, attractor, attr_params) for xyz in init_coords]

    for vect in attractor_vects:
        try:
            rk = getattr(vect, integrator)
            if integrator == "RK2":
                rk(0, sim_time, points, rk2_method)
            else:
                rk(0, sim_time, points)
        except AttributeError as e:
            raise Exception(f"Integrator Error. {integrator} is not an valid integrator") from e
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    if isinstance(palette, str):
        cmap = plt.cm.get_cmap(palette)
    else:
        cmap = get_continuous_cmap(palette)
    
    colors = cmap(np.linspace(0, 1, len(init_coords)))

    lines = sum([ax.plot([], [], [], '-', c=c, linewidth=1, antialiased=True)
                for c in colors], [])
    pts = sum([ax.plot([], [], [], 'o', c=c)
            for c in colors], [])                   

    def init():
        for line, pt in zip(lines, pts):
            line.set_data_3d([], [], [])
            pt.set_data_3d([], [], [])
        return lines + pts

    def update(i):
        i = i % len(attractor_vects[0].X)
        for line, pt, k in zip(lines, pts, attractor_vects):
            if i>15000:
                line.set_data_3d(k.X[i-15000:i], k.Y[i-15000:i], k.Z[i-15000:i])
            else:
                line.set_data_3d(k.X[:i], k.Y[:i], k.Z[:i])
            pt.set_data_3d(k.X[i], k.Y[i], k.Z[i])
        ax.view_init(0.005 * i, 0.05 * i)
        return lines + pts

    if interactive:
        anim = animation.FuncAnimation(fig, update, init_func=init,
                                frames=18000//4, interval=5, blit=False)
        plt.show()
    else:
        ffmpeg_video(fig, update, points, fps, outf)  