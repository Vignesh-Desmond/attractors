#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import re
from src.attractors.anim.gradient import animate_gradient
from src.attractors.anim.sim import animate_simulation
from src.attractors.utils.attr import ATTRACTOR_PARAMS
from argparse import SUPPRESS

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

from src.attractors import data

def case_convert(string):
    string = re.sub(r"[\-_\.\s]", joiner, str(string))

def cli():
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')

    parser.add_argument(
        '-h',
        '--help',
        action='help',
        default=SUPPRESS,
        help='show this help message and exit'
    )

    required.add_argument(
        "-t",
        "--type",
        help=("Simulation Type"),
        type=str,
        choices=["multipoint", "gradient"],
        required=True
    )

    optional.add_argument(
        "--des",
        help=("Differential Equation Solver." " Default: rk4"),
        type=str,
        choices=["euler", "rk2", "rk3", "rk4", "rk5" ],
        default="rk4",
    )

    optional.add_argument(
        "--width",
        help=("Width of the figure" " Default: 16"),
        type=int,
        default=16,
    )
    optional.add_argument(
        "--height",
        help=("Height of the figure" " Default: 9"),
        type=int,
        default=9,
    )
    optional.add_argument(
        "--dpi",
        help=("DPI of the figure" " Default: 120"),
        type=int,
        default=120,
    )
    optional.add_argument(
        "--theme",
        help=("Theme (color palette) to be used"),
        type=str
    )
    required.add_argument(
        "-s",
        "--simtime",
        help=("Simulation time"),
        type=int,
        required=True
    )
    required.add_argument(
        "-p",
        "--simpoints",
        help=("Number of points to be used for the simulation."),
        type=int,
        required=True
    )
    optional.add_argument(
        "--bgcolor",
        help=("Background color for figure in hex. Overrides theme settings."" Default: #000000"),
        type=str,
    )
    optional.add_argument(
        "--cmap",
        help=("Matplotlib cmap for palette. Overrides theme settings."" Default: jet"),
        type=str,
    )
    optional.add_argument(
        "--fps",
        help=("Set FPS for animated video (or interactive plot)"" Default: 60"),
        type=int,
        default=60
    )
    optional.add_argument(
        "--n",
        help=("Number of initial points for Multipoint animation"" Default: 3"),
        type=int,
        default=3
    )
    optional.add_argument(
        "--rk2",
        help=("Method for 2nd order Runge-Kutta if specified to used."" Default: heun"),
        type=str,
        default="heun",
        choices=["heun", "imp_poly", "ralston"]
    )
    optional.add_argument(
        "--outf",
        help=("Output video filename"" Default: output.mp4"),
        type=str,
        default="output.mp4",
    )
    optional.add_argument(
        "--live",
        help=("Live plotting instead of generating video."),
        action='store_true'
    )

    subparsers = parser.add_subparsers(title="Attractor settings", description="Choose one of the attractors and specify its parameters", dest="attractor", metavar="ATTRACTOR")
    
    for attr, attrparams in ATTRACTOR_PARAMS.items():
        attrparser = subparsers.add_parser(f"{attr}", help=f"{attr} attractor")
        attrgroup = attrparser.add_argument_group(title=f"{attr} attractor parameters")

        for i in range(len(attrparams['params'])):
            attrgroup.add_argument(
                f"--{attrparams['params'][i]}",
                help=(f"Parameter for {attr} attractor " f"Default: {attrparams['default_params'][i]}"),
                type=int,
                default=attrparams["default_params"][i])
        attrgroup.add_argument(
                f"--initcoord",
                help=(f"Initial coordinate for {attr} attractor. Input format: \"x,y,z\" " f"Default: {attrparams['init_coord']}"),
                type=lambda s: [int(item) for item in s.split(',')],
                default=attrparams['init_coord'])
        for k in ["x", "y", "z"]:
            attrgroup.add_argument(
                    f"--{k}lim",
                    help=(f"{k} axis limits for figure. Input format: \"{k}min,{k}max\" " f"Default: {attrparams[f'{k}lim']}"),
                    type=lambda s: [int(item) for item in s.split(',')],
                    default=attrparams[f'{k}lim'])

    args = parser.parse_args()

    # * load theme
    raw_themes_data = pkg_resources.open_text(data, 'themes.json')
    themes = json.load(raw_themes_data)

    # print(args)
    # print(args.bgcolor is not None)

    # * load args
    attractor = args.attractor
    theme = themes[args.theme]
    bgcolor =  "#000000"
    palette =  "jet"
    if theme is not None:
        palette_temp = list(theme.values())
        palette_temp.remove(theme["background"])
        bgcolor = args.bgcolor if args.bgcolor is not None else theme["background"] 
        palette = args.cmap if args.cmap is not None else palette_temp  
    width = args.width
    height = args.height
    dpi = args.dpi
    fps = args.fps
    simtime = args.simtime
    simpoints = args.simpoints
    n = args.n
    des = args.des
    rk2_method = args.rk2
    outf = args.outf
    live = args.live

    if args.type == "multipoint":
        animate_simulation(
            attractor, width, height, dpi, bgcolor, palette, simtime, simpoints, n, des, live, rk2_method, fps, outf
        )
    else:
        animate_gradient(
            attractor, width, height, dpi, bgcolor, palette, simtime, simpoints, des, rk2_method, fps, outf
        )

if __name__ == ' __main__':
    cli()