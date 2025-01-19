from collections.abc import Callable
from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

Vector: TypeAlias = NDArray[np.float64]
SystemCallable: TypeAlias = Callable[[Vector, Vector], Vector]
SolverCallable: TypeAlias = Callable[[SystemCallable, Vector, Vector, float], Vector]


class PlotLimits(TypedDict):
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    zlim: tuple[float, float]
