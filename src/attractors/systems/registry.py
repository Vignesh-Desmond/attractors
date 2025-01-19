from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

from numba import njit
from numba.core.dispatcher import Dispatcher

from attractors.type_defs import PlotLimits, SystemCallable, Vector
from attractors.utils.logger import setup_logger

logger = setup_logger(name=__name__)


@dataclass
class System:
    func: SystemCallable
    jitted_func: SystemCallable
    name: str
    params: Vector
    param_names: list[str]
    reference: str
    init_coord: Vector
    plot_lims: PlotLimits | None = None

    def set_params(self, params: Vector) -> None:
        if len(params) != len(self.param_names):
            msg = f"Expected {len(self.param_names)} parameters"
            raise ValueError(msg)
        logger.debug("Setting parameters: %s for system: %s", params, self.name)
        self.params = params

    def set_init_coord(self, coord: Vector) -> None:
        if len(coord) != 3:
            raise ValueError("State vector must have length 3")
        logger.debug("Setting initial coord: %s for system: %s", coord, self.name)
        self.init_coord = coord

    def get_func(self, jitted: bool = True) -> SystemCallable:
        return self.jitted_func if jitted else self.func

    def __repr__(self) -> str:
        return (
            f"System(name={self.name}, params={self.params}, param_names={self.param_names}, "
            f"init_coord={self.init_coord}, plot_lims={self.plot_lims})"
        )


F = TypeVar("F", bound=SystemCallable)


class SystemRegistry:
    _systems: ClassVar[dict[str, System]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        default_params: Vector,
        param_names: list[str],
        reference: str = "",
        init_coord: Vector,
        plot_lims: PlotLimits | None = None,
    ) -> Callable[[F], F]:
        def decorator(f: F) -> F:
            if not isinstance(name, str):
                raise TypeError("Name must be string")
            if not callable(f):
                raise TypeError("Must register callable")
            if name in cls._systems:
                msg = f"System {name} already registered"
                raise ValueError(msg)

            jitted_f = f if cls.is_jitted(f) else njit(f)

            cls._systems[name] = System(
                func=f,
                jitted_func=jitted_f,
                name=name,
                params=default_params,
                param_names=param_names,
                reference=reference,
                init_coord=init_coord,
                plot_lims=plot_lims,
            )
            return f

        logger.debug("Registered system: %s", name)
        return decorator

    @classmethod
    def get(cls, name: str) -> System:
        if name not in cls._systems:
            msg = f"System {name} not found"
            raise KeyError(msg)
        logger.debug("Getting system: %s", name)
        return cls._systems[name]

    @classmethod
    def list_systems(cls) -> list[str]:
        return list(cls._systems.keys())

    @staticmethod
    def is_jitted(func: Callable[..., Any]) -> bool:
        """Check if a function is jitted"""
        return isinstance(func, Dispatcher)
