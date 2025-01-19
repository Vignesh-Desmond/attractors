from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

from numba import njit
from numba.core.dispatcher import Dispatcher

from attractors.type_defs import (
    SolverCallable,
)
from attractors.utils.logger import setup_logger

logger = setup_logger(name=__name__)


@dataclass
class Solver:
    func: SolverCallable
    jitted_func: SolverCallable
    name: str

    def get_func(self, jitted: bool = True) -> SolverCallable:
        return self.jitted_func if jitted else self.func

    def __repr__(self) -> str:
        return f"Solver(name={self.name})"


F = TypeVar("F", bound=SolverCallable)


class SolverRegistry:
    _solvers: ClassVar[dict[str, Solver]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[F], F]:
        def decorator(f: F) -> F:
            if not isinstance(name, str):
                raise TypeError("Name must be string")
            if not callable(f):
                raise TypeError("Must register callable")
            if name in cls._solvers:
                msg = f"Solver {name} already registered"
                raise ValueError(msg)

            jitted_f = njit(f)
            cls._solvers[name] = Solver(f, jitted_f, name)
            return f

        logger.debug("Registered solver: %s", name)
        return decorator

    @classmethod
    def get(cls, name: str) -> Solver:
        if name not in cls._solvers:
            msg = f"Solver {name} not found"
            raise KeyError(msg)
        logger.debug("Getting solver: %s", name)
        return cls._solvers[name]

    @classmethod
    def list_solvers(cls) -> list[str]:
        return list(cls._solvers.keys())

    @staticmethod
    def is_jitted(func: Callable[..., Any]) -> bool:
        """Check if a function is jitted"""
        return isinstance(func, Dispatcher)
