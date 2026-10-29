"""Algorithms included in this clean backend."""
from .core_data import Individual
from .nsga2_parallel import NSGA2Parallel
from .spea2_parallel import SPEA2Parallel

__all__ = ["Individual", "NSGA2Parallel", "SPEA2Parallel"]
