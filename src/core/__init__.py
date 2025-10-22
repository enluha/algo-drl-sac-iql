"""Core algorithms and data structures for OPT STM Generator."""

from .optimizer import GCMMA, SubproblemResult
from .config import TOConfig, ShapeConfig, OptimizerConfig, ProblemConfig
from .fem_continuum import ContinuumFESolver
from .fem_beam import BeamFESolver

__all__ = [
    'GCMMA', 
    'SubproblemResult',
    'TOConfig',
    'ShapeConfig',
    'OptimizerConfig',
    'ProblemConfig',
    'ContinuumFESolver',
    'BeamFESolver',
]
