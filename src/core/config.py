"""
Configuration dataclasses for OPT STM Generator.

This module contains all configuration classes for topology optimization,
shape optimization, and GCMMA optimizer settings.

References to paper sections from:
"Optimisation-based 3D Strut-and-Tie Reinforced Concrete Design"
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TOConfig:
    """Topology-optimization (SIMP) configuration [Sec. 2.1].
    
    Attributes
    ----------
    vol_frac : float
        Target volume fraction V* in (0,1], e.g. 0.05 (pile cap), 0.10 (corbel/box).
    simp_p : float
        SIMP penalization exponent (robust default ~3).
    rho_min : float
        Void floor to keep K non-singular (e.g., 1e-4).
    filter_radius : float
        Density/sensitivity filter radius (≈2.5× mesh; corbel ≈1.5×).
    """
    vol_frac: float = 0.10
    simp_p: float = 3.0
    rho_min: float = 1e-4
    filter_radius: float = 2.5  # measured in mesh-size units or physical, per your FE backend


@dataclass
class ShapeConfig:
    """STM shape-optimization configuration [Sec. 2.3].
    
    Attributes
    ----------
    sts_epsilon : float
        STS tolerance in STS ≥ 1 - epsilon (≈0.05).
    min_length_p : int
        p-norm exponent for continuous minimum bar length (≈10).
    min_length_target : float
        L_min target (same units as coordinates).
    fd_step_rel : float
        Central FD step as a fraction of continuum mesh size (~0.001 = 0.1%).
    coord_move_limit : float
        Per-iteration coordinate move cap (~0.5 × mesh size) — enforce in your orchestrator.
    """
    sts_epsilon: float = 0.05
    min_length_p: int = 10
    min_length_target: float = 0.0
    fd_step_rel: float = 1e-3
    coord_move_limit: float = 0.5


@dataclass
class OptimizerConfig:
    """GCMMA defaults aligned with paper & tests.
    
    TO (m=1): prefer many iters; Shape (m>1): fewer iters, conservative moves.
    
    Attributes
    ----------
    c_default : float
        Linear cost on y variables in GCMMA (reasonably large, ~3000).
    d_default : float
        Quadratic weight on y variables (prefer 1.0).
    move : float
        Move limit (0.45 for TO; 0.25-0.35 for shape optimization).
    asyinit : float
        Initial asymptote distance (~0.5 for TO; ~0.3 for shape).
    asyincr : float
        Asymptote expansion factor (1.2).
    asydecr : float
        Asymptote contraction factor (0.7).
    feas_tol : float
        Feasibility tolerance (start 1e-3, tighten to 1e-5 later).
    kkt_tol : float
        KKT convergence tolerance (1e-5).
    max_outer : int
        Maximum outer iterations (500 for TO; 50-150 for shape).
    max_inner : int
        Maximum inner iterations per outer loop (30).
    sys_choice : str
        Linear system choice: "auto", "lambda" (for TO), or "x" (if m >> n).
    verbose : bool
        Print iteration progress.
    """
    # Shared
    c_default: float = 3000.0
    d_default: float = 1.0
    move: float = 0.45          # TO-friendly; for shape use 0.25–0.35
    asyinit: float = 0.5        # ~0.3 for shape
    asyincr: float = 1.2
    asydecr: float = 0.7
    feas_tol: float = 1e-3      # start loose, tighten later (1e-5)
    kkt_tol: float = 1e-5
    max_outer: int = 500        # TO; for shape ~50–150
    max_inner: int = 30
    sys_choice: str = "auto"    # "lambda" for TO; "x" if m >> n
    verbose: bool = False


@dataclass
class ProblemConfig:
    """Top-level configuration holding all inputs (use case + solver).
    
    This class aggregates all configuration objects for easy parameter management.
    
    Attributes
    ----------
    to : TOConfig
        Topology optimization configuration.
    shape : ShapeConfig
        Shape optimization configuration.
    opt : OptimizerConfig
        GCMMA optimizer configuration.
    """
    to: TOConfig = field(default_factory=TOConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    opt: OptimizerConfig = field(default_factory=OptimizerConfig)
