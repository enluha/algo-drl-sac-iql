"""Shape optimization problem driver for STM beam structures.

This module provides the ShapeProblemDriver class which wraps BeamFESolver and
exposes GCMMA-compatible callbacks for compliance minimization subject to STS
(strut-tie strength) and minimum length constraints.

References
----------
Section 2.3 of project documentation: Shape optimization with central finite
differences for gradients, STS constraint, and p-norm minimum length constraint.
"""

import numpy as np
from typing import Callable, Tuple, Dict

from src.core import BeamFESolver, ProblemConfig


class ShapeProblemDriver:
    """Builds GCMMA callbacks for STM shape optimization [Sec. 2.3].

    Objective
    ---------
    Minimize compliance C(x) = f^T u(x) from beam FE.

    Constraints
    -----------
    1) STS(x) ≥ 1 - ε   →  g1(x) = (1 - ε) - STS(x) ≤ 0
       where STS = (1/n) Σ |N_e| / sqrt(N_e^2 + V_{e1}^2 + V_{e2}^2)

    2) L(x) ≥ L_min     →  g2(x) = L_min - L(x) ≤ 0
       where L(x) = ( Σ (1/L_e)^p )^{-1/p},   p≈10

    Gradients
    ---------
    Central finite differences (paper uses this) with step ≈ 0.1% mesh size; coordinates
    should respect a per-iteration move limit (≈ 50% mesh size) in the orchestrator.
    """
    def __init__(self, fe: BeamFESolver, cfg: ProblemConfig) -> None:
        self.fe = fe
        self.cfg = cfg

    # ---- helper metrics ----
    def _compute_sts(self, u: np.ndarray) -> float:
        N, V1, V2 = self.fe.element_forces(u)
        denom = np.sqrt(N**2 + V1**2 + V2**2) + 1e-16
        return float(np.mean(np.abs(N) / denom))

    def _compute_L_pnorm(self, x: np.ndarray, p:int) -> float:
        L = self.fe.element_lengths(x)
        return float( (np.sum( (1.0 / np.maximum(L, 1e-12))**p )) ** (-1.0/p) )

    def build_gcmma_callbacks(self) -> Tuple[
        Callable[[np.ndarray], float],      # f0
        Callable[[np.ndarray], np.ndarray], # df0
        Callable[[np.ndarray], np.ndarray], # f
        Callable[[np.ndarray], np.ndarray], # df
        np.ndarray, np.ndarray, np.ndarray  # x0, xmin, xmax
    ]:
        sh = self.cfg.shape
        p = int(sh.min_length_p)
        eps = float(sh.sts_epsilon)
        Lmin = float(sh.min_length_target)
        h = float(sh.fd_step_rel) * float(self.fe.mesh_size)  # central FD step

        x0 = self.fe.x0.copy()
        xmin, xmax = self.fe.x_bounds

        # cache last evaluation
        cache: Dict[str, np.ndarray | float] = {}

        def f0(x: np.ndarray) -> float:
            x = np.clip(x, xmin, xmax)
            u = self.fe.assemble_K_and_solve(x)
            cache["u"] = u
            C = float(self.fe.compliance(u))
            cache["C"] = C
            return C

        def _g_vals_from_u_and_x(u: np.ndarray, x: np.ndarray) -> np.ndarray:
            sts = self._compute_sts(u)
            Lx  = self._compute_L_pnorm(x, p)
            g1 = (1.0 - eps) - sts
            g2 = (Lmin - Lx)
            return np.array([g1, g2], dtype=float)

        def f(x: np.ndarray) -> np.ndarray:
            x = np.clip(x, xmin, xmax)
            u = self.fe.assemble_K_and_solve(x)
            cache["u"] = u
            return _g_vals_from_u_and_x(u, x)

        # ----- central FD helpers (vectorized by slicing) -----
        def _central_fd_grad(fun: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
            n = x.size
            g = np.zeros(n, dtype=float)
            for i in range(n):
                xi = x[i]
                hi = h
                x[i] = xi + hi
                f_plus = fun(x)
                x[i] = xi - hi
                f_minus = fun(x)
                x[i] = xi  # restore
                g[i] = (f_plus - f_minus) / (2.0*hi)
            return g

        def _central_fd_jac(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray, m:int) -> np.ndarray:
            n = x.size
            J = np.zeros((m, n), dtype=float)
            for i in range(n):
                xi = x[i]
                hi = h
                x[i] = xi + hi
                f_plus = fun(x)
                x[i] = xi - hi
                f_minus = fun(x)
                x[i] = xi
                J[:, i] = (f_plus - f_minus) / (2.0*hi)
            return J

        def df0(x: np.ndarray) -> np.ndarray:
            # Gradient of compliance via central FD (cheap with beam models; paper's choice).
            # Consider parallelizing across coordinates if this becomes a bottleneck.
            return _central_fd_grad(f0, x.copy())

        def df(x: np.ndarray) -> np.ndarray:
            # 2 constraints (STS and p-norm length), central FD Jacobian (m x n)
            return _central_fd_jac(f, x.copy(), m=2)

        return f0, df0, f, df, x0, xmin, xmax
