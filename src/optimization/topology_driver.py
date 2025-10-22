"""Topology optimization problem driver for SIMP-based compliance minimization.

This module provides the TOProblemDriver class which wraps ContinuumFESolver and
exposes GCMMA-compatible callbacks for compliance + volume constraint optimization.

References
----------
Section 2.1 of project documentation: SIMP with density filtering and sensitivity
back-projection for compliance-driven topology optimization.
"""

import numpy as np
from typing import Callable, Tuple, Dict

from src.core import ContinuumFESolver, ProblemConfig


class TOProblemDriver:
    """Builds GCMMA callbacks for SIMP compliance + volume constraint [Sec. 2.1].

    Notes
    -----
    - Objective: C(ρ) = f^T u(ρ); SIMP stiffness: K(ρ) = Σ (Emin + ρ_e^p (E0-Emin)) K_e^0.
    - Sensitivity: dC/dρ_e = -p ρ_e^{p-1} (E0-Emin) u_e^T K_e^0 u_e.  (Apply filtering/back-projection.)
    - Constraint: g(ρ) = mean(ρ) - V* ≤ 0   (or Σ v_e ρ_e / V_tot - V* ≤ 0).
    """
    def __init__(self, fe: ContinuumFESolver, cfg: ProblemConfig) -> None:
        self.fe = fe
        self.cfg = cfg

    def build_gcmma_callbacks(self) -> Tuple[
        Callable[[np.ndarray], float],   # f0
        Callable[[np.ndarray], np.ndarray],  # df0
        Callable[[np.ndarray], np.ndarray],  # f
        Callable[[np.ndarray], np.ndarray],  # df
        np.ndarray, np.ndarray, np.ndarray   # x0, xmin, xmax
    ]:
        to = self.cfg.to
        vol = self.fe.volumes
        vol_tot = float(np.sum(vol))
        Vstar = float(to.vol_frac)
        p = float(to.simp_p)

        rho_min = self.fe.rho_bounds[0]
        rho_max = self.fe.rho_bounds[1]
        x0 = self.fe.rho0.copy()

        # Cache for last FE solve (to reuse u, φ)
        cache: Dict[str, np.ndarray | float] = {}

        def f0(rho: np.ndarray) -> float:
            """Compliance objective C = f^T u(ρ), with densities filtered in FE driver."""
            rho = np.clip(rho, rho_min, rho_max)
            rho_f = self.fe.filter_densities(rho)
            u = self.fe.assemble_K_and_solve(rho_f)
            cache["u"] = u
            # compliance via FE: prefer f^T u provided by FE driver; fallback: u^T K u
            C = self.fe.compliance(u) if hasattr(self.fe, "compliance") else float(np.dot(u, u))
            cache["C"] = C
            # also cache φ_e for sensitivity (element strain energy density)
            phi = self.fe.element_strain_energy(u)  # shape (ne,)
            cache["phi"] = phi
            cache["rho_f"] = rho_f
            return float(C)

        def df0(rho: np.ndarray) -> np.ndarray:
            """SIMP derivative with filtering back-projection."""
            # use cached quantities if available
            rho = np.clip(rho, rho_min, rho_max)
            if "phi" not in cache or "rho_f" not in cache:
                _ = f0(rho)
            rho_f = cache["rho_f"]
            phi = cache["phi"]
            # SIMP raw derivative wrt filtered density
            # dC/dρ_f ≈ -p ρ_f^{p-1} (E0-Emin) φ_e   (absorbed (E0-Emin) into scaling in FE driver if needed)
            dC_drho_f = -p * np.maximum(rho_f, 0.0)**(p-1) * phi
            # back-project through the filter (FE driver should implement this adjoint or transpose op)
            dC_drho = self.fe.backproject_sensitivities(dC_drho_f)
            # enforce bounds safety (zero out gradient where clamped if you wish)
            return dC_drho

        def f(rho: np.ndarray) -> np.ndarray:
            """Single volume-fraction constraint: g(ρ) = (Σ v_e ρ_e)/V_tot - V* ≤ 0."""
            val = (float(np.dot(vol, rho)) / vol_tot) - Vstar
            return np.array([val], dtype=float)

        def df(rho: np.ndarray) -> np.ndarray:
            """Gradient of volume constraint: ∂g/∂ρ_e = v_e / V_tot."""
            return np.array([vol / vol_tot], dtype=float)

        return f0, df0, f, df, x0, rho_min, rho_max
