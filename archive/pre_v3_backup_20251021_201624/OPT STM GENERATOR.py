"""
3D-OPT-STM pipeline skeleton:
- Step 1: TO Optimization (SIMP compliance)  [Sec. 2.1]
- Step 2: TO Extraction (to truss-like graph) [Sec. 2.2]
- Step 3: STM Shape Optimization (beam model) [Sec. 2.3]

This module provides:
  * Configuration dataclasses keeping **all problem and solver inputs in one place**.
  * Abstract FE back-ends for continuum (TO) and beam (shape).
  * Implemented Drivers that wrap the FE back-ends and produce (f0, df0, f, df)
    for the GCMMA optimizer, with recommended defaults aligned to the paper
    and our tests (documented below).
  * Placeholders for Extraction and Visualisation layers.
  * A thin Orchestrator skeleton.

Citations (paper key points used here):
- SIMP compliance + volume fraction + filtering + PCG/DOF pruning [Sec. 2.1].  #  [oai_citation:4‡Optimisation_based_3D_Strut_n_Tie_Reinf_Conc.pdf](file-service://file-V14xVgnZPJpVSjkipxijwu)
- Extraction: threshold(≈0.1), 3D thinning (Lee et al., 1994), node detection (26-neighborhood),
  connection tracing, merging short bars into a mid-node [Sec. 2.2].             #  [oai_citation:5‡Optimisation_based_3D_Strut_n_Tie_Reinf_Conc.pdf](file-service://file-V14xVgnZPJpVSjkipxijwu)
- Shape: compliance objective (beam FE), constraints STS≥1−ε with ε≈0.05,
  p-norm minimum bar length with p≈10; central FD (~0.1% mesh) and
  move limits (~50% mesh) [Sec. 2.3].                                            #  [oai_citation:6‡Optimisation_based_3D_Strut_n_Tie_Reinf_Conc.pdf](file-service://file-V14xVgnZPJpVSjkipxijwu)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np


# =============================================================================
# Method of the Moving Asymptotes (MMA) solver
# =============================================================================

# A single-class, NumPy-based implementation of GCMMA (Globally Convergent
# Method of Moving Asymptotes) following Svanberg's 2007 Matlab notes
# "MMA and GCMMA – two methods for nonlinear optimization".
#
# This file contains one public class: GCMMA. It solves problems on form (1.2)
# via the extended formulation (1.1) and the GCMMA subproblems (4.1) with the
# interior-point primal–dual solver described in Section 5 of the note.
#
# Key properties:
# - Strict adherence to the approximations, asymptote updates and inner/outer
#   loop logic in the reference document.
# - Fully vectorized with NumPy; avoids building dense diagonal matrices.
# - Dimension-aware Newton linear solves: uses the (m+1)x(m+1) system (Eq. 5.20)
#   when n >> m, and (n+1)x(n+1) system (Eq. 5.22) when m >> n.
# - Conservative-approximation inner loop with rho-updates (Eq. 4.9) ensures the
#   global-convergence safeguards of GCMMA.
# - One public entrypoint: solve(...), which takes user-supplied objective,
#   constraints and gradients.
#
# Usage summary (see class docstring below for details):
#   g = GCMMA(n, m, xmin, xmax, a0=1.0, a=np.zeros(m), c=3000*np.ones(m), d=np.ones(m))
#   x_opt, info = g.solve(x0, f0, f, df0, df, max_outer=500)
#
# Defaults in this implementation are tuned to align with Section 2.1 (TO) and 2.3
# (Shape) of the paper and with the empirical tests we ran. See the docstring for
# mode-specific tips and tuning ranges. PCG, DOF pruning and parallel assembly are
# intentionally left to the TO driver (not inside the optimizer).

# import numpy as np
# from dataclasses import dataclass
# from typing import Callable, Dict, Tuple, Optional, Any

Array = np.ndarray

@dataclass
class SubproblemResult:
    x: Array
    y: Array
    z: float
    lam: Array
    xi: Array
    eta: Array
    mu: Array
    zeta: float
    s: Array
    # diagnostics
    iters: int
    eps: float
    res_inf: float


class GCMMA:
    """
    GCMMA solver (single class) with the primal–dual interior-point subproblem solver.

    Problem form (user-level / "standard NLP", Eq. 1.2):
        minimize   f0(x)
        subject to fi(x) <= 0,  i = 1..m
                   xmin <= x <= xmax

    This class implements Svanberg's globally convergent MMA (GCMMA) by solving the
    augmented problem (1.1) with artificial variables {y >= 0, z >= 0} and coefficients
    (a0, a, c, d), and iterating on conservative convex separable approximations of f0, fi.

    References to equations below are to Svanberg, "MMA and GCMMA – two methods for
    nonlinear optimization" (2007). The implementation mirrors the formulas exactly.

    Parameters
    ----------
    n : int
        Number of design variables.
    m : int
        Number of constraints.
    xmin, xmax : (n,) array_like
        Simple bounds on x. Must satisfy xmin < xmax elementwise.
    a0 : float, default 1.0
        Coefficient on z in the objective of (1.1).
    a : (m,) array_like, default zeros
        Coefficients on z in each constraint of (1.1).
    c : (m,) array_like, default 3000
        Linear costs on y in (1.1); choose "reasonably large" (Sec. 2) to push y -> 0.
    d : (m,) array_like, default ones
        Quadratic weights on y in (1.1). Must be nonnegative; prefer 1.

    Algorithm parameters (defaults = reference values unless noted)
    ----------------------------------------------------------------
    raa0 : float, default 1e-5
        Baseline regularization term used in pij,qij construction in MMA; in GCMMA it is
        replaced by rho_i/(xmax - xmin) but we retain as a minimum safeguard.
    asyinit : float, default 0.5
        Initial relative distance of asymptotes from x (Eq. 3.11).
    asyincr : float, default 1.2
        Asymptote expanding factor (Eq. 3.13).
    asydecr : float, default 0.7
        Asymptote contracting factor (Eq. 3.13).
    albefa : float, default 0.1
        Fraction in alpha/beta box (Eqs. 3.6–3.7).
    move : float, default 0.45
        Move limit (Eqs. 3.6–3.7).

    Subproblem solver parameters
    ----------------------------
    ip_eps_min : float, default 1e-7
        Final barrier epsilon (Section 5.5 termination).
    ip_max_iter : int, default 100
        Max Newton iterations for the subproblem solver.
    ip_tol_factor : float, default 0.9
        Infinity-norm residual threshold factor relative to epsilon.
    sys_choice : {"auto","lambda","x"}, default "auto"
        Choose Eq. (5.20) (lambda-system) or Eq. (5.22) (x-system) or let the solver
        pick automatically by comparing n and m.

    GCMMA outer/inner parameters
    ----------------------------
    max_outer : int, default 500
        Maximum outer iterations.
    max_inner : int, default 30
        Maximum inner iterations per outer loop (should rarely bind with correct logic).
    kkt_tol : float, default 1e-5
        Termination tolerance on a projected-KKT measure for the original problem.
    feas_tol : float, default 1e-3
        Constraint feasibility tolerance (original fi(x) <= feas_tol).

    Notes on usage for two common modes
    -----------------------------------
    • Topology Optimization (TO): m=1, many iterations.
      Objective: compliance (FE), constraint: volume fraction. Use SIMP with a
      small void floor (e.g., 1e-4). Filtering and FE choices (PCG, DOF pruning,
      parallel assembly) belong in the TO driver.
      Recommended defaults (already set): move≈0.45, asyinit=0.5, asyincr=1.2,
      asydecr=0.7, c≈3000, d=1, a0=1, a=0, sys_choice="lambda" (tiny system).
      Iterations: set max_outer≥500. Start with feas_tol≈1e-3 and tighten near
      the end (e.g., 1e-5) if you need stricter feasibility. Filter radius in
      driver: ≈2.5× mesh size (corbel may use ≈1.5×).

    • Shape discovery / multi-constraint: m>1, fewer iterations.
      Objective: beam-model compliance; constraints: STS≥1−ε (ε≈0.05),
      p-norm minimum length (p≈10), plus box bounds on coordinates. Use central
      finite differences in the driver (step≈0.1% of mesh size). Move limit in
      the driver for coordinates: ≈50% of mesh size.
      Recommended solver settings: move≈0.25–0.35, asyinit≈0.3, asyincr=1.2,
      asydecr=0.7, c≈3000, d=1, sys_choice="auto" (or "x" if m≫n),
      max_outer≈50–150, max_inner≈30.

    General tuning rules of thumb
    -----------------------------
    • move: larger accelerates but risks overshoot; 0.25–0.6 typical. Increase
      when progress stalls but constraints are slack; decrease if oscillatory.
    • asyinit: 0.3–0.6. Smaller for delicate multi-constraint shape problems.
    • c (penalty on y): 1e3–1e4. Increase if y doesn’t shrink; don’t overdo it.
    • feas_tol: start at 1e-3 for speed; tighten to 1e-5 at the end.
    • sys_choice: "lambda" if n≫m (TO); "x" if m≫n; otherwise "auto".

    The class does not parallelize user function evaluations because those functions are
    domain-specific, but it operates on vectorized gradients and can be wrapped in your
    own parallel evaluation (e.g., joblib) if computing fi and dfi is the bottleneck. user function evaluations because those functions are
    domain-specific, but it operates on vectorized gradients and can be wrapped in your
    own parallel evaluation (e.g., joblib) if computing fi and dfi is the bottleneck.
    """

    def __init__(
        self,
        n: int,
        m: int,
        xmin: Array,
        xmax: Array,
        a0: float = 1.0,
        a: Optional[Array] = None,
        c: Optional[Array] = None,
        d: Optional[Array] = None,
        *,
        raa0: float = 1e-5,
        asyinit: float = 0.5,
        asyincr: float = 1.2,
        asydecr: float = 0.7,
        albefa: float = 0.1,
        move: float = 0.45,
        ip_eps_min: float = 1e-7,
        ip_max_iter: int = 100,
        ip_tol_factor: float = 0.9,
        sys_choice: str = "auto",
        max_outer: int = 500,
        max_inner: int = 30,
        kkt_tol: float = 1e-5,
        feas_tol: float = 1e-3,
        verbose: bool = False,
        record_history: bool = True,
    ) -> None:
        self.n = int(n)
        self.m = int(m)
        self.xmin = np.asarray(xmin, dtype=float).copy()
        self.xmax = np.asarray(xmax, dtype=float).copy()
        assert self.xmin.shape == (n,) and self.xmax.shape == (n,)
        assert np.all(self.xmin < self.xmax)

        self.a0 = float(a0)
        self.a = np.zeros(m) if a is None else np.asarray(a, dtype=float).copy()
        self.c = (3e3 * np.ones(m)) if c is None else np.asarray(c, dtype=float).copy()
        self.d = np.ones(m) if d is None else np.asarray(d, dtype=float).copy()
        assert self.a.shape == (m,) and self.c.shape == (m,) and self.d.shape == (m,)

        # Parameters (MMA/GCMMA defaults per the paper)
        self.raa0 = float(raa0)
        self.asyinit = float(asyinit)
        self.asyincr = float(asyincr)
        self.asydecr = float(asydecr)
        self.albefa = float(albefa)
        self.move = float(move)

        # Interior-point solver parameters
        self.ip_eps_min = float(ip_eps_min)
        self.ip_max_iter = int(ip_max_iter)
        self.ip_tol_factor = float(ip_tol_factor)
        assert sys_choice in {"auto", "lambda", "x"}
        self.sys_choice = sys_choice

        # Outer/inner GCMMA control
        self.max_outer = int(max_outer)
        self.max_inner = int(max_inner)
        self.kkt_tol = float(kkt_tol)
        self.feas_tol = float(feas_tol)

        self.verbose = bool(verbose)
        self.record_history = bool(record_history)

        # Internal state for asymptotes (l,u) and x-history
        self._l = None  # (n,)
        self._u = None  # (n,)
        self._x_hist = []  # for asymptote update

        # History
        self.history: Dict[str, list] = {
            "x": [], "f0": [], "f": [], "kkt": [], "rho": [], "alpha": [], "beta": []
        } if record_history else {}

    # --------------------------- Public API ---------------------------
    def solve(
        self,
        x0: Array,
        f0: Callable[[Array], float],
        f: Callable[[Array], Array],
        df0: Callable[[Array], Array],
        df: Callable[[Array], Array],
        *,
        max_outer: Optional[int] = None,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> Tuple[Array, Dict[str, Any]]:
        """Run the GCMMA outer/inner iterations from x0.

        Parameters
        ----------
        x0 : (n,) array_like
            Initial design in [xmin, xmax].
        f0, f : callables
            f0(x)->scalar objective; f(x)->(m,) constraint values.
        df0, df : callables
            df0(x)->(n,) gradient; df(x)->(m,n) Jacobian (each row is grad fi).
        max_outer : int, optional
            Override of constructor value for this call.
        callback : callable, optional
            callback(k, state_dict) called each accepted outer iteration.

        Returns
        -------
        x_opt : (n,) ndarray
            Final design.
        info : dict
            Diagnostics and a compact history, including 'status' and 'outer_iters'.
        """
        n, m = self.n, self.m
        xk = np.clip(np.asarray(x0, dtype=float).copy(), self.xmin, self.xmax)
        if self._l is None:
            # Initialize asymptotes (Eq. 3.11, using asyinit)
            self._l = xk - self.asyinit * (self.xmax - self.xmin)
            self._u = xk + self.asyinit * (self.xmax - self.xmin)
        self._x_hist = [xk.copy()]

        max_outer = self.max_outer if max_outer is None else int(max_outer)

        # Rho (inner-loop parameters) – initialized each outer loop as per Eq. (4.6)
        rho = None  # (m+1,)

        # Bookkeeping
        status = "max_outer_reached"
        outer_iters = 0

        for k in range(1, max_outer + 1):
            outer_iters = k
            xk = self._x_hist[-1].copy()

            # Evaluate f0, f and their gradients at xk
            f0k = float(f0(xk))
            fk = np.asarray(f(xk), dtype=float).reshape(m)
            g0k = np.asarray(df0(xk), dtype=float).reshape(n)
            Gk = np.asarray(df(xk), dtype=float).reshape(m, n)

            # Initialize rho per Eq. (4.6)
            if rho is None:
                # For i=0..m
                def rho0_from_grad(gi: Array) -> float:
                    # gi is gradient vector of fi (length n)
                    term = np.abs(gi) * (self.xmax - self.xmin)
                    val = 0.1 / n * np.sum(term)
                    return max(val, 1e-6)
                rho_list = [rho0_from_grad(g0k)] + [rho0_from_grad(Gk[i]) for i in range(m)]
                rho = np.array(rho_list, dtype=float)

            # Update alpha/beta bounds (Eqs. 3.6–3.7)
            alpha, beta = self._compute_alpha_beta(xk)

            # Build the MMA/GCMMA approximation coefficients p,q,r (Eqs. 4.2–4.5)
            p, q, r = self._build_separable_approx(xk, f0k, fk, g0k, Gk, rho)

            # Inner GCMMA loop: make the approximation conservative at the solution
            accepted = False
            for nu in range(self.max_inner):
                # Solve subproblem (Eq. 4.1 / 5.1) with current p,q,r and (alpha,beta)
                subres = self._solve_subproblem(xk, p, q, r, alpha, beta)
                xhat = subres.x

                # Check conservativity ftilde_i(xhat) >= fi(xhat) for i=0..m
                fi_xhat = np.array([f0(xhat)] + list(f(xhat)), dtype=float)
                ftilde_xhat = self._ftilde_eval(xhat, p, q, r)
                viol = fi_xhat - ftilde_xhat  # if any > 0, not conservative

                if np.all(viol <= 0.0 + 1e-12):
                    # Accept outer iterate
                    accepted = True
                    xnext = xhat
                    break

                # Otherwise update rho using Eq. (4.7)-(4.9)
                dk_val = self._d_k(xhat)
                # Avoid division by tiny dk
                dk_val = max(dk_val, 1e-16)
                delta = viol / dk_val  # (m+1,)
                rho_new = rho.copy()
                for i in range(m + 1):
                    if delta[i] > 0.0:
                        rho_new[i] = min(1.1 * (rho[i] + delta[i]), 10.0 * rho[i])
                    # else unchanged
                rho = rho_new
                # Rebuild p,q,r with updated rho
                p, q, r = self._build_separable_approx(xk, f0k, fk, g0k, Gk, rho)

            if not accepted:
                # Fallback: accept the last inner solution even if not perfectly conservative.
                # This should be extremely rare with the prescribed updates.
                xnext = xhat

            # Update asymptotes (Eqs. 3.11–3.14)
            self._update_asymptotes(xnext)

            # Project onto [xmin,xmax] (safety)
            xnext = np.clip(xnext, self.xmin, self.xmax)
            self._x_hist.append(xnext.copy())

            # Compute KKT-like stopping measure on the original problem
            kkt = self._projected_kkt(xnext, f0, f, df0, df)
            feas = np.max(f(xnext))

            # Record history
            if self.record_history:
                self.history["x"].append(xnext.copy())
                self.history["f0"].append(float(f0(xnext)))
                self.history["f"].append(np.asarray(f(xnext), dtype=float).copy())
                self.history["kkt"].append(float(kkt))
                self.history["rho"].append(rho.copy())
                self.history["alpha"].append(alpha.copy())
                self.history["beta"].append(beta.copy())

            if self.verbose:
                print(f"[GCMMA] it={k:4d} f0={f0(xnext):.6e} max(f)={feas:.3e} kkt={kkt:.3e}")

            if callback is not None:
                callback(k, {
                    "x": xnext.copy(), "f0": float(f0(xnext)), "f": np.asarray(f(xnext)),
                    "kkt": float(kkt), "feas": float(feas)
                })

            # Check termination
            if kkt <= self.kkt_tol and feas <= self.feas_tol:
                status = "kkt_converged"
                break

            # Prepare rho for next outer loop: reset per Eq. (4.6) (fresh gradients)
            rho = None

        info = {
            "status": status,
            "outer_iters": outer_iters,
            "x": self._x_hist[-1].copy(),
        }
        if self.record_history:
            info["history"] = self.history
        return self._x_hist[-1].copy(), info

    # ----------------------- Core building blocks -----------------------
    def _compute_alpha_beta(self, xk: Array) -> Tuple[Array, Array]:
        # Eqs. (3.6)–(3.7)
        dx = self.xmax - self.xmin
        alpha = np.maximum.reduce([
            self.xmin,
            self._l + self.albefa * (xk - self._l),
            xk - self.move * dx,
        ])
        beta = np.minimum.reduce([
            self.xmax,
            self._u - self.albefa * (self._u - xk),
            xk + self.move * dx,
        ])
        # Ensure strict separation (interior-point solver uses logs)
        eps = 1e-12
        beta = np.maximum(beta, alpha + eps)
        return alpha, beta

    def _split_grad(self, g: Array) -> Tuple[Array, Array]:
        # Return positive and negative parts as defined under (3.3)–(3.4)
        gp = np.maximum(g, 0.0)
        gn = np.maximum(-g, 0.0)
        return gp, gn

    def _build_separable_approx(
        self,
        xk: Array,
        f0k: float,
        fk: Array,
        g0k: Array,
        Gk: Array,
        rho: Array,  # length m+1
    ) -> Tuple[Array, Array, Array]:
        """Build p_ij, q_ij, r_i for i=0..m, j=1..n (Eqs. 4.2–4.5).
        Returns p (m+1,n), q (m+1,n), r (m+1,).
        """
        n, m = self.n, self.m
        # Prepare arrays
        p = np.zeros((m + 1, n))
        q = np.zeros((m + 1, n))
        r = np.zeros(m + 1)

        dx = self.xmax - self.xmin
        umx = self._u - xk
        xml = xk - self._l
        umx_sq = umx**2
        xml_sq = xml**2

        # i = 0 (objective)
        gp, gn = self._split_grad(g0k)
        p[0] = umx_sq * (1.001 * gp + 0.001 * gn + (rho[0] / dx))
        q[0] = xml_sq * (0.001 * gp + 1.001 * gn + (rho[0] / dx))
        r[0] = f0k - np.sum(p[0] / np.maximum(umx, 1e-16) + q[0] / np.maximum(xml, 1e-16))

        # i = 1..m (constraints)
        for i in range(m):
            gp, gn = self._split_grad(Gk[i])
            p[i + 1] = umx_sq * (1.001 * gp + 0.001 * gn + (rho[i + 1] / dx))
            q[i + 1] = xml_sq * (0.001 * gp + 1.001 * gn + (rho[i + 1] / dx))
            r[i + 1] = fk[i] - np.sum(p[i + 1] / np.maximum(umx, 1e-16) + q[i + 1] / np.maximum(xml, 1e-16))

        return p, q, r

    def _ftilde_eval(self, x: Array, p: Array, q: Array, r: Array) -> Array:
        # Eq. (4.2): sum_j [ p_ij/(u_j - x_j) + q_ij/(x_j - l_j) ] + r_i
        denom_u = np.maximum(self._u - x, 1e-16)
        denom_l = np.maximum(x - self._l, 1e-16)
        return (p / denom_u + q / denom_l).sum(axis=1) + r

    def _d_k(self, x: Array) -> float:
        # Eq. (4.7): scalar d^(k)(x)
        num = (self._u - self._l) * (x - self._x_hist[-1])**2
        denom = (self._u - x) * (x - self._l) * (self.xmax - self.xmin)
        val = np.sum(num / np.maximum(denom, 1e-16))
        return float(val)

    def _update_asymptotes(self, xnew: Array) -> None:
        # Eqs. (3.11)–(3.14)
        n = self.n
        dx = self.xmax - self.xmin
        if len(self._x_hist) < 2:
            # already initialized in __init__
            xk = xnew
            self._l = xk - self.asyinit * dx
            self._u = xk + self.asyinit * dx
            return

        # For k >= 3
        xk = xnew
        xkm1 = self._x_hist[-1]
        xkm2 = self._x_hist[-2] if len(self._x_hist) >= 2 else xkm1

        prod = (xk - xkm1) * (xkm1 - xkm2)
        gamma = np.where(prod < 0, self.asydecr, np.where(prod > 0, self.asyincr, 1.0))

        l_new = xk - gamma * (xkm1 - self._l)
        u_new = xk + gamma * (self._u - xkm1)

        # Enforce Eq. (3.14)
        l_new = np.minimum(l_new, xk - 0.01 * dx)
        l_new = np.maximum(l_new, xk - 10.0 * dx)
        u_new = np.maximum(u_new, xk + 0.01 * dx)
        u_new = np.minimum(u_new, xk + 10.0 * dx)

        self._l = l_new
        self._u = u_new

    # ----------------------- Subproblem solver -------------------------
    def _solve_subproblem(
        self,
        xk: Array,
        p: Array,  # (m+1,n)
        q: Array,  # (m+1,n)
        r: Array,  # (m+1,)
        alpha: Array,
        beta: Array,
    ) -> SubproblemResult:
        """Solve the convex separable subproblem (5.1) via primal–dual IP (Section 5).
        Variables: x in [alpha,beta], y >= 0 (m,), z >= 0 (scalar), slack s >= 0 (m,),
        duals: lam (m,), xi (n,), eta (n,), mu (m,), zeta (scalar).

        The convex functions are gi(x) = sum_j (pij/(uj-xj) + qij/(xj-lj)), i=0..m.
        The subproblem notation in Sec. 5 uses g0 as the objective separable part and
        gi for constraints; we adapt the signs so that constraints are gi(x) - a_i z - y_i <= b_i
        where b_i = -r_i (Eq. 5.1 with r dropped from objective const part).
        """
        n, m = self.n, self.m
        a0, a, c, d = self.a0, self.a, self.c, self.d
        l, u = self._l, self._u

        # Precompute convenient accessors
        def psi_parts(lam: Array) -> Tuple[Array, Array]:
            # pj(λ) and qj(λ) for Eq. (5.4)-(5.5)
            # p0j + sum_i λ_i p_ij; q similarly. Here p[0] = for g0, p[1:] constraints.
            pj = p[0].copy()
            qj = q[0].copy()
            if m > 0:
                pj += lam @ p[1:]
                qj += lam @ q[1:]
            return pj, qj

        def g_vals(x: Array) -> Array:
            # gi(x) for i=0..m using current p,q,r (Eq. 5.2 with r folding)
            denom_u = np.maximum(u - x, 1e-16)
            denom_l = np.maximum(x - l, 1e-16)
            Gall = (p / denom_u + q / denom_l).sum(axis=1) + r  # (m+1,)
            return Gall

        # Build b = -r_i for constraints (Eq. 5.1)
        b = -r[1:].copy()  # (m,)

        # Initialize primal/dual variables per Sec. 5.5
        eps = 1.0
        x = 0.5 * (alpha + beta)
        y = np.ones(m)
        z = 1.0
        lam = np.ones(m)
        s = np.ones(m)
        xi = np.maximum(1.0, 1.0 / np.maximum(x - alpha, 1e-12))
        eta = np.maximum(1.0, 1.0 / np.maximum(beta - x, 1e-12))
        mu = np.maximum(1.0, 0.5 * c)
        zeta = 1.0

        # Helper closures for Newton linear system blocks (avoid materializing diagonals)
        def Dx_vec(lam: Array, x: Array, xi: Array, eta: Array) -> Array:
            pj, qj = psi_parts(lam)
            # (5.11) -> Ψ diagonals, and (5.15a)
            psi_dd = 2.0 * pj / np.maximum((u - x) ** 3, 1e-16) + \
                      2.0 * qj / np.maximum((x - l) ** 3, 1e-16)
            return psi_dd + xi / np.maximum(x - alpha, 1e-16) + \
                   eta / np.maximum(beta - x, 1e-16)

        def Dy_vec(y: Array, mu: Array) -> Array:
            # (5.15b)
            return d + mu / np.maximum(y, 1e-16)

        def Dlam_vec(lam: Array, s: Array) -> Array:
            # (5.15c)
            return s / np.maximum(lam, 1e-16)

        def partial_deriv_psi(lam: Array, x: Array) -> Array:
            # ∂ψ/∂x (vector, length n), Eq. (5.8) with pj(λ), qj(λ)
            pj, qj = psi_parts(lam)
            return pj / np.maximum((u - x) ** 2, 1e-16) - \
                   qj / np.maximum((x - l) ** 2, 1e-16)

        def G_mat(x: Array) -> Array:
            # (5.12) G_{i,j} = ∂g_i/∂x_j  for i=1..m (constraints only)
            denom_u = np.maximum((u - x) ** 2, 1e-16)
            denom_l = np.maximum((x - l) ** 2, 1e-16)
            return (p[1:] / denom_u) - (q[1:] / denom_l)  # (m,n)

        # Residuals for relaxed KKT (5.9a–i). We implement directly in vector form.
        def residuals(x, y, z, lam, xi, eta, mu, zeta, s, eps):
            # 5.9a
            r_a = partial_deriv_psi(lam, x) - xi + eta
            # 5.9b
            r_b = c + d * y - lam - mu
            # 5.9c
            r_c = a0 - zeta - float(np.dot(lam, a))
            # 5.9d: gi(x) - a_i z - y_i + s_i - b_i = 0
            gall = g_vals(x)
            gi = gall[1:]
            r_d = gi - a * z - y + s - b
            # 5.9e
            r_e = xi * (x - alpha) - eps
            # 5.9f
            r_f = eta * (beta - x) - eps
            # 5.9g
            r_g = mu * y - eps
            # 5.9h
            r_h = zeta * z - eps
            # 5.9i
            r_i = lam * s - eps
            return r_a, r_b, r_c, r_d, r_e, r_f, r_g, r_h, r_i

        # Infinity norm of concatenated residuals
        def res_infty(x, y, z, lam, xi, eta, mu, zeta, s, eps) -> float:
            r = residuals(x, y, z, lam, xi, eta, mu, zeta, s, eps)
            scalars = [np.array([r[2], r[7]])]  # r_c and r_h are scalars
            vecs = [r[0], r[1], r[3], r[4], r[5], r[6], r[8]]
            arr = np.concatenate([*(vecs), *scalars], axis=None)
            return float(np.max(np.abs(arr)))

        # Newton iterations with backtracking and ε reduction (Sec. 5.5)
        iters = 0
        while True:
            # Solve to tolerance for current epsilon
            newton_it = -1
            for newton_it in range(self.ip_max_iter):
                iters += 1
                # Shorthand
                dx_vec = Dx_vec(lam, x, xi, eta)
                dy_vec = Dy_vec(y, mu)
                dlam_vec = Dlam_vec(lam, s)
                G = G_mat(x)
                # (5.15d–g)
                dpsi_dx = partial_deriv_psi(lam, x)
                delta_x_tilde = dpsi_dx - (eps / np.maximum(x - alpha, 1e-16)) + \
                                (eps / np.maximum(beta - x, 1e-16))
                delta_y_tilde = c + d * y - lam - (eps / np.maximum(y, 1e-16))
                delta_z_tilde = a0 - float(np.dot(lam, a)) - eps / max(z, 1e-16)
                gall = g_vals(x)
                gi = gall[1:]
                delta_lam_tilde = gi - a * z - y - b + (eps / np.maximum(lam, 1e-16))

                # Eq. (5.17) after eliminating (xi,eta,mu,zeta,s) and y via (5.16)
                # We then eliminate x or lam depending on sys_choice.
                # Precompute Dy^{-1}
                inv_dy = 1.0 / np.maximum(dy_vec, 1e-16)
                Dlam_y = dlam_vec + inv_dy  # (5.18a) diagonal entries
                delta_lam_tilde_y = delta_lam_tilde + inv_dy * delta_y_tilde  # (5.18b)

                # Choose system (5.20) or (5.22)
                use_lambda_sys = self._choose_lambda_sys()
                if use_lambda_sys:
                    # (5.19) ∆x = -Dx^{-1} G^T ∆λ - Dx^{-1} δ̃x
                    inv_dx = 1.0 / np.maximum(dx_vec, 1e-16)
                    # Build (m+1)x(m+1) system (5.20)
                    # A = Dlam_y + G Dx^{-1} G^T, aug with z rows/cols via 'a' and ζ/z
                    # Build H = G Dx^{-1} G^T efficiently
                    # tmp = G * inv_dx (row-wise scaling of columns)
                    tmp = G * inv_dx  # (m,n)
                    H = tmp @ G.T  # (m,m)
                    A11 = (np.diag(Dlam_y) + H)
                    A12 = a.reshape(-1, 1)
                    A21 = a.reshape(1, -1)
                    A22 = np.array([[-zeta / max(z, 1e-16)]] )
                    # RHS
                    rhs1 = delta_lam_tilde_y - (G @ (inv_dx * delta_x_tilde))
                    rhs2 = np.array([delta_z_tilde])
                    # Assemble and solve
                    A = np.block([[A11, A12], [A21, A22]])
                    rhs = np.concatenate([rhs1, rhs2])
                    try:
                        sol = np.linalg.solve(A, rhs)
                    except np.linalg.LinAlgError:
                        # Regularize
                        reg = 1e-10
                        A = A + reg * np.eye(m + 1)
                        sol = np.linalg.solve(A, rhs)
                    dlam = sol[:m]
                    dz = sol[m]
                    dx = -inv_dx * (G.T @ dlam + delta_x_tilde)
                    dy = inv_dy * (dlam - delta_y_tilde)
                else:
                    # Build (n+1)x(n+1) system (5.22)
                    inv_Dlam_y = 1.0 / np.maximum(Dlam_y, 1e-16)
                    GT_invDlamG = G.T @ (inv_Dlam_y[:, None] * G)
                    A11 = np.diag(dx_vec) + GT_invDlamG
                    A12 = -G.T @ (inv_Dlam_y * a)
                    A21 = -A12.T
                    A22 = zeta / max(z, 1e-16) + float(a.T @ (inv_Dlam_y * a))
                    rhs1 = -delta_x_tilde - G.T @ (inv_Dlam_y * delta_lam_tilde_y)
                    rhs2 = -delta_z_tilde + float(a.T @ (inv_Dlam_y * delta_lam_tilde_y))
                    # Solve
                    try:
                        M = np.block([[A11, A12[:, None]], [A21[None, :], np.array([[A22]])]])
                        sol = np.linalg.solve(M, np.concatenate([rhs1, np.array([rhs2])]))
                    except np.linalg.LinAlgError:
                        reg = 1e-10
                        M = np.block([[A11 + reg * np.eye(n), A12[:, None]], [A21[None, :], np.array([[A22 + reg]])]])
                        sol = np.linalg.solve(M, np.concatenate([rhs1, np.array([rhs2])]))
                    dx = sol[:n]
                    dz = sol[n]
                    dlam = inv_Dlam_y * (G @ dx - a * dz + delta_lam_tilde_y)
                    inv_dy = 1.0 / np.maximum(dy_vec, 1e-16)  # recompute safety
                    dy = inv_dy * (dlam - delta_y_tilde)

                # Recover eliminated dual updates via (5.13)
                dxi = -(xi / np.maximum(x - alpha, 1e-16)) * dx - xi + eps / np.maximum(x - alpha, 1e-16)
                deta = (eta / np.maximum(beta - x, 1e-16)) * dx - eta + eps / np.maximum(beta - x, 1e-16)
                dmu = -(mu / np.maximum(y, 1e-16)) * dy - mu + eps / np.maximum(y, 1e-16)
                dzeta = -(zeta / max(z, 1e-16)) * dz - zeta + eps / max(z, 1e-16)
                ds = -(s / np.maximum(lam, 1e-16)) * dlam - s + eps / np.maximum(lam, 1e-16)

                # Step length to maintain positivity and box margins (Sec. 5.4)
                t = 1.0
                # x within (alpha,beta) with 1% margin
                t = min(t, self._fraction_to_boundary(x - alpha, dx))
                t = min(t, self._fraction_to_boundary(beta - x, -dx))
                # All positive variables with 1% margin
                t = min(t, self._fraction_to_boundary(y, dy))
                t = min(t, self._fraction_to_boundary(lam, dlam))
                t = min(t, self._fraction_to_boundary(xi, dxi))
                t = min(t, self._fraction_to_boundary(eta, deta))
                t = min(t, self._fraction_to_boundary(mu, dmu))
                t = min(t, self._fraction_to_boundary(s, ds))
                t = min(t, self._fraction_to_boundary(np.array([z]), np.array([dz])))
                t = min(t, self._fraction_to_boundary(np.array([zeta]), np.array([dzeta])))

                # Backtracking on residual norm
                res0 = res_infty(x, y, z, lam, xi, eta, mu, zeta, s, eps)
                tau = t
                # geometric backtracking until residual reduces
                while True:
                    xn = x + tau * dx
                    yn = y + tau * dy
                    zn = z + tau * dz
                    lmn = lam + tau * dlam
                    xin = xi + tau * dxi
                    etn = eta + tau * deta
                    mun = mu + tau * dmu
                    ztn = zeta + tau * dzeta
                    sn = s + tau * ds

                    # Ensure strict interior (prevent numerical issues)
                    if (xn <= alpha).any() or (xn >= beta).any() or \
                       (yn <= 0).any() or (lmn <= 0).any() or (xin <= 0).any() or \
                       (etn <= 0).any() or (mun <= 0).any() or (sn <= 0).any() or \
                       (zn <= 0) or (ztn <= 0):
                        tau *= 0.5
                        if tau < 1e-16:
                            break
                        continue

                    res1 = res_infty(xn, yn, zn, lmn, xin, etn, mun, ztn, sn, eps)
                    if res1 < res0:
                        x, y, z, lam, xi, eta, mu, zeta, s = xn, yn, zn, lmn, xin, etn, mun, ztn, sn
                        break
                    tau *= 0.5
                    if tau < 1e-16:
                        # Accept tiny step to avoid stalling
                        x, y, z, lam, xi, eta, mu, zeta, s = xn, yn, zn, lmn, xin, etn, mun, ztn, sn
                        break

                # Check termination for current ε
                res_inf = res_infty(x, y, z, lam, xi, eta, mu, zeta, s, eps)
                if res_inf < self.ip_tol_factor * eps:
                    # decrease epsilon (Step 4 Sec. 5.5)
                    eps *= 0.1
                    eps = max(eps, self.ip_eps_min)
                    if eps <= self.ip_eps_min:
                        break

            # If we hit min epsilon or max iterations, stop
            res_inf = res_infty(x, y, z, lam, xi, eta, mu, zeta, s, eps)
            if eps <= self.ip_eps_min or newton_it + 1 >= self.ip_max_iter:
                break

        return SubproblemResult(x=x, y=y, z=float(z), lam=lam, xi=xi, eta=eta,
                                mu=mu, zeta=float(zeta), s=s, iters=iters,
                                eps=float(eps), res_inf=float(res_inf))

    # ------------------------ Utilities & checks -----------------------
    def _choose_lambda_sys(self) -> bool:
        # True -> use (5.20) (lambda system), False -> use (5.22) (x system)
        if self.sys_choice == "lambda":
            return True
        if self.sys_choice == "x":
            return False
        # auto: prefer lambda system when n > m (Section 5.3 note)
        return self.n >= self.m

    @staticmethod
    def _fraction_to_boundary(x: Array, dx: Array, fraction: float = 0.99) -> float:
        # Largest t in (0,1] such that x + t*dx >= (1 - fraction)*x  (for positive x)
        # and similar for box distances (with x being distances). Implements the 1% rule.
        mask = dx < 0
        if not np.any(mask):
            return 1.0
        t = np.min(-fraction * x[mask] / dx[mask])
        return float(max(min(t, 1.0), 0.0))

    def _projected_kkt(
        self,
        x: Array,
        f0: Callable[[Array], float],
        f: Callable[[Array], Array],
        df0: Callable[[Array], Array],
        df: Callable[[Array], Array],
    ) -> float:
        """A conservative projected-KKT measure for the original problem (1.2).
        We compute a least-squares λ >= 0 solving ∇f0 + Σ λ_i ∇fi ≈ 0 in the interior,
        and then measure stationarity projected onto the box. This is not part of the
        GCMMA algorithm itself but provides a robust termination criterion in practice.
        """
        n, m = self.n, self.m
        g0 = np.asarray(df0(x), dtype=float).reshape(n)
        J = np.asarray(df(x), dtype=float).reshape(m, n)
        # Nonnegative least squares for λ: minimize ||g0 + J^T λ||^2 s.t. λ>=0
        # Solve via simple projected gradient (small m), or closed form if unconstrained
        # Start with unconstrained least-squares solution
        JT = J.T
        H = J @ JT + 1e-12 * np.eye(m)
        rhs = -J @ g0
        lam = np.linalg.solve(H, rhs)
        lam = np.maximum(lam, 0.0)
        gradL = g0 + JT @ lam
        # Project gradient onto feasible box
        proj = np.zeros_like(gradL)
        # If x near lower bound, only positive gradient is relevant; if near upper, only negative
        tol = 1e-10
        at_low = x <= self.xmin + tol
        at_up = x >= self.xmax - tol
        free = ~(at_low | at_up)
        proj[free] = gradL[free]
        proj[at_low] = np.minimum(gradL[at_low], 0.0)
        proj[at_up] = np.maximum(gradL[at_up], 0.0)
        return float(np.linalg.norm(proj, ord=np.inf))


# ------------------------------ Example ------------------------------
# The following example can be used for quick verification (similar to Section 6 in
# the note). Uncomment to run as a script.
#
# if __name__ == "__main__":
#     # Example: minimize x1^2 + x2^2 + x3^2 subject to two ball constraints and 0<=x<=5.
#     n, m = 3, 2
#     xmin = np.zeros(3)
#     xmax = 5*np.ones(3)
#     def f0(x):
#         return float(np.dot(x, x))
#     def df0(x):
#         return 2.0*x
#     def f(x):
#         c1 = (x[0]-5)**2 + (x[1]-2)**2 + (x[2]-1)**2 - 9
#         c2 = (x[0]-3)**2 + (x[1]-4)**2 + (x[2]-3)**2 - 9
#         return np.array([c1, c2])
#     def df(x):
#         g1 = 2.0*np.array([x[0]-5, x[1]-2, x[2]-1])
#         g2 = 2.0*np.array([x[0]-3, x[1]-4, x[2]-3])
#         return np.vstack([g1, g2])
#     g = GCMMA(n, m, xmin, xmax, a0=1.0, a=np.zeros(m), c=1000*np.ones(m), d=np.ones(m),
#               verbose=True)
#     x0 = np.array([4.0, 3.0, 2.0])
#     xopt, info = g.solve(x0, f0, f, df0, df, max_outer=50)
#     print("x* =", xopt)
#     print("status:", info["status"], "outer iters:", info["outer_iters"])
    

# =============================================================================
# Configuration — One place for **all** inputs (problem + solver)
# =============================================================================

@dataclass
class TOConfig:
    """Topology-optimization (SIMP) configuration [Sec. 2.1].  #  [oai_citation:7‡Optimisation_based_3D_Strut_n_Tie_Reinf_Conc.pdf](file-service://file-V14xVgnZPJpVSjkipxijwu)
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
    """STM shape-optimization configuration [Sec. 2.3].  #  [oai_citation:8‡Optimisation_based_3D_Strut_n_Tie_Reinf_Conc.pdf](file-service://file-V14xVgnZPJpVSjkipxijwu)
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
    """Top-level, single place holding **all** inputs (use case + solver).
    """
    to: TOConfig = field(default_factory=TOConfig)
    shape: ShapeConfig = field(default_factory=ShapeConfig)
    opt: OptimizerConfig = field(default_factory=OptimizerConfig)

# =============================================================================
# FE Back-end Interfaces
# =============================================================================

class ContinuumFESolver:
    """
    3D continuum FE back-end for SIMP TO [Sec. 2.1].
    - Matrix-free PCG
    - DOF pruning (remove DOFs linked only to near-void elements)
    - Parallelizable assembly via vectorized per-element matvec + scatter
    - Distance-based density filter (precomputed sparse rows) + transpose back-projection

    Parameters
    ----------
    edof : (ne, ndof_e) int
        Element -> global DOF connectivity.
    ke0 : (ndof_e, ndof_e) float
        Reference elemental stiffness (per unit Young's modulus) in local DOFs.
        K_e(ρ) = (Emin + ρ^p (E0 - Emin)) * ke0  (SIMP).
    f : (ndof,) float
        Global load vector.
    fixed_dofs : (nfix,) int
        Global DOF indices with Dirichlet BC (zero displacement).
    grid_shape : (nx, ny, nz)
        Voxel grid shape for elements (order must match `edof` element ordering).
    volumes : (ne,) float
        Element volumes (used for volume constraint and optional weighting).
    mesh_size : float
        Continuum element size (for filter radius).
    rho_min : float
        Floor density to keep K non-singular (default 1e-4).
    E0, Emin : float
        Solid and void Young’s moduli.
    simp_p : float
        SIMP penalization exponent (default 3.0).
    filter_radius_units : float
        Filter radius in multiples of mesh size (e.g., 2.5 for pile cap/box, 1.5 for corbel).
    rho0 : Optional[(ne,) float]
        Initial densities. If None, defaults to 0.5 everywhere.

    Notes
    -----
    - Compliance is f^T u, returned by `compliance(u)`.
    - `element_strain_energy(u)` returns (E0 - Emin) * u_e^T ke0 u_e  (to match paper’s dC/dρ).
    - Filtering uses the standard hat weights  w_ij = max(0, r - dist(i,j)), normalized per-row.
      Back-projection uses the transpose with the same normalization.
    """
    def __init__(
        self,
        edof: np.ndarray,
        ke0: np.ndarray,
        f: np.ndarray,
        fixed_dofs: np.ndarray,
        grid_shape: Tuple[int, int, int],
        volumes: np.ndarray,
        mesh_size: float,
        rho_min: float = 1e-4,
        E0: float = 1.0,
        Emin: float = 1e-9,
        simp_p: float = 3.0,
        filter_radius_units: float = 2.5,
        rho0: Optional[np.ndarray] = None,
    ) -> None:
        # --- store basic mesh/system data ---
        self.edof = np.asarray(edof, dtype=np.int64)
        self.ke0 = np.asarray(ke0, dtype=float)
        self.f_full = np.asarray(f, dtype=float)
        self._fixed = np.unique(np.asarray(fixed_dofs, dtype=np.int64))
        self._grid_shape = tuple(int(v) for v in grid_shape)
        self._volumes = np.asarray(volumes, dtype=float)
        self._mesh_size = float(mesh_size)

        self.ne, self.ndof_e = self.edof.shape
        self.ndof = self.f_full.size

        # --- material / SIMP ---
        self.E0 = float(E0)
        self.Emin = float(Emin)
        self.p = float(simp_p)
        self._rho_min = float(rho_min)

        # --- rho0 / bounds ---
        if rho0 is None:
            self._rho0 = 0.5 * np.ones(self.ne, dtype=float)
        else:
            self._rho0 = np.asarray(rho0, dtype=float).copy()
        self._rho_lb = self._rho_min * np.ones(self.ne, dtype=float)
        self._rho_ub = np.ones(self.ne, dtype=float)

        # --- precompute fixed/free maps (BC elimination) ---
        free_mask = np.ones(self.ndof, dtype=bool)
        free_mask[self._fixed] = False
        self._free_mask = free_mask
        self._free_dofs = np.nonzero(free_mask)[0]
        self._n_free = self._free_dofs.size

        # map global -> reduced index (-1 if fixed)
        self._g2r = -np.ones(self.ndof, dtype=np.int64)
        self._g2r[self._free_dofs] = np.arange(self._n_free, dtype=np.int64)

        # for each element, precompute which local dofs are free and their reduced indices;
        # also pre-extract ke0_ff blocks (masking fixed dofs) — depends only on BCs.
        self._elem_ridx: List[np.ndarray] = []
        self._ke0_ff: List[np.ndarray] = []
        for e in range(self.ne):
            gidx = self.edof[e]
            msk = self._free_mask[gidx]
            loc_free = np.nonzero(msk)[0]
            self._elem_ridx.append(self._g2r[gidx[loc_free]])
            if loc_free.size:
                self._ke0_ff.append(self.ke0[np.ix_(loc_free, loc_free)])
            else:
                # no free dofs in this element
                self._ke0_ff.append(np.zeros((0, 0), dtype=float))

        # --- filter precomputation (CSR-like rows for each element) ---
        self._r_phys = float(filter_radius_units) * self._mesh_size
        self._build_filter_rows()

    # ------------------------------------------------------------------
    # Public API expected by the driver
    # ------------------------------------------------------------------
    def assemble_K_and_solve(self, rho: np.ndarray) -> np.ndarray:
        """
        Solve K(ρ) u = f with:
        - DOF pruning: only free DOFs connected to 'active' elements (ρ > rho_min) are kept
        - Matrix-free PCG with Jacobi preconditioner
        Returns full u (including zeros at fixed DOFs).
        """
        rho = np.asarray(rho, dtype=float)
        assert rho.shape == (self.ne,)
        # --- determine active elements for pruning ---
        active = rho > (1.01 * self._rho_min)
        if not np.any(active):
            # degenerate; return zeros
            u_full = np.zeros(self.ndof, dtype=float)
            return u_full

        # active free DOFs = union of edof over active elements, then intersect with free set
        act_dofs = np.unique(self.edof[active].ravel())
        act_free_mask = np.zeros(self.ndof, dtype=bool)
        act_free_mask[self._free_dofs] = True
        act_free_mask &= np.isin(np.arange(self.ndof), act_dofs)
        red_keep = np.nonzero(act_free_mask[self._free_dofs])[0]  # indices into reduced space
        if red_keep.size == 0:
            u_full = np.zeros(self.ndof, dtype=float)
            return u_full

        # restrict right-hand side to active-free set
        f_red = self.f_full[self._free_dofs][red_keep]

        # convenience closures for per-element data on the active set
        active_elems = np.nonzero(active)[0]
        ridx_list = [self._elem_ridx[e] for e in active_elems]          # reduced indices (full free set)
        ke_ff_list = [self._ke0_ff[e] for e in active_elems]

        # further restrict per-element indices to the active-reduced subset
        # Build a map reduced_full -> reduced_active
        red_full_to_active = -np.ones(self._n_free, dtype=np.int64)
        red_full_to_active[red_keep] = np.arange(red_keep.size, dtype=np.int64)

        elem_aridx: List[np.ndarray] = []
        elem_ke_aff: List[np.ndarray] = []
        for ridx, kef in zip(ridx_list, ke_ff_list):
            if ridx.size == 0:
                elem_aridx.append(np.zeros(0, dtype=np.int64))
                elem_ke_aff.append(np.zeros((0, 0), dtype=float))
                continue
            mask = red_full_to_active[ridx] >= 0
            if not np.any(mask):
                elem_aridx.append(np.zeros(0, dtype=np.int64))
                elem_ke_aff.append(np.zeros((0, 0), dtype=float))
                continue
            aridx = red_full_to_active[ridx[mask]]
            elem_aridx.append(aridx)
            elem_ke_aff.append(kef[np.ix_(mask, mask)])

        # SIMP scaling per element
        scale = self.Emin + (rho[active_elems] ** self.p) * (self.E0 - self.Emin)

        # --- diagonal for Jacobi preconditioner ---
        diag = np.zeros(red_keep.size, dtype=float)
        for s, aridx, kef in zip(scale, elem_aridx, elem_ke_aff):
            if aridx.size:
                # add s * diag(kef) into diag at aridx
                np.add.at(diag, aridx, s * np.diag(kef))
        M_inv = 1.0 / (diag + 1e-30)

        # --- matrix-free matvec on the active-reduced space ---
        def matvec(x: np.ndarray) -> np.ndarray:
            y = np.zeros_like(x)
            for s, aridx, kef in zip(scale, elem_aridx, elem_ke_aff):
                if aridx.size:
                    xe = x[aridx]
                    ye = s * (kef @ xe)
                    np.add.at(y, aridx, ye)
            return y

        # --- PCG solver ---
        u_red = self._pcg(matvec, f_red, M_inv, tol=1e-8, maxit=500)

        # expand back to full vector
        u_free_full = np.zeros(self._n_free, dtype=float)
        u_free_full[red_keep] = u_red
        u_full = np.zeros(self.ndof, dtype=float)
        u_full[self._free_dofs] = u_free_full
        # fixed dofs assumed zero (Dirichlet)
        return u_full

    def element_strain_energy(self, u: np.ndarray) -> np.ndarray:
        """
        Return φ_e = (E0 - Emin) * u_e^T K_e^0 u_e  for each element, shape (ne,).
        (Matches paper’s dC/dρ_f formula used in the driver.)
        """
        ue = u[self.edof]                          # (ne, ndof_e)
        # batch: φ = (u_e ke0 u_e)  -> u_e @ (ke0 @ u_e^T) elementwise sum
        # compute v_e = ue @ ke0^T  then rowwise dot with ue
        v = ue @ self.ke0.T                        # (ne, ndof_e)
        phi = np.einsum('ij,ij->i', ue, v)         # (ne,)
        return (self.E0 - self.Emin) * phi

    def filter_densities(self, rho: np.ndarray) -> np.ndarray:
        """
        Hat filter: ρ_f[i] = Σ_j w_ij ρ[j] / Σ_j w_ij   with w_ij = max(0, r - dist(i,j))
        Implemented via precomputed CSR-like rows.
        """
        rho = np.asarray(rho, dtype=float)
        i_ptr, j_idx, w = self._f_ptr, self._f_idx, self._f_w
        out = np.empty_like(rho)
        for i in range(self.ne):
            s = 0.0
            a = 0.0
            beg, end = i_ptr[i], i_ptr[i+1]
            js = j_idx[beg:end]
            ws = w[beg:end]
            a = np.dot(ws, rho[js])
            s = np.sum(ws)
            out[i] = a / (s + 1e-30)
        return out

    def backproject_sensitivities(self, sens_filtered: np.ndarray) -> np.ndarray:
        """
        Transpose of the filter: dC/dρ = W^T ( dC/dρ_f / row_sum )
        Using the same hat weights and per-row sums as in filter_densities.
        """
        g = np.zeros(self.ne, dtype=float)
        i_ptr, j_idx, w = self._f_ptr, self._f_idx, self._f_w
        for i in range(self.ne):
            beg, end = i_ptr[i], i_ptr[i+1]
            js = j_idx[beg:end]
            ws = w[beg:end]
            s = np.sum(ws) + 1e-30
            coeff = sens_filtered[i] / s
            # scatter-add: for each neighbor j, add ws[j]*coeff
            np.add.at(g, js, ws * coeff)
        # clamp gradient where at bounds (optional, leave clean for GCMMA)
        return g

    # Optional helper for the driver
    def compliance(self, u: np.ndarray) -> float:
        """Compliance C = f^T u."""
        return float(np.dot(self.f_full, u))

    # --- properties expected by the driver ---
    @property
    def rho_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._rho_lb, self._rho_ub

    @property
    def rho0(self) -> np.ndarray:
        return self._rho0

    @property
    def volumes(self) -> np.ndarray:
        return self._volumes

    @property
    def mesh_size(self) -> float:
        return self._mesh_size

    # ------------------------------------------------------------------
    # Internals: filter rows & PCG
    # ------------------------------------------------------------------
    def _build_filter_rows(self) -> None:
        """Precompute CSR-like rows for hat filter on a regular voxel grid."""
        nx, ny, nz = self._grid_shape
        h = self._mesh_size
        r = self._r_phys
        rad = int(np.floor(r / h))
        # centers in grid coordinates (integer indices)
        # Build row by row
        ptr = [0]
        idx = []
        wts = []
        for e in range(self.ne):
            i, j, k = np.unravel_index(e, (nx, ny, nz), order='C')
            i0, i1 = max(0, i - rad), min(nx - 1, i + rad)
            j0, j1 = max(0, j - rad), min(ny - 1, j + rad)
            k0, k1 = max(0, k - rad), min(nz - 1, k + rad)
            for ii in range(i0, i1 + 1):
                di = (ii - i) * h
                for jj in range(j0, j1 + 1):
                    dj = (jj - j) * h
                    for kk in range(k0, k1 + 1):
                        dk = (kk - k) * h
                        dist = np.sqrt(di*di + dj*dj + dk*dk)
                        w = r - dist
                        if w > 0.0:
                            j_lin = np.ravel_multi_index((ii, jj, kk), (nx, ny, nz), order='C')
                            idx.append(j_lin)
                            wts.append(w)
            ptr.append(len(idx))
        self._f_ptr = np.asarray(ptr, dtype=np.int64)
        self._f_idx = np.asarray(idx, dtype=np.int64)
        self._f_w = np.asarray(wts, dtype=float)

    @staticmethod
    def _pcg(A, b, M_inv, tol=1e-8, maxit=500):
        """
        Preconditioned Conjugate Gradient for SPD systems.
        A : callable(x) -> y
        b : (n,)
        M_inv : (n,) Jacobi preconditioner (approx A^{-1} diagonal)
        """
        x = np.zeros_like(b)
        r = b - A(x)
        z = M_inv * r
        p = z.copy()
        rz_old = float(np.dot(r, z))
        b_norm = max(1.0, float(np.linalg.norm(b)))
        for k in range(maxit):
            Ap = A(p)
            alpha = rz_old / (float(np.dot(p, Ap)) + 1e-30)
            x += alpha * p
            r -= alpha * Ap
            if np.linalg.norm(r) / b_norm < tol:
                break
            z = M_inv * r
            rz_new = float(np.dot(r, z))
            beta = rz_new / (rz_old + 1e-30)
            p = z + beta * p
            rz_old = rz_new
        return x

class BeamFESolver:
    """Abstract interface for beam FE used in Shape step [Sec. 2.3].  #  [oai_citation:10‡Optimisation_based_3D_Strut_n_Tie_Reinf_Conc.pdf](file-service://file-V14xVgnZPJpVSjkipxijwu)
    Implements a 3D beam model and exposes equilibrium forces (N, V1, V2).

    Expected responsibilities:
    - assemble_K_and_solve(x) -> u
    - element_forces(u) -> tuple(N, V1, V2) per element
    - element_lengths(x) -> Le per element
    - compliance(u) -> f^T u (or compute via K u)
    - provide bounds for node coordinates and initial x0
    """
    def __init__(self, n_vars:int, mesh_size:float) -> None: ...
    def assemble_K_and_solve(self, x: np.ndarray) -> np.ndarray: ...
    def element_forces(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def element_lengths(self, x: np.ndarray) -> np.ndarray: ...
    def compliance(self, u: np.ndarray) -> float: ...
    @property
    def x_bounds(self) -> Tuple[np.ndarray, np.ndarray]: ...
    @property
    def x0(self) -> np.ndarray: ...
    @property
    def mesh_size(self) -> float: ...

# =============================================================================
# Drivers (implemented): wrap FE back-ends and expose (f0, df0, f, df)
# =============================================================================

class TOProblemDriver:
    """Builds GCMMA callbacks for SIMP compliance + volume constraint [Sec. 2.1].  #  [oai_citation:11‡Optimisation_based_3D_Strut_n_Tie_Reinf_Conc.pdf](file-service://file-V14xVgnZPJpVSjkipxijwu)

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


class ShapeProblemDriver:
    """Builds GCMMA callbacks for STM shape optimization [Sec. 2.3].  #  [oai_citation:12‡Optimisation_based_3D_Strut_n_Tie_Reinf_Conc.pdf](file-service://file-V14xVgnZPJpVSjkipxijwu)

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
            # Gradient of compliance via central FD (cheap with beam models; paper’s choice).
            # Consider parallelizing across coordinates if this becomes a bottleneck.
            return _central_fd_grad(f0, x.copy())

        def df(x: np.ndarray) -> np.ndarray:
            # 2 constraints (STS and p-norm length), central FD Jacobian (m x n)
            return _central_fd_jac(f, x.copy(), m=2)

        return f0, df0, f, df, x0, xmin, xmax

# =============================================================================
# Extraction (skeleton): threshold → thinning → graph (3D)  [Sec. 2.2]
# =============================================================================

class TopologyExtractor3D:
    """Converts voxel densities to a truss-like graph (nodes + straight bars).  #  [oai_citation:13‡Optimisation_based_3D_Strut_n_Tie_Reinf_Conc.pdf](file-service://file-V14xVgnZPJpVSjkipxijwu)

    Pipeline (placeholders):
    1) Thresholding: set density≥τ (≈0.1) → 1, else 0.
    2) Skeletonization (3D thinning): preserve Euler characteristic & connectivity (Lee et al., 1994).
    3) Node detection: voxels with >2 solid neighbors in 26-neighborhood are nodes (allow clusters).
    4) Connection tracing: recursively follow skeleton voxels from each node to next node to form bars.
    5) Merge short bars: replace bars shorter than L_merge with a new node at mid-point.
    """
    def __init__(self, threshold: float = 0.1, merge_length: float = 0.0) -> None:
        self.threshold = float(threshold)
        self.merge_length = float(merge_length)

    def threshold_voxels(self, rho_voxels: np.ndarray) -> np.ndarray:
        """Return binary voxels (1/0) using self.threshold."""
        raise NotImplementedError

    def skeletonize(self, voxels01: np.ndarray) -> np.ndarray:
        """Return 1-voxel-thick skeleton preserving topology (3D thinning)."""
        raise NotImplementedError

    def detect_nodes(self, skeleton01: np.ndarray) -> np.ndarray:
        """Return array of node coordinates (x,y,z)."""
        raise NotImplementedError

    def trace_connections(self, skeleton01: np.ndarray, nodes_xyz: np.ndarray) -> List[Tuple[int, int]]:
        """Return list of (i_node, j_node) connections (bars)."""
        raise NotImplementedError

    def merge_short_bars(self, nodes_xyz: np.ndarray, bars: List[Tuple[int, int]]) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
        """Return simplified (nodes, bars) after merging sub-L_merge bars."""
        raise NotImplementedError

# =============================================================================
# Visualisation (skeleton) — keep all plotting/exports in one place
# =============================================================================

class ResultsVisualizer:
    """Plotters and exporters for TO fields, extracted graphs, STM forces, convergence, etc.
    Methods should accept data-only inputs (numpy arrays / simple dicts) and never mutate state.
    """
    def plot_to_density(self, rho_3d: np.ndarray) -> None:
        """Volume rendering / slices of density field."""
        raise NotImplementedError

    def plot_extracted_graph(self, nodes_xyz: np.ndarray, bars: List[Tuple[int,int]], forces: Optional[np.ndarray] = None) -> None:
        """3D line plot of truss-like graph; optional color/width by force."""
        raise NotImplementedError

    def plot_convergence(self, history: Dict[str, list]) -> None:
        """Objective and max-constraint vs iteration, etc."""
        raise NotImplementedError

    def export_stm_to_cad(self, nodes_xyz: np.ndarray, bars: List[Tuple[int,int]], path: str) -> None:
        """Export to a CAD-friendly format."""
        raise NotImplementedError

# =============================================================================
# Orchestrator (skeleton): stitches steps 1–3 and calls GCMMA via drivers
# =============================================================================

class OPTSTMOrchestrator:
    """High-level pipeline:
        run_to()  -> densities
        extract() -> (nodes, bars)
        run_shape()-> optimized nodes
    Keeps all configs in one ProblemConfig instance.
    """
    def __init__(self, cfg: ProblemConfig, fe_to: ContinuumFESolver, fe_shape: BeamFESolver) -> None:
        self.cfg = cfg
        self.fe_to = fe_to
        self.fe_shape = fe_shape
        self.extractor = TopologyExtractor3D()
        self.viz = ResultsVisualizer()

    def run_to(self) -> np.ndarray:
        """Run TO via GCMMA (SIMP compliance + volume constraint)."""
        driver = TOProblemDriver(self.fe_to, self.cfg)
        f0, df0, f, df, x0, xmin, xmax = driver.build_gcmma_callbacks()

        n = x0.size; m = 1
        solver = GCMMA(n, m, xmin, xmax,
                       a0=1.0, a=np.zeros(m), c=self.cfg.opt.c_default*np.ones(m), d=self.cfg.opt.d_default*np.ones(m),
                       move=self.cfg.opt.move, asyinit=self.cfg.opt.asyinit, asyincr=self.cfg.opt.asyincr, asydecr=self.cfg.opt.asydecr,
                       feas_tol=self.cfg.opt.feas_tol, kkt_tol=self.cfg.opt.kkt_tol,
                       max_outer=self.cfg.opt.max_outer, max_inner=self.cfg.opt.max_inner,
                       sys_choice="lambda", verbose=self.cfg.opt.verbose)
        rho_opt, info = solver.solve(x0, f0, f, df0, df)
        return rho_opt

    def extract(self, rho_voxels: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
        """Threshold + skeletonize + build truss-like graph (placeholders)."""
        vox01 = self.extractor.threshold_voxels(rho_voxels)
        skel01 = self.extractor.skeletonize(vox01)
        nodes = self.extractor.detect_nodes(skel01)
        bars = self.extractor.trace_connections(skel01, nodes)
        nodes2, bars2 = self.extractor.merge_short_bars(nodes, bars)
        return nodes2, bars2

    def run_shape(self) -> np.ndarray:
        """Run shape optimization (beam FE) via GCMMA: compliance + {STS, Lmin}."""
        driver = ShapeProblemDriver(self.fe_shape, self.cfg)
        f0, df0, f, df, x0, xmin, xmax = driver.build_gcmma_callbacks()

        n = x0.size; m = 2
        move = 0.3  # conservative default for shape (paper) — or pass from cfg.opt.move
        solver = GCMMA(n, m, xmin, xmax,
                       a0=1.0, a=np.zeros(m), c=self.cfg.opt.c_default*np.ones(m), d=self.cfg.opt.d_default*np.ones(m),
                       move=move, asyinit=0.3, asyincr=self.cfg.opt.asyincr, asydecr=self.cfg.opt.asydecr,
                       feas_tol=self.cfg.opt.feas_tol, kkt_tol=self.cfg.opt.kkt_tol,
                       max_outer=min(self.cfg.opt.max_outer, 150), max_inner=self.cfg.opt.max_inner,
                       sys_choice=self.cfg.opt.sys_choice, verbose=self.cfg.opt.verbose)
        x_opt, info = solver.solve(x0, f0, f, df0, df)
        return x_opt
    
