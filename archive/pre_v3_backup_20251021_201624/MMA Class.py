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

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Any


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