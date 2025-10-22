"""
Continuum FE Solver for Topology Optimization.

This module implements a 3D continuum FE back-end for SIMP topology optimization
with matrix-free PCG solver, DOF pruning, and density filtering.

References: Section 2.1 of the paper
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


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
        Solid and void Young's moduli.
    simp_p : float
        SIMP penalization exponent (default 3.0).
    filter_radius_units : float
        Filter radius in multiples of mesh size (e.g., 2.5 for pile cap/box, 1.5 for corbel).
    rho0 : Optional[(ne,) float]
        Initial densities. If None, defaults to 0.5 everywhere.

    Notes
    -----
    - Compliance is f^T u, returned by `compliance(u)`.
    - `element_strain_energy(u)` returns (E0 - Emin) * u_e^T ke0 u_e  (to match paper's dC/dρ).
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
        (Matches paper's dC/dρ_f formula used in the driver.)
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
