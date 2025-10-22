"""High-level orchestrator for the complete OPT STM workflow.

This module provides the OPTSTMOrchestrator class which coordinates the three-stage
optimization pipeline: topology optimization → extraction → shape optimization.

Workflow
--------
1. run_to()     : Topology optimization (SIMP) → density field
2. extract()    : Density → discrete graph (nodes, bars)
3. run_shape()  : Shape optimization (beam) → refined STM geometry

References
----------
Complete pipeline integrating all optimization phases as described in project
documentation sections 2.1-2.3.
"""

import numpy as np
from typing import Tuple, List

from src.core import (
    GCMMA,
    ProblemConfig,
    ContinuumFESolver,
    BeamFESolver
)
from src.optimization import TOProblemDriver, ShapeProblemDriver
from src.extraction import TopologyExtractor3D
from src.postprocessing import ResultsVisualizer


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
