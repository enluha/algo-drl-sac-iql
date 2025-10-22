"""
Beam FE Solver Interface for Shape Optimization.

This module defines the abstract interface for beam FE used in the STM shape
optimization step.

References: Section 2.3 of the paper
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


class BeamFESolver:
    """Abstract interface for beam FE used in Shape step [Sec. 2.3].
    
    Implements a 3D beam model and exposes equilibrium forces (N, V1, V2).

    Expected responsibilities:
    - assemble_K_and_solve(x) -> u
    - element_forces(u) -> tuple(N, V1, V2) per element
    - element_lengths(x) -> Le per element
    - compliance(u) -> f^T u (or compute via K u)
    - provide bounds for node coordinates and initial x0
    
    Parameters
    ----------
    n_vars : int
        Number of design variables (typically 3 * n_nodes for x,y,z coordinates).
    mesh_size : float
        Characteristic mesh size from continuum phase (used for normalization).
    """
    
    def __init__(self, n_vars: int, mesh_size: float) -> None:
        """Initialize beam FE solver.
        
        Subclasses should implement full beam element assembly.
        """
        self._n_vars = int(n_vars)
        self._mesh_size = float(mesh_size)
    
    def assemble_K_and_solve(self, x: np.ndarray) -> np.ndarray:
        """Assemble stiffness matrix K(x) and solve K u = f.
        
        Parameters
        ----------
        x : np.ndarray
            Design variables (node coordinates).
        
        Returns
        -------
        u : np.ndarray
            Displacement vector.
        """
        raise NotImplementedError("Subclass must implement assemble_K_and_solve")
    
    def element_forces(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute element internal forces from displacement solution.
        
        Parameters
        ----------
        u : np.ndarray
            Global displacement vector.
        
        Returns
        -------
        N : np.ndarray
            Axial forces per element.
        V1 : np.ndarray
            Shear forces (direction 1) per element.
        V2 : np.ndarray
            Shear forces (direction 2) per element.
        """
        raise NotImplementedError("Subclass must implement element_forces")
    
    def element_lengths(self, x: np.ndarray) -> np.ndarray:
        """Compute element lengths from node coordinates.
        
        Parameters
        ----------
        x : np.ndarray
            Design variables (node coordinates).
        
        Returns
        -------
        Le : np.ndarray
            Element lengths.
        """
        raise NotImplementedError("Subclass must implement element_lengths")
    
    def compliance(self, u: np.ndarray) -> float:
        """Compute structural compliance C = f^T u.
        
        Parameters
        ----------
        u : np.ndarray
            Displacement vector.
        
        Returns
        -------
        C : float
            Compliance value.
        """
        raise NotImplementedError("Subclass must implement compliance")
    
    @property
    def x_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bounds for design variables.
        
        Returns
        -------
        xmin : np.ndarray
            Lower bounds on coordinates.
        xmax : np.ndarray
            Upper bounds on coordinates.
        """
        raise NotImplementedError("Subclass must implement x_bounds property")
    
    @property
    def x0(self) -> np.ndarray:
        """Get initial design variables.
        
        Returns
        -------
        x0 : np.ndarray
            Initial node coordinates.
        """
        raise NotImplementedError("Subclass must implement x0 property")
    
    @property
    def mesh_size(self) -> float:
        """Get characteristic mesh size.
        
        Returns
        -------
        mesh_size : float
            Mesh size from continuum phase.
        """
        return self._mesh_size
