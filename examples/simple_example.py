"""
Simple Example: Running OPT-STM Generator
==========================================

This example demonstrates a basic workflow for the OPT-STM optimization pipeline.

Workflow:
1. Define problem geometry, supports, and loads
2. Run topology optimization (TO) to get density field
3. Extract strut-and-tie model from density field
4. Run shape optimization on the extracted STM
5. Visualize and export results

Author: OPT-STM Project
Date: October 21, 2025
"""

import sys
from pathlib import Path

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.core import (
    ProblemConfig,
    TOConfig,
    ShapeConfig,
    OptimizerConfig,
    ContinuumFESolver,
    BeamFESolver
)
from src.orchestrator import OPTSTMOrchestrator


def create_simple_beam_problem():
    """
    Create a simple cantilever beam problem.
    
    Geometry:
    - Length: 1000 mm (X direction)
    - Height: 400 mm (Y direction)  
    - Width: 200 mm (Z direction)
    
    Boundary Conditions:
    - Fixed at X=0
    - Point load at (1000, 200, 100) in -Y direction
    
    Returns
    -------
    ProblemConfig
        Configuration for the optimization problem
    """
    
    # Create configuration
    to_config = TOConfig(
        vol_frac=0.10,          # Use 10% of material volume
        simp_p=3.0,             # SIMP penalty parameter
        rho_min=1e-4,           # Minimum density (for numerical stability)
        filter_radius=2.5       # Density filter radius (in brick units)
    )
    
    shape_config = ShapeConfig(
        sts_epsilon=0.05,       # STS constraint tolerance
        min_length_p=10,        # p-norm parameter for minimum length
        min_length_target=50.0, # Minimum bar length (mm)
        fd_step_rel=1e-3        # Finite difference step (relative to mesh size)
    )
    
    opt_config = OptimizerConfig(
        move=0.45,              # Move limit for design variables
        asyinit=0.5,           # Initial asymptote distance
        max_outer=500,          # Maximum outer iterations
        max_inner=100,          # Maximum inner iterations
        feas_tol=1e-3,         # Feasibility tolerance
        kkt_tol=1e-3,          # KKT tolerance
        verbose=True            # Print progress
    )
    
    config = ProblemConfig(
        to=to_config,
        shape=shape_config,
        opt=opt_config
    )
    
    return config


def create_continuum_fe_solver():
    """
    Create a simple continuum FE solver for topology optimization.
    
    NOTE: This is a placeholder - you need to implement the actual
    mesh generation, BC application, and force vector from your
    problem_input_definition.py module.
    
    Returns
    -------
    ContinuumFESolver
        FE solver initialized with problem geometry
    """
    
    print("\n" + "="*70)
    print("STEP 1: Create Continuum FE Solver")
    print("="*70)
    print("\n⚠️  NOTE: You need to use problem_input_definition.py to:")
    print("   1. Define geometry (positive/negative volumes)")
    print("   2. Create 3D brick mesh")
    print("   3. Apply boundary conditions (supports)")
    print("   4. Apply loads")
    print("   5. Initialize ContinuumFESolver with this data\n")
    
    # Placeholder - replace with actual solver from your problem definition
    # fe_solver = ContinuumFESolver(...)
    
    raise NotImplementedError(
        "Please integrate with problem_input_definition.py to create "
        "the actual FE solver with your geometry, BCs, and loads."
    )


def create_beam_fe_solver():
    """
    Create a beam FE solver for shape optimization.
    
    NOTE: This requires implementing the BeamFESolver subclass
    with your 3D beam element formulation.
    
    Returns
    -------
    BeamFESolver
        Beam FE solver for extracted STM
    """
    
    print("\n" + "="*70)
    print("STEP 3: Create Beam FE Solver")
    print("="*70)
    print("\n⚠️  NOTE: You need to implement BeamFESolver subclass with:")
    print("   1. 3D beam element assembly")
    print("   2. Element force computation")
    print("   3. Element length computation")
    print("   4. Compliance calculation\n")
    
    raise NotImplementedError(
        "Please implement a concrete BeamFESolver subclass for your "
        "3D beam elements."
    )


def main():
    """
    Main execution function - demonstrates the complete workflow.
    """
    
    print("\n" + "="*70)
    print("OPT-STM GENERATOR - Simple Example")
    print("="*70)
    print("\nThis example shows the basic workflow structure.")
    print("You need to integrate with your existing problem definition.")
    
    # Step 1: Create configuration
    print("\n" + "="*70)
    print("STEP 0: Create Problem Configuration")
    print("="*70)
    config = create_simple_beam_problem()
    print("\n✅ Configuration created:")
    print(f"   - Volume fraction: {config.to.vol_frac*100}%")
    print(f"   - SIMP penalty: {config.to.simp_p}")
    print(f"   - Max iterations: {config.opt.max_outer}")
    
    # Step 2: Create FE solvers (you need to implement these)
    try:
        fe_to = create_continuum_fe_solver()
        fe_shape = create_beam_fe_solver()
        
        # Step 3: Create orchestrator
        print("\n" + "="*70)
        print("STEP 4: Initialize Orchestrator")
        print("="*70)
        orchestrator = OPTSTMOrchestrator(config, fe_to, fe_shape)
        print("✅ Orchestrator initialized")
        
        # Step 4: Run topology optimization
        print("\n" + "="*70)
        print("STEP 5: Run Topology Optimization")
        print("="*70)
        rho_opt = orchestrator.run_to()
        print(f"✅ TO complete - optimized density field shape: {rho_opt.shape}")
        
        # Step 5: Extract STM
        print("\n" + "="*70)
        print("STEP 6: Extract Strut-and-Tie Model")
        print("="*70)
        nodes, bars = orchestrator.extract(rho_opt)
        print(f"✅ Extraction complete:")
        print(f"   - Nodes: {len(nodes)}")
        print(f"   - Bars: {len(bars)}")
        
        # Step 6: Run shape optimization
        print("\n" + "="*70)
        print("STEP 7: Run Shape Optimization")
        print("="*70)
        x_opt = orchestrator.run_shape()
        print(f"✅ Shape optimization complete - optimized coordinates shape: {x_opt.shape}")
        
        # Step 7: Visualize (placeholder)
        print("\n" + "="*70)
        print("STEP 8: Visualize Results")
        print("="*70)
        print("⚠️  Visualization methods need implementation in ResultsVisualizer")
        # orchestrator.viz.plot_to_density(rho_opt)
        # orchestrator.viz.plot_extracted_graph(nodes, bars)
        
        print("\n" + "="*70)
        print("✅ COMPLETE!")
        print("="*70)
        
    except NotImplementedError as e:
        print(f"\n❌ {e}")
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("\n1. Use problem_input_definition.py to define your geometry:")
        print("   - Create Excel template: problem.generate_excel_template()")
        print("   - Fill in geometry, supports, loads")
        print("   - Load: problem.load_from_excel('your_file.xlsx')")
        print("   - Visualize: problem.visualize_3d()")
        print("\n2. Create ContinuumFESolver from problem data:")
        print("   - Extract mesh, BCs, loads from problem")
        print("   - Initialize ContinuumFESolver")
        print("\n3. Implement BeamFESolver subclass for 3D beam elements")
        print("\n4. Run this example again!")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
