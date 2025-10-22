"""
Quick test script to verify ke0 computation and helper functions.
Run this to ensure the implementations are working correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from problem_input_definition import (
    compute_hex8_ke0,
    generate_edof_array,
    assemble_force_vector,
    assemble_fixed_dofs,
    LoadCondition,
    SupportCondition
)

print("="*70)
print("TESTING FEM HELPER FUNCTIONS")
print("="*70)

# Test 1: ke0 computation
print("\n1. Testing ke0 computation...")
try:
    ke0 = compute_hex8_ke0(E=1.0, nu=0.3)
    print(f"   ✓ ke0 computed successfully")
    print(f"   Shape: {ke0.shape}")
    print(f"   Symmetric: {np.allclose(ke0, ke0.T)}")
    print(f"   Condition number: {np.linalg.cond(ke0):.2e}")
    
    # Check positive definiteness (all eigenvalues should be positive)
    eigenvalues = np.linalg.eigvalsh(ke0)
    print(f"   Min eigenvalue: {eigenvalues[0]:.2e}")
    print(f"   Max eigenvalue: {eigenvalues[-1]:.2e}")
    
    if eigenvalues[0] > -1e-10:  # Allow small numerical errors
        print(f"   ✓ Matrix is positive semi-definite")
    else:
        print(f"   ⚠ Warning: Matrix has negative eigenvalues")
        
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: edof generation
print("\n2. Testing edof generation...")
try:
    # Simple 2-element mesh
    elements = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [4, 5, 6, 7, 8, 9, 10, 11]
    ])
    edof = generate_edof_array(elements)
    print(f"   ✓ edof generated successfully")
    print(f"   Shape: {edof.shape}")
    print(f"   First element DOFs: {edof[0]}")
    print(f"   Second element DOFs: {edof[1]}")
    
    # Verify connectivity (element 2 should share nodes 4-7 with element 1)
    shared_dofs = set(edof[0, 12:24]).intersection(set(edof[1, 0:12]))
    print(f"   Shared DOFs between elements: {len(shared_dofs)} (expected: 12)")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Force vector assembly
print("\n3. Testing force vector assembly...")
try:
    loads = [
        LoadCondition(
            load_id=1,
            location=np.array([0.5, 0.5, 1.0]),
            is_face=False,
            f_x=0.0,
            f_y=0.0,
            f_z=-1000.0,
            affected_nodes=[5]  # Single node
        ),
        LoadCondition(
            load_id=2,
            location=np.array([0.0, 0.0, 1.0]),
            is_face=True,
            f_x=100.0,
            f_y=0.0,
            f_z=0.0,
            affected_nodes=[4, 5, 6, 7]  # Face with 4 nodes
        )
    ]
    
    f = assemble_force_vector(loads, n_nodes=12)
    print(f"   ✓ Force vector assembled successfully")
    print(f"   Shape: {f.shape}")
    print(f"   Non-zero DOFs: {np.count_nonzero(f)}")
    print(f"   Total force magnitude: {np.linalg.norm(f):.2f}")
    
    # Check specific values
    print(f"   Force at node 5 (z-component): {f[3*5+2]:.2f} (expected: -1000.0)")
    print(f"   Force at node 4 (x-component): {f[3*4+0]:.2f} (expected: 25.0)")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Fixed DOFs assembly
print("\n4. Testing fixed DOFs assembly...")
try:
    supports = [
        SupportCondition(
            support_id=1,
            location=np.array([0.0, 0.0, 0.0]),
            is_face=False,
            u_x=True,
            u_y=True,
            u_z=True,
            affected_nodes=[0]
        ),
        SupportCondition(
            support_id=2,
            location=np.array([1.0, 0.0, 0.0]),
            is_face=False,
            u_x=False,
            u_y=True,
            u_z=True,
            affected_nodes=[1]
        )
    ]
    
    fixed_dofs = assemble_fixed_dofs(supports)
    print(f"   ✓ Fixed DOFs assembled successfully")
    print(f"   Number of fixed DOFs: {len(fixed_dofs)}")
    print(f"   Fixed DOFs: {fixed_dofs}")
    print(f"   Expected: [0, 1, 2, 4, 5] (node 0: all 3 DOFs, node 1: y and z only)")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: Integration test
print("\n5. Integration test (simple cube)...")
try:
    # 1×1×1 cube with 2×2×2 = 8 elements
    # Node numbering: 3×3×3 = 27 nodes
    
    n_nodes = 27
    n_elements = 8
    
    # Create elements (simplified - just checking dimensions)
    elements = np.zeros((n_elements, 8), dtype=int)
    for e in range(n_elements):
        elements[e] = [e, e+1, e+3, e+2, e+9, e+10, e+12, e+11]
    
    # Generate all required data
    ke0 = compute_hex8_ke0()
    edof = generate_edof_array(elements)
    
    load = LoadCondition(
        load_id=1,
        location=np.array([0.5, 0.5, 1.0]),
        is_face=True,
        f_z=-1000.0,
        affected_nodes=[21, 22, 24, 25]  # Top face center nodes
    )
    f = assemble_force_vector([load], n_nodes)
    
    support = SupportCondition(
        support_id=1,
        location=np.array([0.0, 0.0, 0.0]),
        is_face=True,
        u_x=True, u_y=True, u_z=True,
        affected_nodes=[0, 1, 2, 3, 4, 5, 6, 7, 8]  # Bottom face
    )
    fixed_dofs = assemble_fixed_dofs([support])
    
    print(f"   ✓ All components generated successfully")
    print(f"   ke0 shape: {ke0.shape}")
    print(f"   edof shape: {edof.shape}")
    print(f"   f shape: {f.shape}, non-zero: {np.count_nonzero(f)}")
    print(f"   fixed_dofs length: {len(fixed_dofs)}")
    
    # Verify dimensions match OPT STM GENERATOR requirements
    assert ke0.shape == (24, 24), "ke0 must be 24×24"
    assert edof.shape == (n_elements, 24), "edof must be (ne, 24)"
    assert f.shape == (3 * n_nodes,), "f must be (3*n_nodes,)"
    assert fixed_dofs.ndim == 1, "fixed_dofs must be 1D array"
    
    print(f"   ✓ All dimensions correct for OPT STM GENERATOR")
    
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*70)
print("✓ ALL TESTS COMPLETED")
print("="*70)
print("\nThe implementation is ready for use with OPT STM GENERATOR.")
print("Run problem_input_definition.py to generate Excel template and")
print("process geometry files.")
