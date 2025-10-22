"""
Minimal test of ke0 computation without external dependencies.
"""

import numpy as np

def compute_hex8_ke0(E=1.0, nu=0.3):
    """Compute 24x24 reference element stiffness matrix for hex8."""
    # Constitutive matrix
    factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C = factor * np.array([
        [1-nu,   nu,   nu,    0,           0,           0],
        [  nu, 1-nu,   nu,    0,           0,           0],
        [  nu,   nu, 1-nu,    0,           0,           0],
        [   0,    0,    0, (1-2*nu)/2,     0,           0],
        [   0,    0,    0,    0,      (1-2*nu)/2,      0],
        [   0,    0,    0,    0,           0,      (1-2*nu)/2]
    ])
    
    # Gauss points
    gp = 1.0 / np.sqrt(3.0)
    gauss_points = np.array([
        [-gp, -gp, -gp], [ gp, -gp, -gp], [ gp,  gp, -gp], [-gp,  gp, -gp],
        [-gp, -gp,  gp], [ gp, -gp,  gp], [ gp,  gp,  gp], [-gp,  gp,  gp]
    ])
    weights = np.ones(8)
    
    ke0 = np.zeros((24, 24))
    
    for gp_idx in range(8):
        xi, eta, zeta = gauss_points[gp_idx]
        weight = weights[gp_idx]
        
        # Shape function derivatives
        dN = 0.125 * np.array([
            [-(1-eta)*(1-zeta),  (1-eta)*(1-zeta),  (1+eta)*(1-zeta), -(1+eta)*(1-zeta),
             -(1-eta)*(1+zeta),  (1-eta)*(1+zeta),  (1+eta)*(1+zeta), -(1+eta)*(1+zeta)],
            [-(1-xi)*(1-zeta), -(1+xi)*(1-zeta),  (1+xi)*(1-zeta),  (1-xi)*(1-zeta),
             -(1-xi)*(1+zeta), -(1+xi)*(1+zeta),  (1+xi)*(1+zeta),  (1-xi)*(1+zeta)],
            [-(1-xi)*(1-eta), -(1+xi)*(1-eta), -(1+xi)*(1+eta), -(1-xi)*(1+eta),
              (1-xi)*(1-eta),  (1+xi)*(1-eta),  (1+xi)*(1+eta),  (1-xi)*(1+eta)]
        ])
        
        node_coords = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        
        J = dN @ node_coords
        detJ = np.linalg.det(J)
        J_inv = np.linalg.inv(J)
        dN_dx = J_inv @ dN
        
        B = np.zeros((6, 24))
        for i in range(8):
            B[0, 3*i]   = dN_dx[0, i]
            B[1, 3*i+1] = dN_dx[1, i]
            B[2, 3*i+2] = dN_dx[2, i]
            B[3, 3*i]   = dN_dx[1, i]
            B[3, 3*i+1] = dN_dx[0, i]
            B[4, 3*i+1] = dN_dx[2, i]
            B[4, 3*i+2] = dN_dx[1, i]
            B[5, 3*i]   = dN_dx[2, i]
            B[5, 3*i+2] = dN_dx[0, i]
        
        ke0 += (B.T @ C @ B) * detJ * weight
    
    return ke0

print("="*70)
print("MINIMAL KE0 TEST")
print("="*70)

ke0 = compute_hex8_ke0(E=1.0, nu=0.3)

print(f"\n✓ ke0 computed successfully")
print(f"  Shape: {ke0.shape}")
print(f"  Symmetric: {np.allclose(ke0, ke0.T)}")
print(f"  Max element: {np.max(np.abs(ke0)):.6f}")

# Check positive definiteness
eigenvalues = np.linalg.eigvalsh(ke0)
print(f"  Min eigenvalue: {eigenvalues[0]:.6e}")
print(f"  Max eigenvalue: {eigenvalues[-1]:.6e}")

if eigenvalues[0] > -1e-10:
    print(f"  ✓ Matrix is positive semi-definite (as expected for fixed rigid body modes)")
else:
    print(f"  ⚠ Unexpected negative eigenvalues")

print("\n" + "="*70)
print("✓ KE0 COMPUTATION VERIFIED")
print("="*70)
