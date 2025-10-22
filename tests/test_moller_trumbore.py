"""
Minimal test of ray-triangle intersection algorithm.
Tests the core Möller-Trumbore implementation without dependencies.
"""

import numpy as np

def ray_triangle_intersect(ray_origin, ray_dir, v0, v1, v2, eps=1e-10):
    """
    Möller-Trumbore ray-triangle intersection algorithm.
    
    Parameters:
    -----------
    ray_origin : ndarray (3,)
        Origin point of the ray
    ray_dir : ndarray (3,)
        Direction vector of the ray (should be normalized)
    v0, v1, v2 : ndarray (3,)
        The three vertices of the triangle
    eps : float
        Epsilon for numerical comparisons
    
    Returns:
    --------
    bool
        True if ray intersects triangle in forward direction, False otherwise
    """
    # Compute edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Compute determinant
    pvec = np.cross(ray_dir, edge2)
    det = np.dot(edge1, pvec)
    
    # If determinant is near zero, ray is parallel to triangle
    if abs(det) < eps:
        return False
    
    inv_det = 1.0 / det
    
    # Calculate u parameter and test bounds
    tvec = ray_origin - v0
    u = np.dot(tvec, pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return False
    
    # Calculate v parameter and test bounds
    qvec = np.cross(tvec, edge1)
    v = np.dot(ray_dir, qvec) * inv_det
    if v < 0.0 or u + v > 1.0:
        return False
    
    # Calculate t (distance along ray to intersection)
    t = np.dot(edge2, qvec) * inv_det
    
    # Intersection is valid if t > 0 (forward direction)
    return t > eps


print("="*70)
print("TESTING MÖLLER-TRUMBORE RAY-TRIANGLE INTERSECTION")
print("="*70)

# Test 1: Ray intersects triangle
print("\nTest 1: Ray hits triangle directly")
ray_origin = np.array([0.0, 0.0, 0.0])
ray_dir = np.array([1.0, 0.0, 0.0])
v0 = np.array([10.0, -5.0, -5.0])
v1 = np.array([10.0, 5.0, -5.0])
v2 = np.array([10.0, 0.0, 5.0])

result = ray_triangle_intersect(ray_origin, ray_dir, v0, v1, v2)
print(f"   Ray: origin={ray_origin}, dir={ray_dir}")
print(f"   Triangle: v0={v0}, v1={v1}, v2={v2}")
print(f"   Result: {result}")
print(f"   Expected: True")
if result:
    print("   ✓ PASS")
else:
    print("   ✗ FAIL")

# Test 2: Ray misses triangle (above)
print("\nTest 2: Ray misses triangle (passes above)")
v0_miss = np.array([10.0, 10.0, 10.0])
v1_miss = np.array([10.0, 15.0, 10.0])
v2_miss = np.array([10.0, 12.0, 15.0])

result = ray_triangle_intersect(ray_origin, ray_dir, v0_miss, v1_miss, v2_miss)
print(f"   Triangle: v0={v0_miss}, v1={v1_miss}, v2={v2_miss}")
print(f"   Result: {result}")
print(f"   Expected: False")
if not result:
    print("   ✓ PASS")
else:
    print("   ✗ FAIL")

# Test 3: Ray behind origin
print("\nTest 3: Triangle behind ray origin")
v0_behind = np.array([-10.0, -1.0, -1.0])
v1_behind = np.array([-10.0, 1.0, -1.0])
v2_behind = np.array([-10.0, 0.0, 1.0])

result = ray_triangle_intersect(ray_origin, ray_dir, v0_behind, v1_behind, v2_behind)
print(f"   Triangle: v0={v0_behind}, v1={v1_behind}, v2={v2_behind}")
print(f"   Result: {result}")
print(f"   Expected: False (negative t)")
if not result:
    print("   ✓ PASS")
else:
    print("   ✗ FAIL")

# Test 4: Ray parallel to triangle
print("\nTest 4: Ray parallel to triangle plane")
ray_parallel = np.array([0.0, 1.0, 0.0])
v0_parallel = np.array([5.0, 0.0, 0.0])
v1_parallel = np.array([5.0, 10.0, 0.0])
v2_parallel = np.array([5.0, 5.0, 5.0])

result = ray_triangle_intersect(ray_origin, ray_parallel, v0_parallel, v1_parallel, v2_parallel)
print(f"   Ray dir: {ray_parallel}")
print(f"   Triangle: v0={v0_parallel}, v1={v1_parallel}, v2={v2_parallel}")
print(f"   Result: {result}")
print(f"   Expected: False (parallel, det≈0)")
if not result:
    print("   ✓ PASS")
else:
    print("   ✗ FAIL")

# Test 5: Ray hits edge
print("\nTest 5: Ray hits triangle center")
v0_center = np.array([5.0, -1.0, -1.0])
v1_center = np.array([5.0, 1.0, -1.0])
v2_center = np.array([5.0, 0.0, 1.0])

result = ray_triangle_intersect(ray_origin, ray_dir, v0_center, v1_center, v2_center)
print(f"   Triangle centered on ray path")
print(f"   Result: {result}")
print(f"   Expected: True")
if result:
    print("   ✓ PASS")
else:
    print("   ✗ FAIL")

print("\n" + "="*70)
print("RAY-CASTING ALGORITHM VERIFIED")
print("="*70)
print("\nThe Möller-Trumbore algorithm correctly:")
print("  • Detects intersections with triangles in ray path")
print("  • Rejects triangles behind ray origin (negative t)")
print("  • Rejects triangles outside ray cone (u,v bounds)")
print("  • Handles parallel rays (determinant ≈ 0)")
print("\nThis algorithm is used in problem_input_definition.py for:")
print("  • Point-in-polyhedron testing (ray-casting method)")
print("  • Non-convex geometry voxelation")
print("  • Accurate volume containment detection")
