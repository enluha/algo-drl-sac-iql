"""
Diagnostic script to visualize surface nodes for a specific face.
Non-persistent - creates temporary plot window for inspection.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_face_surface_nodes(nodes, surface_nodes, face_points, face_id, volume_id):
    """
    Plot surface nodes that belong to a specific face for diagnostic purposes.
    Shows ONLY the actual boundary nodes being plotted in the main visualization.
    
    Parameters:
    -----------
    nodes : np.ndarray
        All mesh nodes (N x 3)
    surface_nodes : np.ndarray
        Indices of surface (boundary) nodes
    face_points : np.ndarray
        Points defining the face plane (4 x 3)
    face_id : int
        Face ID to visualize
    volume_id : int
        Volume ID to visualize
    """
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: Plotting surface boundary nodes for Volume_ID={volume_id}, Face_ID={face_id}")
    print(f"{'='*70}")
    
    # Compute face plane normal and center
    face_center = np.mean(face_points, axis=0)
    v1 = face_points[1] - face_points[0]
    v2 = face_points[2] - face_points[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    print(f"Face center: {face_center}")
    print(f"Face normal: {normal}")
    
    # Get coordinates of surface boundary nodes
    surface_node_coords = nodes[surface_nodes]
    
    # Filter to nodes near this face plane (within reasonable tolerance)
    # Use 2.5% above half voxel size (approximately 17.5mm for 33.33mm voxels)
    tolerance = 17.5  # mm - slightly above half voxel to catch surface layer
    
    face_nodes = []
    face_node_indices = []
    for i, node_idx in enumerate(surface_nodes):
        node_coord = surface_node_coords[i]
        dist = abs(np.dot(normal, node_coord - face_center))
        if dist < tolerance:
            face_nodes.append(node_coord)
            face_node_indices.append(node_idx)
    
    face_nodes = np.array(face_nodes) if face_nodes else np.array([]).reshape(0, 3)
    
    print(f"Found {len(face_nodes)} surface boundary nodes near Face_ID={face_id} (within {tolerance:.1f}mm)")
    
    # Compute distance of nodes near this face from the face plane
    if len(face_nodes) > 0:
        distances = []
        for coord in face_nodes:
            dist = abs(np.dot(normal, coord - face_center))
            distances.append(dist)
        
        distances = np.array(distances)
        print(f"Node distances from face plane:")
        print(f"  Min: {distances.min():.2f} mm")
        print(f"  Max: {distances.max():.2f} mm")
        print(f"  Mean: {distances.mean():.2f} mm")
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ONLY surface boundary nodes near this face (exactly as shown in main visualization)
    if len(face_nodes) > 0:
        ax.scatter(face_nodes[:, 0], face_nodes[:, 1], face_nodes[:, 2],
                  c='lightblue', s=20, marker='o', label=f'Surface Boundary Nodes (n={len(face_nodes)})',
                  edgecolors='steelblue', linewidths=0.5, alpha=0.4, depthshade=True)
    
    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    ax.set_zlabel('Z (mm)', fontsize=10)
    ax.set_title(f'Surface Boundary Nodes for Volume_ID={volume_id}, Face_ID={face_id}\n'
                f'{len(face_nodes)} nodes on boundary', 
                fontsize=12, fontweight='bold')
    
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    if len(face_nodes) > 0:
        max_range = np.ptp(face_nodes, axis=0).max() / 2.0
        mid_x = (face_nodes[:, 0].max() + face_nodes[:, 0].min()) * 0.5
        mid_y = (face_nodes[:, 1].max() + face_nodes[:, 1].min()) * 0.5
        mid_z = (face_nodes[:, 2].max() + face_nodes[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    print(f"\nDisplaying diagnostic plot (close window to continue)...")
    plt.show()
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("This is a diagnostic module. Import and call plot_face_surface_nodes() from your main code.")
