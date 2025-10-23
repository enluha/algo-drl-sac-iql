"""
Test script to verify coincident node detection.

Loads the corbel example and verifies no coincident nodes exist.
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from preprocessing.problem_input_definition import (
    GeometryProcessor,
    BoundaryConditionProcessor,
    load_excel_data
)

def test_corbel_example():
    """Load corbel example and verify coincident node check."""
    
    excel_file = project_root / 'Example_1_Corbel_downwards_force.xlsx'
    
    if not excel_file.exists():
        print(f"❌ Example file not found: {excel_file}")
        return
    
    print("=" * 70)
    print("Testing Coincident Node Verification")
    print("=" * 70)
    print(f"\nLoading: {excel_file.name}\n")
    
    # Load Excel data
    data = load_excel_data(excel_file)
    
    # Create geometry processor with loaded config
    processor = GeometryProcessor(data['config'])
    
    # Add positive volumes
    pos_volumes = {}
    for face in data['positive_faces']:
        if face.volume_id not in pos_volumes:
            pos_volumes[face.volume_id] = []
        pos_volumes[face.volume_id].append(face)
    
    for vol_id, faces in pos_volumes.items():
        processor.add_positive_volume(faces)
        print(f"  ✓ Added positive volume {vol_id}: {len(faces)} faces")
    
    # Add negative volumes if any
    if data['negative_faces']:
        neg_volumes = {}
        for face in data['negative_faces']:
            if face.volume_id not in neg_volumes:
                neg_volumes[face.volume_id] = []
            neg_volumes[face.volume_id].append(face)
        
        for vol_id, faces in neg_volumes.items():
            processor.add_negative_volume(faces)
            print(f"  ✓ Added negative volume {vol_id}: {len(faces)} faces")
    
    # Generate mesh - this will trigger the coincident node verification!
    print("\nGenerating voxel mesh...")
    processor.generate_voxel_mesh()
    
    print("\nGenerating node-element connectivity...")
    processor.generate_node_element_connectivity()
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)

if __name__ == '__main__':
    test_corbel_example()
