#!/usr/bin/env python
"""
OPT-STM Generator - Main Entry Point
======================================

This is the main executable for running the OPT-STM optimization pipeline.

Usage:
------
    python main.py --config my_config.json
    python main.py --interactive
    python main.py --example

For detailed usage, run:
    python main.py --help

Author: OPT-STM Project
Date: October 21, 2025
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import (
    ProblemConfig,
    TOConfig,
    ShapeConfig,
    OptimizerConfig,
    ContinuumFESolver,
    BeamFESolver
)
from src.orchestrator import OPTSTMOrchestrator
from src.preprocessing import (
    ExcelTemplateGenerator,
    GeometryProcessor,
    BoundaryConditionProcessor,
    ModelVisualizer,
    load_excel_data
)


def run_example():
    """
    Run a simple example problem to demonstrate the workflow.
    """
    print("\n" + "="*70)
    print("OPT-STM GENERATOR - Example Mode")
    print("="*70)
    print("\n⚠️  This is a demonstration of the workflow structure.")
    print("You need to provide your actual problem geometry and FE solvers.\n")
    
    # Create configuration
    config = ProblemConfig(
        to=TOConfig(
            vol_frac=0.10,
            simp_p=3.0,
            rho_min=1e-4,
            filter_radius=2.5
        ),
        shape=ShapeConfig(
            sts_epsilon=0.05,
            min_length_p=10,
            min_length_target=50.0,
            fd_step_rel=1e-3
        ),
        opt=OptimizerConfig(
            move=0.45,
            max_outer=500,
            verbose=True
        )
    )
    
    print("✅ Configuration created:")
    print(f"   - Volume fraction: {config.to.vol_frac*100}%")
    print(f"   - SIMP penalty: {config.to.simp_p}")
    print(f"   - Max iterations: {config.opt.max_outer}\n")
    
    print("="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. Define your problem geometry using problem_input_definition.py")
    print("2. Create ContinuumFESolver from your mesh data")
    print("3. Implement BeamFESolver subclass")
    print("4. Initialize orchestrator and run optimization")
    print("\nSee USAGE_GUIDE.md for complete instructions!")
    print("See examples/simple_example.py for code structure!\n")


def run_interactive():
    """
    Interactive mode for guided problem setup.
    """
    print("\n" + "="*70)
    print("OPT-STM GENERATOR - Interactive Mode")
    print("="*70)
    
    # Step 1: Generate template
    print("\nStep 1: Excel Template Generation")
    print("-" * 40)
    
    try:
        need_template = input("Generate new Excel template? (Y/N): ").strip().upper()
    except EOFError:
        print("\n❌ No input provided. Use 'python main.py --config <file.xlsx>' instead.")
        return
    
    if need_template == 'Y':
        try:
            filename = input("Enter template filename (default: OPT_STM_Input_Template.xlsx): ").strip()
        except EOFError:
            filename = "OPT_STM_Input_Template.xlsx"
        
        if not filename:
            filename = "OPT_STM_Input_Template.xlsx"
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'
        
        template_path = Path(filename)
        print(f"\nGenerating template: {template_path}")
        
        try:
            generator = ExcelTemplateGenerator(template_path)
            generator.generate()
            print(f"\n✅ Template generated: {template_path}")
            print("\nNext steps:")
            print("1. Fill in the template with your geometry, supports, and loads")
            print("2. Run: python main.py --config " + str(template_path))
        except Exception as e:
            print(f"\n❌ Error generating template: {e}")
        
        return
    
    # Step 2: Load existing file
    print("\nStep 2: Load Existing Configuration")
    print("-" * 40)
    
    try:
        filepath_str = input("Enter path to Excel file: ").strip().strip('"')
    except EOFError:
        print("\n❌ No input provided. Use 'python main.py --config <file.xlsx>' instead.")
        return
    
    if filepath_str:
        run_from_config(filepath_str)
    else:
        print("\n❌ No file specified. Exiting.")


def run_from_config(config_path: str):
    """
    Run optimization from configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file (JSON or Excel)
    """
    print("\n" + "="*70)
    print("OPT-STM GENERATOR - Running from Configuration")
    print("="*70)
    print(f"\nConfiguration file: {config_path}\n")
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        print(f"❌ Error: Configuration file not found: {config_path}")
        return
    
    if config_file.suffix in ['.xlsx', '.xls']:
        print("Loading Excel configuration...")
        try:
            # Load Excel data
            data = load_excel_data(config_file)
            
            # Process geometry
            print("\nProcessing geometry...")
            geometry = GeometryProcessor(data['config'])
            
            # Add positive volumes
            pos_volumes = {}
            for face in data['positive_faces']:
                if face.volume_id not in pos_volumes:
                    pos_volumes[face.volume_id] = []
                pos_volumes[face.volume_id].append(face)
            
            for vol_id, faces in pos_volumes.items():
                geometry.add_positive_volume(faces)
                print(f"  ✓ Added positive volume {vol_id}: {len(faces)} faces")
            
            # Add negative volumes if any
            if data['negative_faces']:
                neg_volumes = {}
                for face in data['negative_faces']:
                    if face.volume_id not in neg_volumes:
                        neg_volumes[face.volume_id] = []
                    neg_volumes[face.volume_id].append(face)
                
                for vol_id, faces in neg_volumes.items():
                    geometry.add_negative_volume(faces)
                    print(f"  ✓ Added negative volume {vol_id}: {len(faces)} faces")
            
            # Apply origin shift to place model in first octant
            shift = geometry.compute_origin_shift()
            if np.any(shift > 1e-10):
                print(f"\n  Model offset to place in first octant.")
                print(f"  Origin shift applied: [{shift[0]:.3f}, {shift[1]:.3f}, {shift[2]:.3f}] {data['config'].units_length}")
                geometry.apply_shift(shift)
            else:
                print(f"\n  Model is already in first octant. No origin shift applied.")
            
            # Generate mesh
            print("\nGenerating voxel mesh...")
            geometry.generate_voxel_mesh()
            geometry.generate_node_element_connectivity()
            
            # Process boundary conditions
            print("\nProcessing boundary conditions...")
            bc_processor = BoundaryConditionProcessor(geometry)
            
            for support in data['supports']:
                bc_processor.add_support(support)
            
            for load in data['loads']:
                bc_processor.add_load(load)
            
            # Visualize
            print("\nVisualizing model...")
            visualizer = ModelVisualizer(geometry, bc_processor)
            plot_path = config_file.parent / f"{config_file.stem}_preview.png"
            visualizer.plot_model(show_plot=True, save_path=plot_path)
            
            print("\n✅ Excel configuration loaded successfully!")
            print("\n⚠️  Next step: Integrate this mesh with ContinuumFESolver")
            print("    See USAGE_GUIDE.md for details on running optimization")
            
        except Exception as e:
            print(f"\n❌ Error processing Excel file: {e}")
            import traceback
            traceback.print_exc()
            
    elif config_file.suffix == '.json':
        print("⚠️  JSON configuration loading not yet implemented.")
        print("You can define configuration programmatically using ProblemConfig.\n")
        print("See examples/simple_example.py for an example!\n")
    else:
        print(f"❌ Error: Unsupported configuration file type: {config_file.suffix}")
        print("Supported types: .xlsx, .json\n")


def main():
    """
    Main entry point with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="OPT-STM Generator - Optimization-based Strut-and-Tie Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --example              Run example demonstration
  python main.py --interactive          Interactive problem setup (not yet available)
  python main.py --config problem.xlsx  Run from Excel configuration
  python main.py --config config.json   Run from JSON configuration

For detailed usage instructions, see USAGE_GUIDE.md
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (Excel or JSON)'
    )
    
    parser.add_argument(
        '--example',
        action='store_true',
        help='Run example demonstration'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode for problem setup'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='OPT-STM Generator v3.0 (Modular Architecture)'
    )
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.example:
        run_example()
    elif args.interactive:
        run_interactive()
    elif args.config:
        run_from_config(args.config)
    else:
        # No arguments - show help
        parser.print_help()
        print("\n" + "="*70)
        print("QUICK START:")
        print("="*70)
        print("\n1. Read the usage guide:")
        print("     cat USAGE_GUIDE.md")
        print("\n2. Run the example:")
        print("     python main.py --example")
        print("\n3. Or run example script:")
        print("     python examples/simple_example.py")
        print("\n4. For detailed instructions, see USAGE_GUIDE.md\n")


if __name__ == "__main__":
    main()
