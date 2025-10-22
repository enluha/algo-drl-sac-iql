"""
Migration Script for OPT STM Generator Restructuring
====================================================

This script helps migrate from the old structure to the new modular structure.

Usage:
    python scripts/migrate_files.py --preview    # Preview changes (dry run)
    python scripts/migrate_files.py --execute    # Execute migration

What it does:
1. Backs up current files to archive/
2. Splits OPT STM GENERATOR.py into modules
3. Renames files to PEP 8 standards
4. Updates imports across all files
5. Validates the migration

Author: AtkinsRéalis
Date: October 21, 2025
"""

import shutil
import sys
from pathlib import Path
from datetime import datetime


class FileMigrator:
    """Handles the migration of files to new structure."""
    
    def __init__(self, root_dir: Path):
        self.root = Path(root_dir)
        self.backup_dir = self.root / "archive" / f"pre_v3_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.src_dir = self.root / "src"
        
    def preview_migration(self):
        """Preview what will be migrated without making changes."""
        print("="*70)
        print("MIGRATION PREVIEW")
        print("="*70)
        
        print("\n1. Files to backup:")
        files_to_backup = [
            "MMA Class.py",
            "OPT STM GENERATOR.py",
            "OPT STM GENERATOR_v2.py",
            "problem_input_definition.py",
        ]
        for f in files_to_backup:
            src = self.root / f
            if src.exists():
                print(f"   ✓ {f}")
            else:
                print(f"   ✗ {f} (not found)")
        
        print("\n2. Proposed splitting of 'OPT STM GENERATOR.py':")
        splits = {
            "Lines 70-905": "src/core/optimizer.py (GCMMA)",
            "Lines 909-978": "src/core/config.py (Config classes)",
            "Lines 980-1318": "src/core/fem_continuum.py (ContinuumFESolver)",
            "Lines 1320-1345": "src/core/fem_beam.py (BeamFESolver)",
            "Lines 1347-1421": "src/optimization/topology_driver.py (TOProblemDriver)",
            "Lines 1423-1539": "src/optimization/shape_driver.py (ShapeProblemDriver)",
            "Lines 1541-1577": "src/extraction/topology_extractor.py (TopologyExtractor3D)",
            "Lines 1579-1601": "src/postprocessing/visualizer.py (ResultsVisualizer)",
            "Lines 1603-1657": "src/orchestrator.py (OPTSTMOrchestrator)",
        }
        for lines, dest in splits.items():
            print(f"   {lines:20s} → {dest}")
        
        print("\n3. Proposed splitting of 'problem_input_definition.py':")
        splits_input = {
            "Geometry classes": "src/preprocessing/geometry.py",
            "BC classes": "src/preprocessing/boundary_conditions.py",
            "Mesh generator": "src/preprocessing/mesh_generator.py",
            "Excel interface": "src/preprocessing/excel_interface.py",
        }
        for component, dest in splits_input.items():
            print(f"   {component:25s} → {dest}")
        
        print("\n4. Files to rename:")
        renames = {
            "MMA Class.py": "src/core/optimizer.py",
        }
        for old, new in renames.items():
            print(f"   {old:30s} → {new}")
        
        print("\n5. Import updates required:")
        print("   • All files importing from 'OPT STM GENERATOR'")
        print("   • All files importing from 'MMA Class'")
        print("   • All files importing from 'problem_input_definition'")
        
        print("\n" + "="*70)
        print("This is a PREVIEW only. No files have been modified.")
        print("="*70)
    
    def execute_migration(self):
        """Execute the migration (NOT IMPLEMENTED - MANUAL REQUIRED)."""
        print("="*70)
        print("MIGRATION EXECUTION")
        print("="*70)
        print("\n⚠️  WARNING: Automatic migration not yet implemented.")
        print("\nThe migration requires manual code extraction due to:")
        print("  • Complex class dependencies")
        print("  • Inter-module imports that need careful handling")
        print("  • Risk of breaking existing functionality")
        print("\nRECOMMENDED APPROACH:")
        print("  1. Review RESTRUCTURING_PROPOSAL.md")
        print("  2. Follow the phased migration plan")
        print("  3. Migrate one module at a time")
        print("  4. Test after each phase")
        print("\nFor automated backup only, use --backup flag")
        print("="*70)
    
    def backup_files(self):
        """Create backup of current files."""
        print(f"\nCreating backup in: {self.backup_dir}")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        files_to_backup = [
            "MMA Class.py",
            "OPT STM GENERATOR.py", 
            "OPT STM GENERATOR_v2.py",
            "problem_input_definition.py",
        ]
        
        for filename in files_to_backup:
            src = self.root / filename
            if src.exists():
                dst = self.backup_dir / filename
                shutil.copy2(src, dst)
                print(f"  ✓ Backed up: {filename}")
            else:
                print(f"  ⚠ Not found: {filename}")
        
        print(f"\n✓ Backup complete: {self.backup_dir}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python migrate_files.py --preview     # Preview migration")
        print("  python migrate_files.py --execute     # Execute migration (manual)")
        print("  python migrate_files.py --backup      # Backup files only")
        sys.exit(1)
    
    # Get repository root (parent of scripts/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    migrator = FileMigrator(repo_root)
    
    mode = sys.argv[1]
    
    if mode == "--preview":
        migrator.preview_migration()
    elif mode == "--execute":
        migrator.execute_migration()
    elif mode == "--backup":
        migrator.backup_files()
    else:
        print(f"Unknown option: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
