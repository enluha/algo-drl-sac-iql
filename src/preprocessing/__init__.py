"""Preprocessing module for geometry definition and mesh generation.

Exports
-------
All classes and functions from problem_input_definition module for easy access.
"""

from .problem_input_definition import (
    GeometryConfig,
    Face,
    SupportCondition,
    LoadCondition,
    ExcelTemplateGenerator,
    GeometryProcessor,
    BoundaryConditionProcessor,
    ModelVisualizer,
    ModelExporter,
    compute_hex8_ke0,
    generate_edof_array,
    assemble_force_vector,
    assemble_fixed_dofs,
    load_excel_data,
    parse_face_definitions,
    parse_supports,
    parse_loads
)

__all__ = [
    "GeometryConfig",
    "Face",
    "SupportCondition",
    "LoadCondition",
    "ExcelTemplateGenerator",
    "GeometryProcessor",
    "BoundaryConditionProcessor",
    "ModelVisualizer",
    "ModelExporter",
    "compute_hex8_ke0",
    "generate_edof_array",
    "assemble_force_vector",
    "assemble_fixed_dofs",
    "load_excel_data",
    "parse_face_definitions",
    "parse_supports",
    "parse_loads"
]
