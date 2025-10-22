# OPT STM Generator

**Optimization-based Strut-and-Tie Model Generator for Reinforced Concrete Design**

## Overview

OPT STM Generator is a comprehensive Python-based tool that combines topology optimization with strut-and-tie modeling to design efficient reinforcement layouts for concrete structures. The tool implements a three-stage pipeline:

1. **Topology Optimization (TO)**: SIMP-based compliance minimization
2. **Topology Extraction**: 3D thinning and graph extraction
3. **Shape Optimization**: Beam-based strut-and-tie refinement

## Features

- ✅ **Advanced FEM**: 8-node hexahedral elements with PCG solver
- ✅ **GCMMA Optimizer**: Globally Convergent Method of Moving Asymptotes
- ✅ **Non-convex Geometry**: Ray-casting algorithm for complex shapes
- ✅ **Excel Interface**: User-friendly input definition via templates
- ✅ **3D Visualization**: Interactive matplotlib-based geometry preview
- 🚧 **PyQt5 GUI**: Desktop application (planned)
- 🚧 **SQLite Database**: Model persistence (planned)
- 🚧 **Robot Export**: Autodesk Robot Structural Analysis compatibility (planned)

## Installation

### Prerequisites
- Python 3.9 or higher
- NumPy, SciPy, pandas, matplotlib, openpyxl

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies (Future Features)

```bash
# For GUI development
pip install PyQt5

# For testing
pip install pytest pytest-cov
```

## Quick Start

### **Easiest Way: Double-click `run_interactive.bat`** 🚀

Or use command line:

```powershell
# Interactive mode - Generate Excel template
.\venv\Scripts\python.exe main.py --interactive

# Run built-in example
.\venv\Scripts\python.exe main.py --example

# Run from Excel configuration
.\venv\Scripts\python.exe main.py --config my_problem.xlsx
```

**📖 See `HOW_TO_RUN.md` for detailed instructions**

### 1. Define Problem Geometry (Programmatic API)

```python
from src.preprocessing import GeometryProcessor, Face, GeometryConfig

# Configure geometry
config = GeometryConfig()
config.brick_size_x = 10.0  # mm
config.brick_size_y = 10.0
config.brick_size_z = 10.0

# Create processor
geom = GeometryProcessor(config)

# Define cube faces
cube_faces = [
    Face(points=np.array([[0,0,0], [100,0,0], [100,100,0], [0,100,0]]), 
         face_id=1, volume_id=1),
    # ... more faces
]

geom.add_positive_volume(cube_faces)
```

### 2. Run Optimization

```python
from OPT_STM_GENERATOR import OPTSTMOrchestrator

# Load problem
orchestrator = OPTSTMOrchestrator()
orchestrator.load_problem_from_pickle("problem.pkl")

# Run optimization pipeline
results = orchestrator.run_pipeline()
```

## Project Structure

```
opt_stm_generator/
├── src/                      # Source code
│   ├── core/                 # Core algorithms (FEM, optimizer)
│   ├── preprocessing/        # Geometry and mesh generation
│   ├── optimization/         # TO and shape optimization
│   ├── extraction/           # Topology extraction
│   ├── postprocessing/       # Visualization and export
│   ├── database/             # SQLite persistence (future)
│   ├── exporters/            # Format converters (future)
│   └── ui/                   # PyQt5 interface (future)
├── tests/                    # Unit and integration tests
├── examples/                 # Usage examples
├── docs/                     # Documentation
└── requirements.txt          # Python dependencies
```

## Documentation

- **User Guide**: See `docs/user_guide.md`
- **API Reference**: See `docs/api_reference.md`
- **Theory**: See `docs/theory.md`
- **Restructuring Proposal**: See `RESTRUCTURING_PROPOSAL.md`

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_optimizer.py
```

## Contributing

This is a proprietary project for AtkinsRéalis. For internal contributions:

1. Create a feature branch
2. Make changes with tests
3. Submit pull request with detailed description
4. Ensure all tests pass

## Roadmap

### Version 3.0 (Current)
- ✅ Core optimization pipeline
- ✅ Excel-based input
- ✅ Non-convex geometry support
- 🚧 Code refactoring and modularization

### Version 3.1 (Planned)
- 🔜 PyQt5 desktop GUI
- 🔜 SQLite model database
- 🔜 Enhanced visualization

### Version 3.2 (Future)
- 🔜 Autodesk Robot export
- 🔜 Batch processing
- 🔜 Parametric studies

## License

Proprietary - AtkinsRéalis
All rights reserved.

## Authors

**AtkinsRéalis Nuclear Division**
- Project: Strut-and-Tie Modeling
- Location: Desktop/LOCAL PF/053 - Nuclear/112 Strut n Tie

## References

Based on research in topology optimization for concrete structures:
- SIMP method (Bendsøe & Sigmund)
- GCMMA optimizer (Svanberg, 2007)
- 3D thinning algorithm (Lee et al., 1994)

## Contact

For questions or issues, contact the Nuclear Division team at AtkinsRéalis.

---

**Last Updated**: October 21, 2025
