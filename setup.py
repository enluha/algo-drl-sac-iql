"""
Setup configuration for OPT STM Generator package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="opt-stm-generator",
    version="3.0.0",
    author="AtkinsRéalis",
    author_email="nuclear.division@atkinsrealis.com",
    description="Optimization-based Strut-and-Tie Model Generator for Reinforced Concrete",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atkinsrealis/opt-stm-generator",  # Update with actual URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: Other/Proprietary License",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gui": [
            "PyQt5>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "opt-stm=src.orchestrator:main",  # Future: Add main entry point
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
