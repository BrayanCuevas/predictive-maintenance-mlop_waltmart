"""
Test suite for predictive maintenance MLOps project.

This package contains all test modules for the predictive maintenance system,
including unit tests, integration tests, and API tests.

Author: Brayan Cuevas
Created for technical challenge implementation.
"""

# Test configuration
import os
import sys
from pathlib import Path

# Add src to Python path for imports during testing
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Test constants
TEST_DATA_DIR = project_root / "test" / "data"
TEST_MODELS_DIR = project_root / "test" / "models"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_MODELS_DIR.mkdir(exist_ok=True)

# Version info
__version__ = "1.0.0"
__author__ = "Brayan Cuevas"

# Test utilities available at package level
from .conftest import *  # This will be created next

print(f"Test suite initialized - Version {__version__}")