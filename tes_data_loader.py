#!/usr/bin/env python3
"""
Test script for data loader module
Tests the modular data loading functionality.

Author: Brayan Cuevas
Created: December 2024
"""

import os
import sys

def test_environment():
    """Test that we're in the right directory and files exist."""
    print("=== ENVIRONMENT CHECK ===")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.executable}")
    
    # Check required files exist
    required_files = [
        'src/__init__.py',
        'src/data/__init__.py', 
        'src/data/data_loader.py',
        'data/raw/PdM_telemetry.csv',
        'data/raw/PdM_failures.csv'
    ]
    
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "PASS" if exists else "FAIL"
        print(f"[{status}] {file_path}")
        
        if not exists and 'csv' in file_path:
            print("   WARNING: CSV files need to be downloaded from Kaggle")
    
    print()

def test_imports():
    """Test that we can import our modules."""
    print("=== IMPORT TEST ===")
    try:
        from src.data import load_maintenance_data, MaintenanceDataLoader
        print("PASS: Successfully imported modules")
        return True
    except ImportError as e:
        print(f"FAIL: Import failed: {e}")
        return False
    except Exception as e:
        print(f"FAIL: Unexpected error: {e}")
        return False

def test_simple_function():
    """Test the simple convenience function."""
    print("=== TESTING SIMPLE FUNCTION ===")
    try:
        from src.data import load_maintenance_data
        
        telemetry, failures = load_maintenance_data()
        
        print(f"PASS: Telemetry loaded: {len(telemetry):,} records")
        print(f"PASS: Failures loaded: {len(failures)} events")
        print(f"PASS: Date range: {telemetry['datetime'].min().date()} to {telemetry['datetime'].max().date()}")
        
        return telemetry, failures
        
    except FileNotFoundError as e:
        print(f"FAIL: File not found: {e}")
        print("   INFO: Make sure CSV files are in data/raw/")
        return None, None
    except Exception as e:
        print(f"FAIL: Error: {e}")
        return None, None

def test_class_with_validation():
    """Test the full class with validation."""
    print("\n=== TESTING CLASS WITH VALIDATION ===")
    try:
        from src.data import MaintenanceDataLoader
        
        loader = MaintenanceDataLoader()
        telemetry, failures, validation = loader.load_and_validate()
        
        print(f"\nValidation Summary:")
        print(f"  Date range: {validation['date_range'][0].date()} to {validation['date_range'][1].date()}")
        print(f"  Unique machines: {validation['unique_machines']}")
        print(f"  Total failures: {validation['total_failures']}")
        print(f"  Failure types: {len(validation['failure_types'])}")
        
        # Show failure breakdown
        print(f"\nFailure Type Breakdown:")
        for failure_type, count in validation['failure_types'].items():
            print(f"  {failure_type}: {count}")
        
        return telemetry, failures, validation
        
    except Exception as e:
        print(f"FAIL: Error in validation test: {e}")
        return None, None, None

def test_data_quality():
    """Test basic data quality checks."""
    print("\n=== DATA QUALITY CHECKS ===")
    try:
        from src.data import MaintenanceDataLoader
        
        loader = MaintenanceDataLoader()
        telemetry, failures, validation = loader.load_and_validate()
        
        # Check for missing values
        missing_values = validation['missing_values']
        total_missing = sum(missing_values.values())
        
        if total_missing == 0:
            print("PASS: No missing values found")
        else:
            print(f"WARNING: Found {total_missing} missing values:")
            for col, count in missing_values.items():
                if count > 0:
                    print(f"  {col}: {count}")
        
        # Check date consistency
        print(f"PASS: Data covers {(validation['date_range'][1] - validation['date_range'][0]).days} days")
        
        return True
        
    except Exception as e:
        print(f"FAIL: Data quality check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("TESTING DATA LOADER MODULE")
    print("=" * 50)
    
    # Test environment
    test_environment()
    
    # Test imports
    if not test_imports():
        print("FAIL: Cannot proceed - import failed")
        return
    
    # Test functionality
    telemetry, failures = test_simple_function()
    
    if telemetry is not None:
        test_class_with_validation()
        test_data_quality()
        
        print("\nALL TESTS COMPLETED SUCCESSFULLY")
        print("Data loader module is working correctly")
        print("Ready to proceed with feature engineering")
    else:
        print("\nTESTS FAILED")
        print("Check that CSV files are in data/raw/ directory")

if __name__ == "__main__":
    main()