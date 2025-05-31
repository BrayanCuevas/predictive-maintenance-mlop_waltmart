"""
Test configuration and fixtures for predictive maintenance tests.
Shared fixtures and utilities for all test modules.

Author: Brayan Cuevas
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
import tempfile
import os
import joblib
from pathlib import Path
from datetime import datetime, timedelta

# Set random seed for reproducible tests
np.random.seed(42)


@pytest.fixture
def sample_telemetry_data():
    """Create sample telemetry data for testing."""
    
    data = {
        'datetime': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'machineID': np.random.randint(1, 11, 1000),
        'volt': np.random.normal(150, 10, 1000),
        'rotate': np.random.normal(480, 20, 1000),
        'pressure': np.random.normal(95, 5, 1000),
        'vibration': np.random.normal(40, 8, 1000)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_failures_data():
    """Create sample failures data for testing."""
    
    data = {
        'datetime': pd.date_range('2024-01-01', periods=10, freq='7D'),
        'machineID': np.random.randint(1, 11, 10),
        'failure': np.random.choice(['comp1', 'comp2', 'comp3', 'comp4'], 10)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_processed_data(sample_telemetry_data, sample_failures_data):
    """Create sample processed data with features."""
    from src.data.feature_engineering import MaintenanceFeatureEngineer
    
    engineer = MaintenanceFeatureEngineer(prediction_window_days=3)
    processed_df, feature_names = engineer.process_features(
        sample_telemetry_data, sample_failures_data
    )
    
    return processed_df, feature_names


@pytest.fixture
def temp_model_file():
    """Create temporary model file for testing."""
    
    # Create a simple mock model package
    mock_model_data = {
        'model': None,  # Will be filled with actual model in tests
        'model_name': 'Test Random Forest',
        'feature_names': [
            'volt', 'rotate', 'pressure', 'vibration',
            'volt_3h_mean', 'volt_3h_std', 'volt_24h_mean', 'volt_24h_std'
        ],
        'evaluation': {
            'auc_score': 0.85,
            'confusion_matrix': {
                'true_negatives': 800,
                'false_positives': 50,
                'false_negatives': 30,
                'true_positives': 120
            }
        },
        'training_date': datetime.now().isoformat()
    }
    
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        joblib.dump(mock_model_data, f.name)
        yield f.name
    
    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    from src.api.main import app
    return TestClient(app)


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request data."""
    return {
        "machineID": 1,
        "volt": 150.5,
        "rotate": 480.2,
        "pressure": 95.1,
        "vibration": 38.7
    }


@pytest.fixture
def temp_csv_files():
    """Create temporary CSV files for testing data loading."""
    
    # Create sample telemetry CSV
    telemetry_data = {
        'datetime': pd.date_range('2024-01-01', periods=100, freq='H'),
        'machineID': np.random.randint(1, 6, 100),
        'volt': np.random.normal(150, 10, 100),
        'rotate': np.random.normal(480, 20, 100),
        'pressure': np.random.normal(95, 5, 100),
        'vibration': np.random.normal(40, 8, 100)
    }
    telemetry_df = pd.DataFrame(telemetry_data)
    
    # Create sample failures CSV
    failures_data = {
        'datetime': ['2024-01-05', '2024-01-12', '2024-01-18'],
        'machineID': [1, 3, 2],
        'failure': ['comp1', 'comp2', 'comp1']
    }
    failures_df = pd.DataFrame(failures_data)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='_telemetry.csv', delete=False) as telem_f:
        telemetry_df.to_csv(telem_f.name, index=False)
        telem_path = telem_f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_failures.csv', delete=False) as fail_f:
        failures_df.to_csv(fail_f.name, index=False)
        fail_path = fail_f.name
    
    yield telem_path, fail_path
    
    # Cleanup
    os.unlink(telem_path)
    os.unlink(fail_path)


@pytest.fixture
def test_data_directory():
    """Create and manage test data directory."""
    test_dir = Path("test/data")
    test_dir.mkdir(exist_ok=True)
    
    yield test_dir
    
    # Optional cleanup - comment out if you want to keep test data
    # import shutil
    # shutil.rmtree(test_dir, ignore_errors=True)


# Test configuration
def pytest_configure(config):
    """Pytest configuration."""
    
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests as API tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# Test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def assert_dataframe_equal(df1, df2, check_dtype=True):
        """Assert two DataFrames are equal with better error messages."""
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
    
    @staticmethod
    def create_mock_model():
        """Create a mock trained model for testing."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create synthetic data
        X, y = make_classification(n_samples=100, n_features=8, random_state=42)
        
        # Train simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils()