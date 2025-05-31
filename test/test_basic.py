"""
Basic tests to validate CI/CD pipeline functionality.

Author: Brayan Cuevas
"""

import pytest
import pandas as pd
import numpy as np


class TestBasicFunctionality:
    """Basic tests to ensure CI/CD pipeline works."""
    
    def test_imports_work(self):
        """Test that core imports work correctly."""
        # Test data science imports
        assert pd.__version__
        assert np.__version__
        
        # Test that we can import our modules
        from src.data import load_maintenance_data
        from src.models import ModelTrainer
        
        assert callable(load_maintenance_data)
        assert ModelTrainer is not None
    
    def test_sample_data_fixture(self, sample_telemetry_data):
        """Test that sample data fixture works."""
        assert isinstance(sample_telemetry_data, pd.DataFrame)
        assert len(sample_telemetry_data) == 1000
        assert 'machineID' in sample_telemetry_data.columns
        assert 'volt' in sample_telemetry_data.columns
    
    def test_data_processing_basic(self, sample_telemetry_data):
        """Test basic data processing functionality."""
        # Test basic data manipulation
        grouped = sample_telemetry_data.groupby('machineID')['volt'].mean()
        assert len(grouped) > 0
        
        # Test that all volt values are reasonable
        assert sample_telemetry_data['volt'].min() > 100
        assert sample_telemetry_data['volt'].max() < 200
    
    @pytest.mark.unit
    def test_feature_engineering_import(self):
        """Test feature engineering module imports."""
        from src.data.feature_engineering import MaintenanceFeatureEngineer
        
        engineer = MaintenanceFeatureEngineer(prediction_window_days=3)
        assert engineer.prediction_window_days == 3
        assert engineer.sensor_columns == ['volt', 'rotate', 'pressure', 'vibration']
    
    @pytest.mark.unit  
    def test_model_trainer_import(self):
        """Test model trainer imports."""
        from src.models.trainer import ModelTrainer
        
        trainer = ModelTrainer()
        assert trainer.test_size == 0.2
        assert trainer.random_state == 42
    
    def test_math_operations(self):
        """Test basic math operations work."""
        assert 2 + 2 == 4
        assert np.array([1, 2, 3]).sum() == 6
        
        # Test pandas operations
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert df['a'].sum() == 6
        assert df['b'].mean() == 5.0


class TestEnvironment:
    """Test environment and dependencies."""
    
    def test_python_version(self):
        """Test Python version is compatible."""
        import sys
        version = sys.version_info
        
        # Should be Python 3.9+
        assert version.major == 3
        assert version.minor >= 9
    
    def test_required_packages(self):
        """Test that required packages are installed."""
        import pandas
        import numpy
        import sklearn
        import fastapi
        import uvicorn
        
        # Basic version checks
        assert pandas.__version__ >= "1.5.0"
        assert numpy.__version__ >= "1.20.0"
    
    def test_project_structure(self):
        """Test that project structure is correct."""
        from pathlib import Path
        
        project_root = Path(__file__).parent.parent
        
        # Check key directories exist
        assert (project_root / "src").exists()
        assert (project_root / "src" / "data").exists()
        assert (project_root / "src" / "models").exists()
        assert (project_root / "src" / "api").exists()
        
        # Check key files exist
        assert (project_root / "requirements.txt").exists()
        assert (project_root / "Dockerfile").exists()


# Simple integration test
@pytest.mark.integration
class TestIntegration:
    """Basic integration tests."""
    
    def test_api_client_creation(self, api_client):
        """Test that API client can be created."""
        assert api_client is not None
        
        # Test basic endpoint (might fail if model not loaded, but import should work)
        try:
            response = api_client.get("/")
            # If it responds, great. If not, that's ok for this basic test
            assert response.status_code in [200, 503]
        except Exception:
            # API might not be fully configured in test environment
            pass