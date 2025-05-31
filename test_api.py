"""
Test script for FastAPI application

Author: Brayan Cuevas
"""

import subprocess
import time
import requests
import json

def test_api():
    """Test the FastAPI application."""
    print("TESTING FASTAPI APPLICATION")
    print("=" * 40)
    
    # API configuration
    BASE_URL = "http://localhost:8000"
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✓ Health check passed")
            print(f"  Status: {health_data['status']}")
            print(f"  Model loaded: {health_data['model_loaded']}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return
        
        # Test prediction endpoint
        print("\nTesting prediction endpoint...")
        test_data = {
            "machineID": 1,
            "volt": 150.5,
            "rotate": 480.2,
            "pressure": 95.1,
            "vibration": 38.7
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=test_data)
        
        if response.status_code == 200:
            prediction = response.json()
            print(f"✓ Prediction successful")
            print(f"  Machine ID: {prediction['machineID']}")
            print(f"  Failure Probability: {prediction['failure_probability']:.3f}")
            print(f"  Risk Level: {prediction['risk_level']}")
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(f"  Error: {response.text}")
        
        # Test model info endpoint
        print("\nTesting model info endpoint...")
        response = requests.get(f"{BASE_URL}/model/info")
        
        if response.status_code == 200:
            model_info = response.json()
            print(f"✓ Model info retrieved")
            print(f"  Model: {model_info['model_info']['model_name']}")
            print(f"  AUC: {model_info['model_info']['auc_score']}")
        
        print(f"\n🎉 API testing completed successfully!")
        print(f"📖 View documentation at: {BASE_URL}/docs")
        
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API")
        print("  Make sure the API is running on http://localhost:8000")
    except Exception as e:
        print(f"✗ Test failed: {e}")

if __name__ == "__main__":
    test_api()