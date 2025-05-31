"""
Model prediction logic for API.

Author: Brayan Cuevas
"""

import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging
from datetime import datetime


class ModelPredictor:
    """Handle model loading and predictions for API."""

    def __init__(self, model_path: str = "models/baseline_model.joblib"):
        """
        Initialize predictor with model.

        Args:
            model_path: Path to trained model file
        """
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.feature_names = None
        self.model_info = None
        self.logger = logging.getLogger(__name__)

        # Load model on initialization
        self.load_model()

    def load_model(self) -> bool:
        """
        Load the trained model from disk.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not Path(self.model_path).exists():
                self.logger.error(f"Model file not found: {self.model_path}")
                return False

            self.logger.info(f"Loading model from {self.model_path}")
            self.model_data = joblib.load(self.model_path)

            # Extract model components
            self.model = self.model_data["model"]
            self.feature_names = self.model_data["feature_names"]

            # Extract model info
            self.model_info = {
                "model_name": self.model_data.get("model_name", "Unknown"),
                "training_date": self.model_data.get("training_date", "Unknown"),
                "auc_score": self.model_data.get("evaluation", {}).get("auc_score", "Unknown"),
                "feature_count": len(self.feature_names),
            }

            self.logger.info("Model loaded successfully")
            self.logger.info(f"Model: {self.model_info['model_name']}")
            self.logger.info(f"Features: {self.model_info['feature_count']}")
            self.logger.info(f"AUC: {self.model_info['auc_score']}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if predictor is ready to make predictions."""
        return self.model is not None and self.feature_names is not None

    def create_features_from_request(self, request_data: Dict) -> pd.DataFrame:
        """
        Create feature vector from API request data.

        Args:
            request_data: Dictionary with sensor readings

        Returns:
            DataFrame with features ready for prediction
        """
        # Create basic features from request
        base_features = {
            "volt": request_data["volt"],
            "rotate": request_data["rotate"],
            "pressure": request_data["pressure"],
            "vibration": request_data["vibration"],
        }

        # For API predictions, we simulate rolling features using current values
        # In production, you'd maintain a time series buffer per machine
        sensors = ["volt", "rotate", "pressure", "vibration"]
        windows = [3, 24]

        features = base_features.copy()

        # Create simplified rolling features (using current values as approximation)
        for sensor in sensors:
            sensor_value = base_features[sensor]
            for window in windows:
                # Simulate rolling statistics with current value
                features[f"{sensor}_{window}h_mean"] = sensor_value
                features[f"{sensor}_{window}h_std"] = sensor_value * 0.05  # 5% variation
                features[f"{sensor}_{window}h_max"] = sensor_value * 1.1  # 10% higher
                features[f"{sensor}_{window}h_min"] = sensor_value * 0.9  # 10% lower

        # Create DataFrame with all required features
        feature_df = pd.DataFrame([features])

        # Ensure all model features are present (fill missing with 0)
        for feature_name in self.feature_names:
            if feature_name not in feature_df.columns:
                feature_df[feature_name] = 0.0

        # Select only the features the model expects
        feature_df = feature_df[self.feature_names]

        return feature_df

    def predict_single(self, request_data: Dict) -> Dict:
        """
        Make prediction for a single machine.

        Args:
            request_data: Dictionary with machine data

        Returns:
            Dictionary with prediction results
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded or not ready")

        try:
            # Create features
            features_df = self.create_features_from_request(request_data)

            # Make prediction
            failure_probability = self.model.predict_proba(features_df)[0, 1]
            failure_prediction = int(failure_probability > 0.5)

            # Determine risk level
            if failure_probability < 0.3:
                risk_level = "LOW"
            elif failure_probability < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            return {
                "machineID": request_data["machineID"],
                "failure_probability": float(failure_probability),
                "failure_prediction": failure_prediction,
                "risk_level": risk_level,
                "prediction_timestamp": datetime.now(),
            }

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_batch(self, requests_data: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple machines.

        Args:
            requests_data: List of machine data dictionaries

        Returns:
            List of prediction result dictionaries
        """
        if not self.is_ready():
            raise RuntimeError("Model not loaded or not ready")

        results = []
        for request_data in requests_data:
            try:
                result = self.predict_single(request_data)
                results.append(result)
            except Exception as e:
                self.logger.error(
                    f"Failed prediction for machine {request_data.get('machineID', 'unknown')}: {e}"
                )
                # Add error result
                results.append(
                    {
                        "machineID": request_data.get("machineID", -1),
                        "failure_probability": -1.0,
                        "failure_prediction": -1,
                        "risk_level": "ERROR",
                        "prediction_timestamp": datetime.now(),
                        "error": str(e),
                    }
                )

        return results

    def get_model_info(self) -> Optional[Dict]:
        """Get information about the loaded model."""
        return self.model_info
