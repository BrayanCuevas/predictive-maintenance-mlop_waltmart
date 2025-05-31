"""
API module for predictive maintenance model serving.
FastAPI application for real-time predictions.

Author: Brayan Cuevas
Created for technical challenge implementation.
"""

from .main import app
from .predictor import ModelPredictor
from .schemas import PredictionRequest, PredictionResponse

__all__ = ["app", "ModelPredictor", "PredictionRequest", "PredictionResponse"]
