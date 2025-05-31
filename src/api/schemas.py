"""
Pydantic schemas for API request/response validation.

Author: Brayan Cuevas
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for machine failure prediction."""

    machineID: int = Field(..., description="Machine identifier", example=1)
    volt: float = Field(..., description="Voltage reading", example=150.5)
    rotate: float = Field(..., description="Rotation speed", example=480.2)
    pressure: float = Field(..., description="Pressure reading", example=95.1)
    vibration: float = Field(..., description="Vibration level", example=38.7)

    class Config:
        schema_extra = {
            "example": {
                "machineID": 1,
                "volt": 150.5,
                "rotate": 480.2,
                "pressure": 95.1,
                "vibration": 38.7,
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""

    machineID: int = Field(..., description="Machine identifier")
    failure_probability: float = Field(..., description="Probability of failure (0-1)")
    failure_prediction: int = Field(..., description="Binary prediction (0=no failure, 1=failure)")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    prediction_timestamp: datetime = Field(..., description="When prediction was made")

    class Config:
        schema_extra = {
            "example": {
                "machineID": 1,
                "failure_probability": 0.23,
                "failure_prediction": 0,
                "risk_level": "LOW",
                "prediction_timestamp": "2024-12-30T15:30:00",
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_info: Optional[dict] = Field(None, description="Model metadata")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""

    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "machineID": 1,
                        "volt": 150.5,
                        "rotate": 480.2,
                        "pressure": 95.1,
                        "vibration": 38.7,
                    },
                    {
                        "machineID": 2,
                        "volt": 148.3,
                        "rotate": 475.8,
                        "pressure": 97.2,
                        "vibration": 42.1,
                    },
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema."""

    predictions: List[PredictionResponse] = Field(..., description="List of prediction results")
    total_predictions: int = Field(..., description="Total number of predictions made")
    processing_time_seconds: float = Field(..., description="Time taken to process batch")
