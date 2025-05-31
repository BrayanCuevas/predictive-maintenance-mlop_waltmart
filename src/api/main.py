"""
FastAPI application for predictive maintenance model serving.
Provides REST API endpoints for machine failure predictions.

Author: Brayan Cuevas
Created for technical challenge implementation.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Metrics imports
from prometheus_client import generate_latest
from .metrics import (
    metrics_collector, 
    track_request_metrics, 
    track_prediction_metrics,
    api_health_status,
    model_loaded_status
)

from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from .predictor import ModelPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="ML API for predicting equipment failures using sensor data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
predictor: ModelPredictor = None
start_time = time.time()

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline_model.joblib")


@app.on_event("startup")
async def startup_event():
    """Initialize the model predictor on startup."""
    global predictor

    logger.info("Starting Predictive Maintenance API...")
    logger.info(f"Model path: {MODEL_PATH}")

    try:
        predictor = ModelPredictor(model_path=MODEL_PATH)

        if predictor.is_ready():
            model_info = predictor.get_model_info()
            logger.info("API startup completed successfully")
            logger.info(f"Model loaded: {model_info['model_name']}")
            logger.info(f"AUC Score: {model_info['auc_score']}")
            
            # Update initial metrics
            api_health_status.set(1)
            model_loaded_status.set(1)
        else:
            logger.error("Model failed to load during startup")
            api_health_status.set(0)
            model_loaded_status.set(0)

    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        api_health_status.set(0)
        model_loaded_status.set(0)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Predictive Maintenance API...")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Predictive Maintenance API",
        "version": "1.0.0",
        "author": "Brayan Cuevas",
        "description": "ML API for predicting equipment failures",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "metrics": "/metrics",
            "metrics_summary": "/metrics/summary",
            "docs": "/docs",
        },
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    global predictor
    
    # Update metrics before returning
    metrics_collector.collect_all_metrics(predictor)
    
    # Return Prometheus format
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


@app.get("/metrics/summary")
async def get_metrics_summary():
    """Human-readable metrics summary."""
    summary = metrics_collector.get_metrics_summary()
    
    return {
        "service": "Predictive Maintenance API",
        "status": "healthy" if predictor and predictor.is_ready() else "unhealthy",
        "metrics": summary,
        "model_info": predictor.get_model_info() if predictor and predictor.is_ready() else None
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    global predictor, start_time

    uptime = time.time() - start_time

    if predictor is None:
        return HealthResponse(
            status="unhealthy", model_loaded=False, model_info=None, uptime_seconds=uptime
        )

    model_info = predictor.get_model_info() if predictor.is_ready() else None

    return HealthResponse(
        status="healthy" if predictor.is_ready() else "unhealthy",
        model_loaded=predictor.is_ready(),
        model_info=model_info,
        uptime_seconds=uptime,
    )


@app.post("/predict", response_model=PredictionResponse)
@track_request_metrics
async def predict_failure(request: PredictionRequest):
    """
    Predict machine failure probability.

    Args:
        request: Machine sensor data

    Returns:
        Prediction results with probability and risk level
    """
    global predictor

    if predictor is None or not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded or not ready")

    try:
        # Convert request to dict
        request_data = request.dict()

        # Make prediction
        result = predictor.predict_single(request_data)
        
        # Track prediction metrics
        track_prediction_metrics(result)

        # Return structured response
        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
@track_request_metrics
async def predict_batch_failures(request: BatchPredictionRequest):
    """
    Predict failures for multiple machines.

    Args:
        request: Batch of machine sensor data

    Returns:
        Batch prediction results
    """
    global predictor

    if predictor is None or not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded or not ready")

    if len(request.predictions) > 100:
        raise HTTPException(
            status_code=400, detail="Batch size too large. Maximum 100 predictions per request."
        )

    try:
        start_time_batch = time.time()

        # Convert requests to list of dicts
        requests_data = [pred.dict() for pred in request.predictions]

        # Make batch predictions
        results = predictor.predict_batch(requests_data)

        # Convert results to response objects
        prediction_responses = []
        for result in results:
            if "error" not in result:
                # Track each prediction
                track_prediction_metrics(result)
                prediction_responses.append(PredictionResponse(**result))
            else:
                # Handle error cases in batch
                logger.warning(f"Error in batch prediction: {result['error']}")
                # Skip errored predictions or handle differently

        processing_time = time.time() - start_time_batch

        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_predictions=len(prediction_responses),
            processing_time_seconds=processing_time,
        )

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", response_model=dict)
async def get_model_info():
    """Get information about the loaded model."""
    global predictor

    if predictor is None or not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_info = predictor.get_model_info()
    return {
        "model_info": model_info,
        "feature_count": len(predictor.feature_names),
        "sample_features": predictor.feature_names[:10] if predictor.feature_names else [],
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# For development/testing
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting development server on {host}:{port}")

    uvicorn.run("main:app", host=host, port=port, reload=True, log_level="info")