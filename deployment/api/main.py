"""FastAPI application for Rossmann sales forecasting predictions.

This module provides a REST API for:
- Health checks
- Sales prediction endpoints (coming soon)
- Model metadata and version information
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{PROJECT_ROOT}/mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Import prediction logger
from prediction_logger import PredictionLogger
from utils.io import read_csv

from data.prepare_predictions import prepare_prediction_data, validate_input_data
from models.model_registry import get_model_version, list_registered_models, load_model

logger = logging.getLogger(__name__)

# Global model cache
_model_cache = {"model": None, "version": None, "stage": None, "loaded_at": None}

# Global store metadata cache (loaded once at startup)
_store_metadata = None

# Global prediction logger
_prediction_logger = None

# Initialize FastAPI app
app = FastAPI(
    title="Rossmann Sales Forecasting API",
    description="Production API for predicting daily sales across 3,000+ European stores using ensemble ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether a model is loaded in cache")
    model_version: str | None = Field(None, description="Loaded model version")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    registered_models: list[str] = Field(..., description="List of registered model names")
    production_version: str | None = Field(None, description="Production model version")
    staging_version: str | None = Field(None, description="Staging model version")
    mlflow_uri: str = Field(..., description="MLflow tracking URI")
    timestamp: str = Field(..., description="Response timestamp")


class ModelLoadResponse(BaseModel):
    """Response model for model loading."""

    status: str = Field(..., description="Load status")
    model_version: str = Field(..., description="Loaded model version")
    stage: str = Field(..., description="Model stage loaded")
    loaded_at: str = Field(..., description="Timestamp when model was loaded")
    timestamp: str = Field(..., description="Response timestamp")


class PredictionInput(BaseModel):
    """Input features for a single sales prediction.

    Matches the format of train.csv - the API will automatically:
    1. Merge with store metadata
    2. Clean the data
    3. Engineer features
    4. Make predictions
    """

    Store: int = Field(..., ge=1, description="Store ID (1-1115)")
    DayOfWeek: int = Field(..., ge=1, le=7, description="Day of week (1=Monday, 7=Sunday)")
    Date: str = Field(..., description="Date in YYYY-MM-DD format")
    Open: int = Field(..., ge=0, le=1, description="Is store open? (0=closed, 1=open)")
    Promo: int = Field(..., ge=0, le=1, description="Is promotion running? (0=no, 1=yes)")
    StateHoliday: str = Field("0", description="State holiday indicator (0, a, b, c)")
    SchoolHoliday: int = Field(..., ge=0, le=1, description="Is school holiday? (0=no, 1=yes)")


class PredictionRequest(BaseModel):
    """Request model for batch predictions."""

    inputs: list[PredictionInput] = Field(..., min_length=1, description="List of feature sets")
    model_stage: str = Field("Production", description="Model stage (Production/Staging/version)")


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predictions: list[float] = Field(..., description="Predicted sales values")
    model_version: str = Field(..., description="Model version used")
    count: int = Field(..., description="Number of predictions")
    timestamp: str = Field(..., description="Prediction timestamp")


# Helper functions
def get_or_load_model(stage: str = "Production"):
    """Get model from cache or load it.

    Parameters
    ----------
    stage : str
        Model stage to load (Production, Staging, or version number)

    Returns
    -------
    tuple
        (model, version)
    """
    global _model_cache

    # Get the current version for this stage
    if stage in ["Production", "Staging"]:
        current_version = str(get_model_version("rossmann-ensemble", stage=stage))
    else:
        current_version = stage

    # Check if we need to reload
    if _model_cache["model"] is None or _model_cache["version"] != current_version:
        logger.info(f"Loading model version {current_version} from stage {stage}")
        model = load_model("rossmann-ensemble", stage=stage)
        _model_cache = {
            "model": model,
            "version": current_version,
            "stage": stage,
            "loaded_at": datetime.now().isoformat(),
        }
        logger.info(f"Model loaded successfully: version {current_version}")

    return _model_cache["model"], _model_cache["version"]


# Startup event
@app.on_event("startup")
async def startup_event():
    """Log startup information and load store metadata."""
    global _store_metadata, _prediction_logger

    logger.info("=" * 70)
    logger.info("Rossmann Sales Forecasting API Starting")
    logger.info("=" * 70)
    logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"Project Root: {PROJECT_ROOT}")

    # Initialize prediction logger
    try:
        db_path = PROJECT_ROOT / "data" / "monitoring" / "predictions.db"
        _prediction_logger = PredictionLogger(db_path)
        logger.info(f"Prediction logger initialized at {db_path}")
    except Exception as e:
        logger.error(f"Failed to initialize prediction logger: {e}")
        _prediction_logger = None

    # Load store metadata
    try:
        store_path = PROJECT_ROOT / "data" / "raw" / "store.csv"
        logger.info(f"Loading store metadata from {store_path}")
        _store_metadata = read_csv(store_path)
        logger.info(f"Loaded metadata for {len(_store_metadata)} stores")
    except Exception as e:
        logger.error(f"Failed to load store metadata: {e}")
        _store_metadata = None

    # Check MLflow models
    try:
        models = list_registered_models()
        logger.info(f"Registered models: {models}")
        if "rossmann-ensemble" in models:
            prod_version = get_model_version("rossmann-ensemble", "Production")
            staging_version = get_model_version("rossmann-ensemble", "Staging")
            logger.info(f"Production version: {prod_version}")
            logger.info(f"Staging version: {staging_version}")
    except Exception as e:
        logger.warning(f"Could not load model info: {e}")

    logger.info("Target: RMSPE < 0.09856 (Top 50 Kaggle leaderboard)")
    logger.info("=" * 70)


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Rossmann Sales Forecasting API",
        "version": "1.0.0",
        "description": "Production API for predicting daily sales across 3,000+ European stores",
        "model": "Ensemble (LightGBM + XGBoost + CatBoost)",
        "target_metric": "RMSPE < 0.09856",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "model_load": "/model/load",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint.

    Returns
    -------
    HealthResponse
        Health status information
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=_model_cache["model"] is not None,
        model_version=_model_cache.get("version"),
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about registered models.

    Returns
    -------
    ModelInfoResponse
        Model registry information
    """
    try:
        models = list_registered_models()
        prod_version = None
        staging_version = None

        if "rossmann-ensemble" in models:
            try:
                prod_version = str(get_model_version("rossmann-ensemble", "Production"))
            except Exception:
                pass
            try:
                staging_version = str(get_model_version("rossmann-ensemble", "Staging"))
            except Exception:
                pass

        return ModelInfoResponse(
            registered_models=models,
            production_version=prod_version,
            staging_version=staging_version,
            mlflow_uri=MLFLOW_TRACKING_URI,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}") from e


@app.post("/model/load", response_model=ModelLoadResponse)
async def load_model_endpoint(stage: str = "Production"):
    """Load a model into memory.

    Parameters
    ----------
    stage : str
        Model stage to load (Production, Staging, or version number)

    Returns
    -------
    ModelLoadResponse
        Model loading status and metadata
    """
    try:
        logger.info(f"Loading model for stage: {stage}")
        model, version = get_or_load_model(stage)

        return ModelLoadResponse(
            status="loaded",
            model_version=version,
            stage=stage,
            loaded_at=_model_cache["loaded_at"],
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}") from e


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate sales predictions using the loaded model.

    This endpoint mimics a production batch prediction workflow:
    1. Accepts raw data (matching train.csv format)
    2. Validates input data
    3. Prepares features using the prediction pipeline
    4. Makes predictions
    5. Logs predictions for monitoring

    Parameters
    ----------
    request : PredictionRequest
        Prediction request with raw input data and model stage

    Returns
    -------
    PredictionResponse
        Predicted sales values and metadata
    """
    start_time = time.time()

    try:
        # Check store metadata is loaded
        if _store_metadata is None:
            raise HTTPException(
                status_code=500, detail="Store metadata not loaded. Please restart the API server."
            )

        # Load model if needed
        model, version = get_or_load_model(request.model_stage)

        logger.info(f"Processing {len(request.inputs)} raw inputs for prediction")

        # Step 1: Convert inputs to DataFrame (raw train.csv format)
        input_dicts = [inp.model_dump() for inp in request.inputs]
        raw_df = pd.DataFrame(input_dicts)

        # Step 2: Validate input data
        is_valid, errors = validate_input_data(raw_df)
        if not is_valid:
            raise HTTPException(status_code=422, detail=f"Input validation failed: {errors}")

        logger.info("✓ Input data validated successfully")

        # Step 3: Prepare prediction features using unified pipeline
        # This handles: merging store metadata, cleaning, feature engineering, and column selection
        # Note: We need to get the full feature set BEFORE column selection for logging
        from features.build_features import build_all_features

        from data.make_dataset import basic_cleaning, merge_store_info

        # Create a copy with dummy Sales/Customers for full feature generation
        raw_df_with_dummy = raw_df.copy()
        if "Sales" not in raw_df_with_dummy.columns:
            raw_df_with_dummy["Sales"] = 0
        if "Customers" not in raw_df_with_dummy.columns:
            raw_df_with_dummy["Customers"] = 0

        # Merge and clean data
        merged_df = merge_store_info(raw_df_with_dummy, _store_metadata)
        cleaned_df = basic_cleaning(merged_df)

        # Engineer features (get FULL feature set for logging)
        full_features = build_all_features(cleaned_df)

        # Select model features (subset for prediction)
        model_input = prepare_prediction_data(raw_df, store_metadata=_store_metadata)

        logger.info(
            f"✓ Features prepared: {len(model_input)} samples, {model_input.shape[1]} features"
        )
        logger.info(f"Making predictions with model version {version}")

        # Step 4: Make predictions
        predictions = model.predict(model_input)

        # Convert to list for JSON serialization
        predictions_list = (
            predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
        )

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000

        logger.info(f"✓ Prediction complete: {len(predictions_list)} predictions generated")

        # Step 5: Log predictions for monitoring (if logger available)
        if _prediction_logger is not None:
            try:
                batch_id = _prediction_logger.log_predictions(
                    raw_inputs=raw_df,
                    features=full_features,
                    predictions=predictions_list,
                    model_version=version,
                    model_stage=request.model_stage,
                    response_time_ms=response_time_ms,
                )
                logger.info(f"✓ Predictions logged (batch_id={batch_id})")
            except Exception as e:
                logger.warning(f"Failed to log predictions: {e}")

        return PredictionResponse(
            predictions=predictions_list,
            model_version=version,
            count=len(predictions_list),
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from e


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
