# FastAPI application for Abalone Age Prediction

import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .lib.models import AbaloneFeatures, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse, HealthResponse
from .lib.inference import predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Abalone Age Prediction API",
    description="A machine learning API to predict the age of abalone using physical measurements",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
def home() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        message="Abalone Age Prediction API is up and running!",
        model_loaded=predictor.is_loaded,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Detailed health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "unhealthy",
        message="Model loaded successfully" if predictor.is_loaded else "Model not loaded",
        model_loaded=predictor.is_loaded,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(payload: AbaloneFeatures) -> PredictionResponse:
    """Make a single prediction for abalone age."""
    try:
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded. Please check server logs."
            )

        prediction = predictor.predict_single(payload)
        logger.info(f"Prediction made: {prediction.predicted_age}")
        return prediction

    except ValueError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Prediction failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error occurred")


@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
    """Make batch predictions for multiple abalone instances."""
    try:
        if not predictor.is_loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not loaded. Please check server logs."
            )

        if len(payload.instances) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No instances provided for prediction")

        if len(payload.instances) > 100:  # Limit batch size
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Batch size too large. Maximum 100 instances allowed."
            )

        predictions = predictor.predict_batch(payload.instances)
        logger.info(f"Batch prediction made for {len(predictions)} instances")

        return BatchPredictionResponse(predictions=predictions, total_predictions=len(predictions))

    except ValueError as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Batch prediction failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in batch prediction: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error occurred")


@app.get("/model/info")
def model_info() -> dict:
    """Get information about the loaded model."""
    return {
        "model_loaded": predictor.is_loaded,
        "model_version": predictor.model_version,
        "model_path": str(predictor.model_path),
        "timestamp": datetime.now().isoformat(),
    }