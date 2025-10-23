# Pydantic models for the web service

from typing import List, Optional
from pydantic import BaseModel, Field


class AbaloneFeatures(BaseModel):
    """Input features for abalone age prediction."""

    sex: str = Field(..., description="Sex of the abalone (M/F/I)", regex="^[MFI]$")
    length: float = Field(..., description="Longest shell measurement", gt=0)
    diameter: float = Field(..., description="Perpendicular to length", gt=0)
    height: float = Field(..., description="With meat in shell", gt=0)
    whole_weight: float = Field(..., description="Whole abalone weight", gt=0)
    shucked_weight: float = Field(..., description="Weight of meat", gt=0)
    viscera_weight: float = Field(..., description="Gut weight (after bleeding)", gt=0)
    shell_weight: float = Field(..., description="After being dried", gt=0)

    class Config:
        """Pydantic configuration for AbaloneFeatures."""

        schema_extra = {
            "example": {
                "sex": "M",
                "length": 0.455,
                "diameter": 0.365,
                "height": 0.095,
                "whole_weight": 0.514,
                "shucked_weight": 0.2245,
                "viscera_weight": 0.101,
                "shell_weight": 0.15,
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predicted_age: float = Field(..., description="Predicted age of the abalone")
    confidence: Optional[float] = Field(None, description="Prediction confidence score")
    model_version: Optional[str] = Field(None, description="Version of the model used")

    class Config:
        """Pydantic configuration for PredictionResponse."""

        schema_extra = {"example": {"predicted_age": 9.5, "confidence": 0.85, "model_version": "v1.0.0"}}


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    instances: List[AbaloneFeatures] = Field(..., description="List of abalone instances to predict")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_predictions: int = Field(..., description="Total number of predictions made")


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Health check message")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    timestamp: str = Field(..., description="Current timestamp")
