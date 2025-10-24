# Inference logic for the ML model

import pickle
import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from .models import AbaloneFeatures, PredictionResponse

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handles model loading and predictions."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """Initialize the predictor with optional model path."""
        self.model = None
        self.scaler = None
        self.model_version = "v1.0.0"
        self.is_loaded = False

        if model_path is None:
            # robust path: from this file → lib → (parent) web_service → local_objects
            model_path = Path(__file__).resolve().parent.parent / "local_objects"

        self.model_path = Path(model_path)
        self.load_model()

    def load_model(self) -> bool:
        """Load the trained model and scaler from pickle files."""
        try:
            model_file = self.model_path / "linear_regression_model.pkl"
            scaler_file = self.model_path / "scaler.pkl"

            if model_file.exists():
                with model_file.open("rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"Model loaded successfully from {model_file}")
            else:
                logger.warning(f"Model file not found at {model_file}")
                return False

            if scaler_file.exists():
                with scaler_file.open("rb") as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Scaler loaded successfully from {scaler_file}")
            else:
                logger.info("No scaler file found; proceeding without feature scaling.")

            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False

    def preprocess_features(self, features: AbaloneFeatures) -> np.ndarray:
        """Preprocess input features for prediction."""
        # Convert Pydantic model to dictionary
        feature_dict = features.dict()

        # Create DataFrame for consistent preprocessing
        df = pd.DataFrame([feature_dict])

        # Encode 'sex' if present (simple mapping)
        if "sex" in df.columns:
            sex_mapping = {"M": 0, "F": 1, "I": 2}
            df["sex"] = df["sex"].map(sex_mapping)

        # Apply scaler to numeric columns if available
        if self.scaler is not None:
            try:
                # Prefer scaler's feature_names_in_ if present
                if hasattr(self.scaler, "feature_names_in_"):
                    cols_to_scale = [c for c in self.scaler.feature_names_in_ if c in df.columns]
                    if cols_to_scale:
                        df.loc[:, cols_to_scale] = self.scaler.transform(df[cols_to_scale])
                else:
                    # Fallback: scale all numeric columns
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if num_cols:
                        df.loc[:, num_cols] = self.scaler.transform(df[num_cols])
            except Exception as e:
                logger.warning(f"Scaling failed, proceeding unscaled. Reason: {e}")

        return df.values

    def predict_single(self, features: AbaloneFeatures) -> PredictionResponse:
        """Make a single prediction."""
        if not self.is_loaded or self.model is None:
            raise ValueError("Model is not loaded. Please check model files.")

        try:
            processed_features = self.preprocess_features(features)
            prediction = self.model.predict(processed_features)[0]

            # Linear regression: no predict_proba
            confidence = None

            return PredictionResponse(
                predicted_age=float(prediction),
                confidence=confidence,
                model_version=self.model_version
            )

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")

    def predict_batch(self, features_list: List[AbaloneFeatures]) -> List[PredictionResponse]:
        """Make batch predictions."""
        if not self.is_loaded or self.model is None:
            raise ValueError("Model is not loaded. Please check model files.")

        predictions = []
        for features in features_list:
            try:
                prediction = self.predict_single(features)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error in batch prediction: {str(e)}")
                predictions.append(
                    PredictionResponse(
                        predicted_age=0.0,
                        confidence=0.0,
                        model_version=self.model_version
                    )
                )

        return predictions


# Global predictor instance
predictor = ModelPredictor()
