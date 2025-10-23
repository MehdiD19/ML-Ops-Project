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
        self.encoder = None
        self.model_version = "v1.0.0"
        self.is_loaded = False

        if model_path is None:
            # Default path where the model should be saved
            model_path = Path("src/web_service/local_objects")

        self.model_path = Path(model_path)
        self.load_model()

    def load_model(self) -> bool:
        """Load the trained model and encoder from pickle files."""
        try:
            model_file = self.model_path / "model.pkl"
            encoder_file = self.model_path / "encoder.pkl"

            if model_file.exists():
                with model_file.open("rb") as f:
                    self.model = pickle.load(f)
                logger.info(f"Model loaded successfully from {model_file}")
            else:
                logger.warning(f"Model file not found at {model_file}")
                return False

            if encoder_file.exists():
                with encoder_file.open("rb") as f:
                    self.encoder = pickle.load(f)
                logger.info(f"Encoder loaded successfully from {encoder_file}")
            else:
                logger.info("No encoder file found, assuming no categorical encoding needed")

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

        # Handle categorical encoding if encoder exists
        if self.encoder is not None:
            # Encode categorical features (e.g., sex)
            if "sex" in df.columns:
                df["sex"] = self.encoder.transform(df[["sex"]])
        else:
            # Simple encoding for sex if no encoder is available
            if "sex" in df.columns:
                sex_mapping = {"M": 0, "F": 1, "I": 2}
                df["sex"] = df["sex"].map(sex_mapping)

        return df.values

    def predict_single(self, features: AbaloneFeatures) -> PredictionResponse:
        """Make a single prediction."""
        if not self.is_loaded or self.model is None:
            raise ValueError("Model is not loaded. Please check model files.")

        try:
            # Preprocess features
            processed_features = self.preprocess_features(features)

            # Make prediction
            prediction = self.model.predict(processed_features)[0]

            # Calculate confidence if the model supports it
            confidence = None
            if hasattr(self.model, "predict_proba"):
                try:
                    proba = self.model.predict_proba(processed_features)[0]
                    confidence = float(np.max(proba))
                except Exception:
                    pass

            return PredictionResponse(predicted_age=float(prediction), confidence=confidence, model_version=self.model_version)

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
                # You might want to handle this differently based on requirements
                predictions.append(PredictionResponse(predicted_age=0.0, confidence=0.0, model_version=self.model_version))

        return predictions


# Global predictor instance
predictor = ModelPredictor()
