# Inference logic for the ML model

import pickle
import logging
from pathlib import Path
from typing import List, Optional, Tuple
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
            # Robust path relative to this file: lib -> (parent) web_service -> local_objects
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

            # Diagnostics: what does the model/scaler expect?
            try:
                if hasattr(self.model, "feature_names_in_"):
                    logger.info(f"Model expects columns: {list(self.model.feature_names_in_)}")
                if hasattr(self.model, "n_features_in_"):
                    logger.info(f"Model n_features_in_: {self.model.n_features_in_}")
            except Exception as e:
                logger.warning(f"Could not introspect model features: {e}")

            try:
                if self.scaler is not None and hasattr(self.scaler, "feature_names_in_"):
                    logger.info(f"Scaler expects columns: {list(self.scaler.feature_names_in_)}")
            except Exception as e:
                logger.warning(f"Could not introspect scaler features: {e}")

            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False

    # ---------------------------
    # Feature engineering helpers
    # ---------------------------

    @staticmethod
    def _sex_to_int(series: pd.Series) -> pd.Series:
        """Map sex categorical to a single integer feature."""
        return series.map({"M": 0, "F": 1, "I": 2}).fillna(0).astype(float)

    @staticmethod
    def _add_missing_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Ensure all columns exist in df; create missing as 0."""
        for col in columns:
            if col not in df.columns:
                df[col] = 0.0
        return df

    def _build_from_expected_columns(
        self, row: pd.DataFrame, expected_cols: List[str]
    ) -> pd.DataFrame:
        """
        Build exactly the columns the scaler/model expects.
        - If it expects sex dummies (sex_*), create them.
        - Else if it expects a single 'sex' column, map to int.
        """
        row = row.copy()
        sex_dummy_cols = [c for c in expected_cols if c.startswith("sex_")]

        if sex_dummy_cols:
            # Remove raw 'sex' and create all dummies = 0
            sex_value = row.at[0, "sex"] if "sex" in row.columns else None
            if "sex" in row.columns:
                row = row.drop(columns=["sex"])
            for col in sex_dummy_cols:
                row[col] = 0.0
            if sex_value is not None:
                target = f"sex_{sex_value}"
                if target in sex_dummy_cols:
                    row.at[0, target] = 1.0
            # If baseline category (drop_first) is used, all zeros is correct.
        else:
            # Single 'sex' expected
            if "sex" in expected_cols:
                if "sex" in row.columns:
                    row["sex"] = self._sex_to_int(row["sex"])
                else:
                    row["sex"] = 0.0  # default baseline

        # Ensure all expected columns exist, then order them
        row = self._add_missing_columns(row, expected_cols)
        row = row[expected_cols]
        return row

    def _build_candidate_matrices(self, row: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        """
        Build several candidate matrices when no expected column names are available.
        We will later select the one matching model.n_features_in_ (if available).
        Returns a list of (label, df) candidates.
        """
        candidates: List[Tuple[str, pd.DataFrame]] = []

        # Base numeric-only (drop raw 'sex')
        base = row.copy()
        if "sex" in base.columns:
            base = base.drop(columns=["sex"])
        candidates.append(("numeric_only", base))

        # Sex as single int column
        with_sex_int = base.copy()
        with_sex_int["sex"] = self._sex_to_int(row["sex"]) if "sex" in row.columns else 0.0
        candidates.append(("sex_as_int", with_sex_int))

        # One-hot without drop_first: sex_F, sex_I, sex_M
        one_hot_cols = ["sex_F", "sex_I", "sex_M"]
        one_hot = base.copy()
        for c in one_hot_cols:
            one_hot[c] = 0.0
        if "sex" in row.columns:
            val = row.at[0, "sex"]
            target = f"sex_{val}"
            if target in one_hot.columns:
                one_hot.at[0, target] = 1.0
        candidates.append(("one_hot_full", one_hot))

        # One-hot with drop_first (assume baseline F -> keep I and M)
        one_hot_drop = base.copy()
        for c in ["sex_I", "sex_M"]:
            one_hot_drop[c] = 0.0
        if "sex" in row.columns:
            val = row.at[0, "sex"]
            target = f"sex_{val}"
            if target in one_hot_drop.columns:
                one_hot_drop.at[0, target] = 1.0
            # if val == 'F', both stay 0 (baseline)
        candidates.append(("one_hot_dropF", one_hot_drop))

        return candidates

    def _select_best_candidate(
        self, candidates: List[Tuple[str, pd.DataFrame]]
    ) -> pd.DataFrame:
        """
        Select the candidate whose number of columns matches model.n_features_in_ if available,
        otherwise return the widest (max columns) as a safer default.
        """
        if hasattr(self.model, "n_features_in_"):
            need = int(self.model.n_features_in_)
            for label, df in candidates:
                if df.shape[1] == need:
                    logger.info(f"Selected candidate '{label}' matching n_features_in_={need}")
                    return df
            # No exact match; log and fall back to closest by absolute difference
            labeled = [(label, df, abs(df.shape[1] - need)) for label, df in candidates]
            labeled.sort(key=lambda x: x[2])
            chosen = labeled[0]
            logger.warning(
                f"No exact candidate feature count match (need {need}). "
                f"Choosing '{chosen[0]}' with shape {chosen[1].shape}"
            )
            return chosen[1]
        else:
            # No clue about target width; prefer widest candidate
            candidates_sorted = sorted(candidates, key=lambda x: x[1].shape[1], reverse=True)
            logger.info(
                f"Model has no n_features_in_; selecting widest candidate '{candidates_sorted[0][0]}' "
                f"with shape {candidates_sorted[0][1].shape}"
            )
            return candidates_sorted[0][1]

    # ---------------------------
    # Main preprocessing
    # ---------------------------

    def preprocess_features(self, features: AbaloneFeatures) -> np.ndarray:
        """Preprocess input features to match training schema exactly, with robust fallbacks."""
        # 1) Build base row from input
        row = pd.DataFrame([features.dict()])

        # 2) Determine expected columns from scaler or model if available
        expected_cols = None
        expected_source = None
        if self.scaler is not None and hasattr(self.scaler, "feature_names_in_"):
            expected_cols = list(self.scaler.feature_names_in_)
            expected_source = "scaler"
        elif hasattr(self.model, "feature_names_in_"):
            expected_cols = list(self.model.feature_names_in_)
            expected_source = "model"

        # 3) Build the design matrix
        if expected_cols is not None:
            X = self._build_from_expected_columns(row, expected_cols)
            logger.debug(f"Preprocess using {expected_source} schema, shape={X.shape}")
        else:
            # No explicit schema available: try multiple candidates and select best
            cands = self._build_candidate_matrices(row)
            X = self._select_best_candidate(cands)
            logger.debug(f"Preprocess using auto-selected candidate, shape={X.shape}")

        # 4) Apply scaler if available (keep DataFrame with same columns)
        if self.scaler is not None:
            try:
                transformed = self.scaler.transform(X.values)
                X = pd.DataFrame(transformed, columns=X.columns, index=X.index)
            except Exception as e:
                logger.warning(f"Scaling failed, proceeding unscaled. Reason: {e}")

        # 5) Final sanity check / adjustment
        if hasattr(self.model, "n_features_in_"):
            need = int(self.model.n_features_in_)
            have = X.shape[1]
            if have != need:
                logger.warning(f"Feature count mismatch before predict: have {have}, need {need}. Attempting adjustment.")
                # Try to adjust by padding zeros or trimming extras (last columns)
                if have < need:
                    # Pad with dummy columns
                    to_add = need - have
                    for i in range(to_add):
                        X[f"_pad_{i}"] = 0.0
                else:
                    # Trim extra columns from the end
                    X = X.iloc[:, :need]
                logger.info(f"Adjusted feature matrix shape to {X.shape}")

        logger.debug(f"Final feature matrix shape: {X.shape}")
        return X.values  # shape (1, n_features)

    # ---------------------------
    # Prediction methods
    # ---------------------------

    def predict_single(self, features: AbaloneFeatures) -> PredictionResponse:
        """Make a single prediction."""
        if not self.is_loaded or self.model is None:
            raise ValueError("Model is not loaded. Please check model files.")

        try:
            X = self.preprocess_features(features)
            prediction = self.model.predict(X)[0]

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
