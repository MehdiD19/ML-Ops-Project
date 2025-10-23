import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Tuple, Any

from .utils import load_pickle_object


def load_model_and_scaler(model_path: Path, scaler_path: Path) -> Tuple[Any, Any]:
    """Load the trained model and scaler from pickle files."""
    model = load_pickle_object(model_path)
    scaler = load_pickle_object(scaler_path)
    return model, scaler


def preprocess_for_prediction(data: Union[pd.DataFrame, dict], scaler: Any) -> pd.DataFrame:
    """Preprocess new data for prediction using the same transformations as training."""
    if isinstance(data, dict):
        data = pd.DataFrame([data])

    df = data.copy()

    # Apply the same feature engineering as in training
    df["Age"] = df["Rings"] + 1.5  # This won't be used for prediction but needed for consistency

    # Remove invalid samples
    df = df[df["Height"] > 0]
    df = df[df["Whole weight"] > 0]

    # One Hot Encoding of sex features
    df = pd.get_dummies(df, columns=["Sex"], drop_first=False)

    # Create ratio features
    df["density"] = df["Whole weight"] / (df["Length"] * df["Diameter"] * df["Height"])
    df["meat_ratio"] = df["Shucked weight"] / df["Whole weight"]
    df["shell_ratio"] = df["Shell weight"] / df["Whole weight"]
    df["water_loss"] = df["Whole weight"] - df["Shucked weight"] - df["Viscera weight"] - df["Shell weight"]

    # Drop original measurement columns
    columns_to_drop = ["Length", "Diameter", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings", "Age"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Apply log transformations
    df["Height_log"] = np.log1p(df["Height"])
    df["density_log"] = np.log1p(df["density"])
    df["meat_ratio_log"] = np.log1p(df["meat_ratio"])
    df["shell_ratio_log"] = np.log1p(df["shell_ratio"])

    # Drop original versions after log transform
    df = df.drop(columns=["Height", "density", "meat_ratio", "shell_ratio"])

    # Scale water_loss feature using the fitted scaler
    df[["water_loss"]] = scaler.transform(df[["water_loss"]])

    return df


def predict_age(
    data: Union[pd.DataFrame, dict],
    model_path: str = "src/web_service/local_objects/linear_regression_model.pkl",
    scaler_path: str = "src/web_service/local_objects/scaler.pkl",
) -> Union[float, List[float]]:
    """Make age predictions on new data.

    Args:
        data: Input data as DataFrame or dictionary
        model_path: Path to the pickled model
        scaler_path: Path to the pickled scaler

    Returns:
        Predicted age(s) in original scale (years)
    """
    # Load model and scaler
    model, scaler = load_model_and_scaler(Path(model_path), Path(scaler_path))

    # Preprocess data
    X = preprocess_for_prediction(data, scaler)

    # Make predictions (in log scale)
    y_pred_log = model.predict(X)

    # Convert back to original scale
    y_pred = np.expm1(y_pred_log)

    # Return single value if single prediction, list otherwise
    if len(y_pred) == 1:
        return float(y_pred[0])
    else:
        return y_pred.tolist()


def predict_rings(
    data: Union[pd.DataFrame, dict],
    model_path: str = "src/web_service/local_objects/linear_regression_model.pkl",
    scaler_path: str = "src/web_service/local_objects/scaler.pkl",
) -> Union[int, List[int]]:
    """Make ring count predictions on new data.

    Args:
        data: Input data as DataFrame or dictionary
        model_path: Path to the pickled model
        scaler_path: Path to the pickled scaler

    Returns:
        Predicted ring count(s) (Age - 1.5)
    """
    age_predictions = predict_age(data, model_path, scaler_path)

    if isinstance(age_predictions, float):
        return int(round(age_predictions - 1.5))
    else:
        return [int(round(age - 1.5)) for age in age_predictions]
