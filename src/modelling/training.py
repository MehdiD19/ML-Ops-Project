import mlflow
import mlflow.sklearn
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Any, Tuple
import pandas as pd
from prefect import task, flow
from preprocessing import preprocess_data
from utils import pickle_object


@task(name="Training linear model.")
def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


@task(name="Training random forest.")
def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Train a Random Forest model with regularization."""
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
    model.fit(X_train, y_train)
    return model


@task(name="Evaluating model.")
def evaluate_model(
    model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> Tuple[dict, Any]:
    """Evaluate model performance on train and test sets."""
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "test_r2": r2_score(y_test, y_test_pred),
    }

    return metrics, y_test_pred


@flow(name="Training workflow (preprocessing, training and evaluation.)")
def train_model(data_path: str, model_type: str = "linear_regression") -> None:
    """Complete training pipeline with MLflow tracking."""
    # Set up MLflow experiment
    mlflow.set_experiment("abalone_age_prediction")

    with mlflow.start_run():
        # Preprocess data
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data_path)

        # Train model
        if model_type == "linear_regression":
            model = train_linear_regression(X_train, y_train)
        elif model_type == "random_forest":
            model = train_random_forest(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Evaluate model
        metrics, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

        # Log metrics to MLflow
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save model and scaler as pickle files
        model_dir = Path("src/web_service/local_objects")
        pickle_object(model, model_dir / f"{model_type}_model.pkl")
        pickle_object(scaler, model_dir / "scaler.pkl")

        # Print results
        print(f"\n{model_type.replace('_', ' ').title()} Results:")
        print(f"Train RMSE: {metrics['train_rmse']:.4f}")
        print(f"Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.4f}")
        print(f"Test R2: {metrics['test_r2']:.4f}")

        # Calculate metrics in original scale (rings/age)
        y_test_orig = np.expm1(y_test)
        y_pred_test_orig = np.expm1(y_test_pred)

        test_rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
        test_mae_orig = mean_absolute_error(y_test_orig, y_pred_test_orig)

        print(f"Test RMSE (original scale): {test_rmse_orig:.4f}")
        print(f"Test MAE (original scale): {test_mae_orig:.4f}")
