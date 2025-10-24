import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
from prefect import task


def load_data(data_path: str) -> pd.DataFrame:
    """Load abalone dataset from CSV file."""
    return pd.read_csv(data_path)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering steps from the notebook."""
    df_processed = df.copy()

    # Create Age variable
    df_processed["Age"] = df_processed["Rings"] + 1.5

    # Remove samples with invalid height
    df_processed = df_processed[df_processed["Height"] > 0]

    # One Hot Encoding of sex features
    df_processed = pd.get_dummies(df_processed, columns=["Sex"], drop_first=False)

    # Ensure no division by zero
    df_processed = df_processed[df_processed["Whole weight"] > 0]

    # Create ratio features
    df_processed["density"] = df_processed["Whole weight"] / (
        df_processed["Length"] * df_processed["Diameter"] * df_processed["Height"]
    )
    df_processed["meat_ratio"] = df_processed["Shucked weight"] / df_processed["Whole weight"]
    df_processed["shell_ratio"] = df_processed["Shell weight"] / df_processed["Whole weight"]
    df_processed["water_loss"] = (
        df_processed["Whole weight"]
        - df_processed["Shucked weight"]
        - df_processed["Viscera weight"]
        - df_processed["Shell weight"]
    )

    # Drop original measurement columns
    columns_to_drop = ["Length", "Diameter", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight"]
    df_processed = df_processed.drop(columns=columns_to_drop)

    # Apply log transformations
    df_processed["Age_log"] = np.log1p(df_processed["Age"])
    df_processed["Height_log"] = np.log1p(df_processed["Height"])
    df_processed["density_log"] = np.log1p(df_processed["density"])
    df_processed["meat_ratio_log"] = np.log1p(df_processed["meat_ratio"])
    df_processed["shell_ratio_log"] = np.log1p(df_processed["shell_ratio"])

    # Drop original versions after log transform
    df_processed = df_processed.drop(columns=["Rings", "Height", "density", "meat_ratio", "shell_ratio"])

    # Scale water_loss feature
    scaler = StandardScaler()
    df_processed[["water_loss"]] = scaler.fit_transform(df_processed[["water_loss"]])

    return df_processed, scaler


def prepare_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    # Separate features and target
    X = df.drop(columns=["Age_log"])
    y = df["Age_log"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


@task(name="Preprocessing")
def preprocess_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Complete preprocessing pipeline."""
    # Load data
    df = load_data(data_path)

    # Apply feature engineering
    df_processed, scaler = feature_engineering(df)

    # Split data
    X_train, X_test, y_train, y_test = prepare_train_test_split(df_processed)

    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")

    return X_train, X_test, y_train, y_test, scaler
