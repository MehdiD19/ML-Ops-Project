# This module is the training flow: it reads the data, preprocesses it, trains a model and saves it.

import argparse
from pathlib import Path

from .training import train_model
from .utils import download_abalone_dataset


def main(trainset_path: str, model_type: str = "linear_regression") -> None:
    """Train a model using the data at the given path and save the model (pickle).

    Args:
        trainset_path: Path to the training CSV file
        model_type: Type of model to train ('linear_regression' or 'random_forest')
    """
    print(f"Starting training pipeline with {model_type} model...")
    print(f"Data path: {trainset_path}")

    # Download data if it doesn't exist
    if not Path(trainset_path).exists():
        print(f"Training data not found at: {trainset_path}")
        if trainset_path == "abalone.csv":
            print("Downloading abalone dataset...")
            download_abalone_dataset(trainset_path)
        else:
            raise FileNotFoundError(f"Training data not found at: {trainset_path}")

    # Train model with complete pipeline
    train_model(trainset_path, model_type)

    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the data at the given path.")
    parser.add_argument("trainset_path", type=str, help="Path to the training set CSV file")
    parser.add_argument(
        "--model_type",
        type=str,
        default="linear_regression",
        choices=["linear_regression", "random_forest"],
        help="Type of model to train (default: linear_regression)",
    )

    args = parser.parse_args()
    main(args.trainset_path, args.model_type)
