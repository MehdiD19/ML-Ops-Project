# This module allows model retraining through prefect deployment.
# import argparse
# from pathlib import Path

from training import train_model
from prefect import serve


if __name__ == "__main__":
    retrain_deployment = train_model.to_deployment(
        name="Retrain Deployment",
        version="1.0.0",
        tags=["Retraining"],
        interval=60,
        parameters={"data_path": "../../data/abalone.csv", "model_type": "linear_regression"},
    )
    serve(retrain_deployment)
