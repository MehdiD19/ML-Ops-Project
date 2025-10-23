import pickle
import urllib.request
from pathlib import Path
from typing import Any


def pickle_object(obj: Any, filepath: Path) -> None:
    """Save any object to a pickle file.

    Args:
        obj: The object to pickle (model, scaler, etc.)
        filepath: Path where to save the pickle file
    """
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("wb") as f:
        pickle.dump(obj, f)

    print(f"Object saved to {filepath}")


def load_pickle_object(filepath: Path) -> Any:
    """Load an object from a pickle file.

    Args:
        filepath: Path to the pickle file

    Returns:
        The loaded object
    """
    with filepath.open("rb") as f:
        obj = pickle.load(f)

    print(f"Object loaded from {filepath}")
    return obj


def download_abalone_dataset(output_path: str = "abalone.csv") -> None:
    """Download the abalone dataset from UCI ML Repository.

    Args:
        output_path: Path where to save the dataset
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

    print(f"Downloading abalone dataset from {url}...")
    urllib.request.urlretrieve(url, output_path)

    # Add column names
    import pandas as pd

    column_names = [
        "Sex",
        "Length",
        "Diameter",
        "Height",
        "Whole weight",
        "Shucked weight",
        "Viscera weight",
        "Shell weight",
        "Rings",
    ]

    df = pd.read_csv(output_path, names=column_names)
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path} with {len(df)} samples")
