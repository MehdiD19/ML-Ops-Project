# Utils file. TODO: add a `load_object` function to load pickle objects
# Utility functions for the web service

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_object(file_path: Path) -> Optional[Any]:
    """Load a pickle object from file."""
    try:
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        with file_path.open("rb") as f:
            obj = pickle.load(f)

        logger.info(f"Successfully loaded object from {file_path}")
        return obj

    except Exception as e:
        logger.error(f"Error loading object from {file_path}: {str(e)}")
        return None


def save_object(obj: Any, file_path: Path) -> bool:
    """Save an object to a pickle file."""
    try:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open("wb") as f:
            pickle.dump(obj, f)

        logger.info(f"Successfully saved object to {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving object to {file_path}: {str(e)}")
        return False


def get_model_path() -> Path:
    """Get the path to the model directory."""
    return Path("src/web_service/local_objects")


def ensure_model_directory() -> None:
    """Ensure the model directory exists."""
    model_dir = get_model_path()
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model directory ensured at: {model_dir}")


def validate_model_files() -> Dict[str, bool]:
    """Check if required model files exist."""
    model_dir = get_model_path()

    files_status = {
        "model.pkl": (model_dir / "model.pkl").exists(),
        "encoder.pkl": (model_dir / "encoder.pkl").exists(),
    }

    logger.info(f"Model files status: {files_status}")
    return files_status


def get_environment_info() -> Dict[str, Any]:
    """Get information about the current environment."""
    return {
        "python_version": os.sys.version,
        "working_directory": str(Path.cwd()),
        "model_directory": str(get_model_path()),
        "model_files": validate_model_files(),
    }


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )