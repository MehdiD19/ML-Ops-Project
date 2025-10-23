# This module provides a CLI for making predictions using the trained model.

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from predicting import predict_age, predict_rings


def create_sample_dict(
    sex: str,
    length: float,
    diameter: float,
    height: float,
    whole_weight: float,
    shucked_weight: float,
    viscera_weight: float,
    shell_weight: float,
    rings: int = 10,  # Default value, not used for prediction but needed for preprocessing
) -> Dict[str, Any]:
    """Create a sample dictionary from individual features."""
    return {
        "Sex": sex,
        "Length": length,
        "Diameter": diameter,
        "Height": height,
        "Whole weight": whole_weight,
        "Shucked weight": shucked_weight,
        "Viscera weight": viscera_weight,
        "Shell weight": shell_weight,
        "Rings": rings,
    }


def predict_from_features(
    sex: str,
    length: float,
    diameter: float,
    height: float,
    whole_weight: float,
    shucked_weight: float,
    viscera_weight: float,
    shell_weight: float,
    model_type: str = "linear_regression",
    output_type: str = "age",
) -> None:
    """Make predictions from individual feature values."""
    # Create sample data
    sample_data = create_sample_dict(
        sex=sex,
        length=length,
        diameter=diameter,
        height=height,
        whole_weight=whole_weight,
        shucked_weight=shucked_weight,
        viscera_weight=viscera_weight,
        shell_weight=shell_weight,
    )

    # Determine model path
    model_path = f"src/web_service/local_objects/{model_type}_model.pkl"
    scaler_path = "src/web_service/local_objects/scaler.pkl"

    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print(f"Please train a {model_type} model first by running:")
        print(f"python -m src.modelling.main abalone.csv --model_type {model_type}")
        sys.exit(1)

    try:
        # Make prediction
        if output_type == "age":
            prediction = predict_age(sample_data, model_path, scaler_path)
            print("\n🔮 Prediction Results:")
            print(f"Predicted Age: {prediction:.2f} years")
        else:  # rings
            prediction = predict_rings(sample_data, model_path, scaler_path)
            print("\n🔮 Prediction Results:")
            print(f"Predicted Rings: {prediction}")

        # Display input features for reference
        print("\n📊 Input Features:")
        for key, value in sample_data.items():
            if key != "Rings":  # Don't show rings as it's not actually used
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)


def predict_from_json(json_input: str, model_type: str = "linear_regression", output_type: str = "age") -> None:
    """Make predictions from JSON input."""
    try:
        # Parse JSON input
        sample_data = json.loads(json_input)

        # Validate required fields
        required_fields = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
        ]

        missing_fields = [field for field in required_fields if field not in sample_data]
        if missing_fields:
            print(f"Error: Missing required fields: {missing_fields}")
            sys.exit(1)

        # Add rings if not present (needed for preprocessing)
        if "Rings" not in sample_data:
            sample_data["Rings"] = 10

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)

    # Determine model path
    model_path = f"src/web_service/local_objects/{model_type}_model.pkl"
    scaler_path = "src/web_service/local_objects/scaler.pkl"

    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print(f"Please train a {model_type} model first by running:")
        print(f"python -m src.modelling.main abalone.csv --model_type {model_type}")
        sys.exit(1)

    try:
        # Make prediction
        if output_type == "age":
            prediction = predict_age(sample_data, model_path, scaler_path)
            print("\n🔮 Prediction Results:")
            print(f"Predicted Age: {prediction:.2f} years")
        else:  # rings
            prediction = predict_rings(sample_data, model_path, scaler_path)
            print("\n🔮 Prediction Results:")
            print(f"Predicted Rings: {prediction}")

        # Display input features for reference
        print("\n📊 Input Features:")
        for key, value in sample_data.items():
            if key != "Rings":  # Don't show rings as it's not actually used
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error making prediction: {e}")
        sys.exit(1)


def main() -> None:
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Make age/ring predictions for abalone using trained model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Individual features
  python -m src.modelling.predict --sex M --length 0.455 --diameter 0.365 --height 0.095 \\
    --whole_weight 0.514 --shucked_weight 0.2245 --viscera_weight 0.101 --shell_weight 0.15

  # JSON input
  python -m src.modelling.predict --json \\
    '{"Sex": "M", "Length": 0.455, "Diameter": 0.365, "Height": 0.095, "Whole weight": 0.514, \\
     "Shucked weight": 0.2245, "Viscera weight": 0.101, "Shell weight": 0.15}'

  # Predict rings instead of age
  python -m src.modelling.predict --sex F --length 0.350 --diameter 0.265 --height 0.090 \\
    --whole_weight 0.2255 --shucked_weight 0.0995 --viscera_weight 0.0485 --shell_weight 0.070 --output rings
        """,
    )

    # Input method (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)

    # JSON input option
    input_group.add_argument("--json", type=str, help="JSON string with abalone features")

    # Individual feature options
    input_group.add_argument("--sex", type=str, choices=["M", "F", "I"], help="Sex of abalone (M=Male, F=Female, I=Infant)")

    # Additional individual feature arguments (only used if --sex is provided)
    parser.add_argument("--length", type=float, help="Longest shell measurement (mm)")
    parser.add_argument("--diameter", type=float, help="Perpendicular to length (mm)")
    parser.add_argument("--height", type=float, help="Height with meat in shell (mm)")
    parser.add_argument("--whole_weight", type=float, help="Whole abalone weight (grams)")
    parser.add_argument("--shucked_weight", type=float, help="Weight of meat (grams)")
    parser.add_argument("--viscera_weight", type=float, help="Gut weight after bleeding (grams)")
    parser.add_argument("--shell_weight", type=float, help="Weight after being dried (grams)")

    # Model and output options
    parser.add_argument(
        "--model_type",
        type=str,
        default="linear_regression",
        choices=["linear_regression", "random_forest"],
        help="Type of model to use for prediction (default: linear_regression)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="age",
        choices=["age", "rings"],
        help="Output type: age in years or ring count (default: age)",
    )

    args = parser.parse_args()

    if args.json:
        # JSON input mode
        predict_from_json(args.json, args.model_type, args.output)
    else:
        # Individual features mode
        if args.sex:
            # Check that all required individual arguments are provided
            required_args = ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]
            missing_args = [arg for arg in required_args if getattr(args, arg) is None]

            if missing_args:
                print("Error: When using --sex, all feature arguments are required.")
                print(f"Missing: {missing_args}")
                parser.print_help()
                sys.exit(1)

            predict_from_features(
                sex=args.sex,
                length=args.length,
                diameter=args.diameter,
                height=args.height,
                whole_weight=args.whole_weight,
                shucked_weight=args.shucked_weight,
                viscera_weight=args.viscera_weight,
                shell_weight=args.shell_weight,
                model_type=args.model_type,
                output_type=args.output,
            )


if __name__ == "__main__":
    main()
