#!/usr/bin/env python3
"""Simple test script for the Abalone Age Prediction API.

Run this after starting the API to test if it's working correctly.
"""

import requests

# API base URL
BASE_URL = "http://localhost:8001"


def test_health_check() -> bool:
    """Test the health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_model_info() -> bool:
    """Test the model info endpoint."""
    print("\nTesting model info...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_prediction() -> bool:
    """Test a single prediction."""
    print("\nTesting single prediction...")

    # Sample abalone data
    sample_data = {
        "sex": "M",
        "length": 0.455,
        "diameter": 0.365,
        "height": 0.095,
        "whole_weight": 0.514,
        "shucked_weight": 0.2245,
        "viscera_weight": 0.101,
        "shell_weight": 0.15,
    }

    try:
        response = requests.post(f"{BASE_URL}/predict", json=sample_data, headers={"Content-Type": "application/json"})
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_batch_prediction() -> bool:
    """Test batch prediction."""
    print("\nTesting batch prediction...")

    # Sample batch data
    batch_data = {
        "instances": [
            {
                "sex": "M",
                "length": 0.455,
                "diameter": 0.365,
                "height": 0.095,
                "whole_weight": 0.514,
                "shucked_weight": 0.2245,
                "viscera_weight": 0.101,
                "shell_weight": 0.15,
            },
            {
                "sex": "F",
                "length": 0.35,
                "diameter": 0.265,
                "height": 0.09,
                "whole_weight": 0.2255,
                "shucked_weight": 0.0995,
                "viscera_weight": 0.0485,
                "shell_weight": 0.07,
            },
        ]
    }

    try:
        response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data, headers={"Content-Type": "application/json"})
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def main() -> None:
    """Run all tests."""
    print("=" * 50)
    print("Testing Abalone Age Prediction API")
    print("=" * 50)

    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_prediction),
        ("Batch Prediction", test_batch_prediction),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 30}")
        result = test_func()
        results.append((test_name, result))

    print(f"\n{'=' * 50}")
    print("Test Results Summary:")
    print(f"{'=' * 50}")

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")


if __name__ == "__main__":
    main()
