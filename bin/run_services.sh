#!/bin/bash
set -e

# Start Prefect server (background)
echo "Starting Prefect server on port 4201..."
prefect server start --host 0.0.0.0 --port 4201 &

# Give Prefect a few seconds to start
sleep 5

# Start FastAPI with Uvicorn (foreground)
echo "Starting FastAPI app on port 8001..."
exec uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
