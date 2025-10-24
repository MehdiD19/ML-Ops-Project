# Dockerfile for Abalone Age Prediction API
FROM python:3.11-slim

# Prevent .pyc and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Ensure uv-created venv binaries are on PATH
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Set working directory
WORKDIR /app

# System deps (needed for compiling some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# Install uv (dependency manager)
RUN pip install --no-cache-dir uv

# Copy dependency files and install exactly what's locked
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy application code
COPY src/ ./src/
COPY bin/ ./bin/

# Make the run script executable
RUN chmod +x ./bin/run_services.sh

# Create directory for model artifacts (present also at runtime)
RUN mkdir -p ./src/web_service/local_objects

# Expose ports (8001 FastAPI API, 4201 Prefect)
EXPOSE 8001 4201

# Start services
CMD ["bash", "-c", "./bin/run_services.sh"]
