# Make sure to check bin/run_services.sh, which can be used here

# Do not forget to expose the right ports! (Check the PR_4.md)
# Dockerfile for Abalone Age Prediction API
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy application code
COPY src/ ./src/
COPY bin/ ./bin/

# Make the run script executable and verify it exists
RUN chmod +x ./bin/run_services.sh && \
    ls -la ./bin/ && \
    cat ./bin/run_services.sh

# Create directory for model files
RUN mkdir -p ./src/web_service/local_objects

# Expose ports as specified in PR_4.md
# Port 8001 for FastAPI, Port 4201 for Prefect
EXPOSE 8001 4201

# Use the run script to start services
CMD ["/bin/bash", "-c", "ls -la ./bin/ && ./bin/run_services.sh"]