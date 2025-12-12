# Dockerfile for Rossmann Forecasting API
FROM python:3.10-slim

LABEL maintainer="Bradley Boehmke <bradleyboehmke@gmail.com>"
LABEL description="Production-ready ML API for Rossmann sales forecasting"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for layer caching)
COPY pyproject.toml README.md ./

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install --no-cache -e .

# Copy application code
COPY src/ src/
COPY deployment/ deployment/
COPY config/ config/

# Create necessary directories
RUN mkdir -p models mlruns logs

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI with uvicorn
CMD ["uvicorn", "deployment.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
