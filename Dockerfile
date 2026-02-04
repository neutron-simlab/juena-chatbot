# Dockerfile for juena-chatbot
# Multi-stage build for efficient image size

# -----------------------------------------------------------------------------
# Stage 1: Base Python image with dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    bash \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml README.md ./

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY main.py ./

# Copy entrypoint script
COPY docker-entrypoint.sh ./

# Install dependencies using uv
RUN uv pip install --system -e .

# Make entrypoint script executable
RUN chmod +x docker-entrypoint.sh

# Create directories for logs and data
RUN mkdir -p /data/logs

# Expose ports
# 8000: FastAPI server
# 8501: Streamlit UI
EXPOSE 8000 8501

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["python", "main.py"]
