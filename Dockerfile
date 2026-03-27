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
    
    # Copy source code and config
    COPY src/ ./src/
    COPY app/ ./app/
    COPY config/ ./config/
    COPY main.py ./
    
    # Copy entrypoint script
    COPY docker-entrypoint.sh ./
    
    # If the repo was checked out on Windows, make sure we don't have CRLF.
    RUN sed -i 's/\r$//' docker-entrypoint.sh
    
    # Install dependencies using uv
    RUN uv pip install --system -e .
    
    # Make entrypoint script executable
    RUN chmod +x docker-entrypoint.sh
    
    # Create directories for logs, data, vector index, sparse index, and manifests
    RUN mkdir -p /data/logs /data/repos /data/vector_index /data/sparse_index /data/manifests
    
    # Expose ports
    # 8080: FastAPI server
    # 9501: Streamlit UI
    EXPOSE 8080 9501
    
    # Set entrypoint
    ENTRYPOINT ["./docker-entrypoint.sh"]
    
    # Default command (can be overridden)
    CMD ["python", "main.py"]
    