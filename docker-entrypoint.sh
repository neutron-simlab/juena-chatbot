#!/bin/bash
set -e

# Function to handle shutdown
cleanup() {
    echo "Shutting down services..."
    kill $STREAMLIT_PID 2>/dev/null || true
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGTERM SIGINT EXIT

# Create required directories
mkdir -p /data/logs
echo "✅ Log directory: /data/logs"

# Activate virtual environment if it exists (for uv)
if [ -f "/app/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source /app/.venv/bin/activate
fi

# Load environment file if specified
if [ -f "/etc/juena/.env" ]; then
    export JUENA_ENV_PATH=/etc/juena/.env
    echo "✅ Using production environment file: /etc/juena/.env"
elif [ -f "/app/.env" ]; then
    export JUENA_ENV_PATH=/app/.env
    echo "✅ Using local environment file: /app/.env"
else
    echo "ℹ️  No .env file found, using environment variables"
fi

# Start Streamlit in the background
echo "Starting Streamlit UI on port 8501..."
streamlit run app/streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    &
STREAMLIT_PID=$!

# Wait a moment for Streamlit to start
sleep 2

# Check if Streamlit is running
if kill -0 $STREAMLIT_PID 2>/dev/null; then
    echo "✅ Streamlit started successfully (PID: $STREAMLIT_PID)"
else
    echo "⚠️  Warning: Streamlit failed to start"
fi

# Start FastAPI in the foreground (so we see logs and it's the main process)
echo "Starting FastAPI server on port 8000..."
python main.py
