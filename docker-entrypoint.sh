#!/bin/sh
set -e

# Function to handle shutdown
cleanup() {
    echo "Shutting down services..."
    kill $STREAMLIT_PID 2>/dev/null || true
    exit 0
}

# Trap signals for graceful shutdown (EXIT is not POSIX across all shells)
trap cleanup TERM INT

# Create required directories
mkdir -p /data/logs
echo "✅ Log directory: /data/logs"

mkdir -p /data/saml

if [ "${ENVIRONMENT:-development}" != "production" ] && \
   { [ ! -f "${SAML_SP_CERT_FILE:-/data/saml/sp.crt}" ] || [ ! -f "${SAML_SP_KEY_FILE:-/data/saml/sp.key}" ]; }; then
    echo "Generating a development-only SAML signing certificate..."
    openssl req -x509 -newkey rsa:3072 -nodes -days 30 \
        -subj "/CN=juena-development-sp" \
        -keyout "${SAML_SP_KEY_FILE:-/data/saml/sp.key}" \
        -out "${SAML_SP_CERT_FILE:-/data/saml/sp.crt}"
fi

# Activate virtual environment if it exists (for uv)
if [ -f "/app/.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    . /app/.venv/bin/activate
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

# Bootstrap repositories before starting any services.
echo "Bootstrapping repositories and vector indices..."
python -m juena.indexing.bootstrap
export JUENA_BOOTSTRAP_DONE=1
echo "✅ Repository bootstrap completed"

# Start Streamlit in the background
echo "Starting Streamlit UI on port 9501..."
streamlit run app/streamlit_app.py \
    --server.port=9501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
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
echo "Starting FastAPI server on port 8080..."
python main.py
