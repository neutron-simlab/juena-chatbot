# Dockerfile for juena-chatbot

FROM node:24-bookworm-slim AS filesystem-build
WORKDIR /core/frontend/filesystem
COPY juena-core/frontend/filesystem/package.json juena-core/frontend/filesystem/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY juena-core/frontend/filesystem/ ./
RUN npm run build

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    bash \
    git \
    openssl \
    xmlsec1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

# The build context is the PARENT of this repository, not this repository, so
# every source path below is prefixed with a repository name. It has to be:
# juena-core is a sibling path dependency, and a build context cannot reach
# outside itself. See docker-compose.yml's `context: ..`.
#
# Because the context root is the parent, Docker reads `Dockerfile.dockerignore`
# beside this file and ignores the repository's own `.dockerignore` entirely.

# Copy lockfile and project metadata first for better layer caching.
COPY juena-chatbot/pyproject.toml juena-chatbot/uv.lock juena-chatbot/README.md ./

# juena-core, the sibling source this application now consumes. `[tool.uv.sources]`
# declares it as `{ path = "../juena-core" }`, and uv resolves that relative to the
# project directory -- which is /app here, so core must land at /juena-core for the
# sync below to find it. uv builds a wheel from this tree; nothing imports it from
# here at runtime.
COPY juena-core/ /juena-core/
COPY --from=filesystem-build /core/src/juena_core/ui/filesystem_dist/ /juena-core/src/juena_core/ui/filesystem_dist/

# Install locked third-party dependencies into /app/.venv.
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code and config
COPY juena-chatbot/src/ ./src/
COPY juena-chatbot/app/ ./app/
COPY juena-chatbot/config/ ./config/
# The Streamlit theme. Resolved relative to the working directory, which is
# where the entrypoint runs `streamlit run`, so it must land at /app/.streamlit.
COPY juena-chatbot/.streamlit/ ./.streamlit/
COPY juena-chatbot/main.py ./

# Copy entrypoint script
COPY juena-chatbot/docker-entrypoint.sh ./

# If the repo was checked out on Windows, make sure we don't have CRLF.
RUN sed -i 's/\r$//' docker-entrypoint.sh

# Install the project itself into the same locked environment.
RUN uv sync --frozen --no-dev

# Make entrypoint script executable
RUN chmod +x docker-entrypoint.sh

# Create directories for logs, data, vector index, and manifests
RUN mkdir -p /data/logs /data/repos /data/vector_index /data/manifests /data/saml \
    /data/artifacts /data/audit

# Expose ports
# 8080: FastAPI server
# 9501: Streamlit UI
EXPOSE 8080 9501

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["python", "main.py"]
    
