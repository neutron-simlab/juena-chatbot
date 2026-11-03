# Build core's frontend as part of its Git dependency wheel. Node stays here.
FROM node:24-bookworm-slim AS node
FROM python:3.11-slim-bookworm AS build
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.7.15 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra services --no-install-project
COPY src/ ./src/
RUN uv sync --frozen --no-dev --extra services --no-editable

FROM python:3.11-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl bash git openssl xmlsec1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
COPY --from=build /app/.venv /app/.venv
COPY app/ ./app/
COPY config/ ./config/
COPY .streamlit/ ./.streamlit/
COPY main.py docker-entrypoint.sh ./
RUN sed -i 's/\r$//' docker-entrypoint.sh && chmod +x docker-entrypoint.sh \
    && mkdir -p /data/logs /data/repos /data/saml /data/artifacts /data/audit
EXPOSE 8080 9501
ENTRYPOINT ["./docker-entrypoint.sh"]
