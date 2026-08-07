# JüNA development shortcuts.
#
# These wrap the commands already documented in README.md, HOW-TO-ADD-REPO.md
# and SAML-IFFLOGIN.md -- nothing here does anything those files do not.
# Production deployment deliberately has only a config check and an explicit
# `prod-up`, so a stray `make` in the wrong directory cannot deploy.

# Development always loads the base file plus the dev overlay. The overlay is
# opt-in by design: it supplies localhost SAML URLs and an insecure cookie, and
# must never be merged into a production run.
COMPOSE_DEV := docker compose -f docker-compose.yml -f docker-compose.dev.yml
COMPOSE_PROD := docker compose -f docker-compose.yml

# Production reads its settings from a file outside the repo.
PROD_ENV_FILE ?= /etc/juena/juena.env

# Host-side DSN for the integration tests. Matches the development defaults in
# env.example; override on the command line if you changed them.
POSTGRES_USER ?= juena
POSTGRES_PASSWORD ?= juena-development-only
POSTGRES_DB ?= juena
POSTGRES_PORT ?= 5432
TEST_DATABASE_URL ?= postgresql://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:$(POSTGRES_PORT)/$(POSTGRES_DB)

API_PORT ?= 8080
UI_PORT ?= 9501

.DEFAULT_GOAL := help

.PHONY: help env run ui bootstrap \
        postgres-up postgres-down postgres-reset \
        up down build logs ps \
        test test-integration test-all \
        sp-metadata health \
        prod-config prod-up clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

env: ## Create .env from env.example if it does not exist
	@if [ -f .env ]; then \
		echo ".env already exists; leaving it alone"; \
	else \
		cp env.example .env; \
		echo "Created .env from env.example -- add at least one LLM provider key"; \
	fi

# Compose auto-loads .env from the repository root, and the base file now
# *requires* POSTGRES_PASSWORD. Failing here with a readable message beats
# failing inside compose interpolation.
.env-required:
	@test -f .env || { \
		echo "error: .env is missing. Run 'make env' first."; \
		exit 1; \
	}

# ---------------------------------------------------------------------------
# Running natively (uv)
# ---------------------------------------------------------------------------

run: ## Run FastAPI natively on :8080 (bootstraps repositories first)
	uv run main.py

ui: ## Run the Streamlit UI natively on :9501
	uv run streamlit run app/streamlit_app.py \
		--server.port=$(UI_PORT) \
		--server.address=0.0.0.0

bootstrap: ## Clone/refresh repositories and rebuild stale vector indices
	uv run python -m juena.indexing.bootstrap

# ---------------------------------------------------------------------------
# Development Postgres
# ---------------------------------------------------------------------------

postgres-up: .env-required ## Start only Postgres, published on :5432
	$(COMPOSE_DEV) up -d postgres

postgres-down: ## Stop Postgres, keeping its data volume
	$(COMPOSE_DEV) stop postgres

postgres-reset: ## DESTROYS development Postgres data. Requires CONFIRM=yes
	@test "$(CONFIRM)" = "yes" || { \
		echo "refusing: this deletes the juena-chatbot-postgres volume and every"; \
		echo "development user, chat, checkpoint and memory in it."; \
		echo "Re-run with: make postgres-reset CONFIRM=yes"; \
		exit 1; \
	}
	$(COMPOSE_DEV) down
	docker volume rm juena-chatbot-postgres

# ---------------------------------------------------------------------------
# Development stack
# ---------------------------------------------------------------------------

up: .env-required ## Start the full development stack (build if needed)
	$(COMPOSE_DEV) up --build -d
	@echo "UI      http://localhost:$(UI_PORT)"
	@echo "API     http://localhost:$(API_PORT)"

down: ## Stop the development stack, keeping volumes
	$(COMPOSE_DEV) down

build: .env-required ## Rebuild the image without starting anything
	$(COMPOSE_DEV) build

logs: ## Follow application logs (bootstrap progress appears here)
	$(COMPOSE_DEV) logs -f juena-chatbot

ps: ## Show development stack status
	$(COMPOSE_DEV) ps

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test: ## Run the unit suite
	uv run pytest -q --ignore=tests/test_postgres_integration.py

test-integration: postgres-up ## Run the Postgres-backed tests against dev Postgres
	JUENA_TEST_DATABASE_URL=$(TEST_DATABASE_URL) \
		uv run pytest tests/test_postgres_integration.py -q

test-all: ## Run the unit suite and the Postgres-backed tests
	$(MAKE) test
	$(MAKE) test-integration

# ---------------------------------------------------------------------------
# SAML / health
# ---------------------------------------------------------------------------

health: ## Check the API health endpoint
	curl -fsS http://localhost:$(API_PORT)/health && echo

sp-metadata: ## Print SP metadata (the document to register with the FZJ IdP)
	@# Returns 500 while AUTH_DEV_BYPASS is on: the bypass skips saml_lifespan,
	@# so no SP is initialised. Unset it to produce real metadata.
	@# Redirect to a file to hand it over: make sp-metadata > sp-metadata.xml
	curl -fsS http://localhost:$(API_PORT)/auth/metadata

# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

prod-config: ## Validate the production compose config against PROD_ENV_FILE
	$(COMPOSE_PROD) --env-file $(PROD_ENV_FILE) config

prod-up: ## Start the production stack (base compose file only)
	$(COMPOSE_PROD) --env-file $(PROD_ENV_FILE) up -d

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean: ## Remove Python caches and test artefacts
	find . -path ./.venv -prune -o -type d -name __pycache__ -print0 \
		| xargs -0 rm -rf
	rm -rf .pytest_cache
