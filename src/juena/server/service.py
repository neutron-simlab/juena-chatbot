"""
FastAPI service for Chatbot Template.

This module sets up the FastAPI application and registers all endpoint routers.
"""
import asyncio
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core._api import LangChainBetaWarning

from juena.core.config import global_config
from juena.core.log import get_logger
from juena.schema.server import HealthStatus
from juena.server.agent.registry import shutdown_agents
from juena.server.api.endpoints import router as api_router
from juena.server.auth.endpoints import router as auth_router
from juena.server.chat.endpoints import router as chat_router
from juena.server.database.checkpointer import checkpointer_lifespan
from juena.server.database import database_lifespan
from juena.server.auth.saml import saml_lifespan
from juena.server.database.store import store_lifespan
from juena.sandbox.broker_client import close_broker_client, get_broker_client

# Imported for its side effect: the module self-registers the agent factory.
# Importing *this* module has to be enough, because the process serving the API
# is not always `main.py` -- production runs `uvicorn juena.server.service:app`,
# and the deployment plan splits the API and the UI into separate containers.
# Registering only from `main.py` leaves those processes with an empty registry,
# so every invocation would 404.
import juena.agents.juena_agent  # noqa: F401  register the JüNA supervisor

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = get_logger(__name__)
logger.info("Service logging initialized")


@asynccontextmanager
async def _optional_saml_lifespan() -> AsyncGenerator[None, None]:
    """Start the SAML SP unless the development bypass replaces it.

    Skipping it is what lets a local checkout run without SP certificates,
    `xmlsec1`, or network access to the FZJ IdP.
    """
    if global_config.AUTH_DEV_BYPASS:
        logger.warning(
            "AUTH_DEV_BYPASS is enabled: SAML is not configured and /auth/dev-login "
            "issues sessions without verifying identity. Development only."
        )
        yield
        return
    async with saml_lifespan():
        yield


def _warm_vector_index() -> None:
    """Load the repository collections before the first search needs them.

    Chroma loads a collection's HNSW segment lazily, which costs several
    seconds across all repositories. Chroma shares those segments per index
    path within the process, so warming here also serves the separate
    ``RepoVectorIndex`` the agent builds later.

    Never raises: an unsearchable index must not stop the server from starting.
    """
    try:
        from juena.indexing.repo_manager import RepoManager
        from juena.indexing.vector_index import RepoVectorIndex

        RepoVectorIndex(RepoManager()).warm()
    except Exception:
        logger.warning("Vector index warm-up failed; searches stay cold", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Initialize Postgres, the LangGraph checkpointer, and SAML metadata.
    """
    if global_config.SANDBOX_ENABLED:
        await asyncio.to_thread(get_broker_client().assert_healthy)
    # Deliberately not awaited: warming takes seconds and the service is
    # perfectly usable while it runs, so it must not delay readiness.
    warm_task = asyncio.create_task(asyncio.to_thread(_warm_vector_index))
    async with database_lifespan():
        async with checkpointer_lifespan():
            async with store_lifespan():
                async with _optional_saml_lifespan():
                    try:
                        yield
                    finally:
                        warm_task.cancel()
                        # Agents hold network clients (Context7 MCP) for the
                        # process lifetime; close them before the pools go.
                        await shutdown_agents()
                        close_broker_client()


app = FastAPI(
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# Register routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(api_router)


@app.get("/health")
async def health_check() -> HealthStatus:
    """Health check endpoint."""
    return HealthStatus(
        status="ok",
        version="0.1.0",
        details={"service": "juena", "uptime": "running"}
    )
