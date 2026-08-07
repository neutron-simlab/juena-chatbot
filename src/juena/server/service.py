"""
FastAPI service for Chatbot Template.

This module sets up the FastAPI application and registers all endpoint routers.
"""
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core._api import LangChainBetaWarning

from juena.core.config import global_config
from juena.core.log import get_logger
from juena.schema.server import HealthStatus
from juena.server.agent_registry import shutdown_agents
from juena.server.api_endpoints import router as api_router
from juena.server.auth_endpoints import router as auth_router
from juena.server.chat_endpoints import router as chat_router
from juena.server.checkpointer import checkpointer_lifespan
from juena.server.database import database_lifespan
from juena.server.saml_service import saml_lifespan
from juena.server.store import store_lifespan

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Initialize Postgres, the LangGraph checkpointer, and SAML metadata.
    """
    async with database_lifespan():
        async with checkpointer_lifespan():
            async with store_lifespan():
                async with _optional_saml_lifespan():
                    try:
                        yield
                    finally:
                        # Agents hold network clients (Context7 MCP) for the
                        # process lifetime; close them before the pools go.
                        await shutdown_agents()


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
