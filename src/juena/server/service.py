"""
FastAPI service for Chatbot Template.

This module sets up the FastAPI application and registers all endpoint routers.
"""
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_core._api import LangChainBetaWarning

from juena.core.log import get_logger
from juena.schema.server import HealthStatus
from juena.server.api_endpoints import router as api_router

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = get_logger(__name__)
logger.info("Service logging initialized")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Simple lifespan for in-memory only operation.
    """
    # No database/store initialization needed for in-memory operation
    yield


app = FastAPI(lifespan=lifespan)

# Register routers
app.include_router(api_router)


@app.get("/health")
async def health_check() -> HealthStatus:
    """Health check endpoint."""
    return HealthStatus(
        status="ok",
        version="0.1.0",
        details={"service": "juena", "uptime": "running"}
    )
