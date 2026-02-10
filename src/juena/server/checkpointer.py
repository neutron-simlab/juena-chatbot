"""
SQLite checkpointer for LangGraph agents.

This module provides a singleton AsyncSqliteSaver instance that persists
conversation state across sessions using SQLite. The async saver is required
because the FastAPI server uses async methods (astream, aget_state).
"""
from contextlib import asynccontextmanager
from typing import Optional

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from juena.core.config import global_config
from juena.core.log import get_logger

logger = get_logger(__name__)

# Singleton checkpointer instance (set during lifespan)
_checkpointer: Optional[AsyncSqliteSaver] = None


def get_checkpointer() -> AsyncSqliteSaver:
    """
    Get the singleton AsyncSqliteSaver instance.

    Must be called after the app lifespan has started (checkpointer is
    initialized in FastAPI lifespan).

    Returns:
        AsyncSqliteSaver instance for LangGraph checkpoint persistence

    Raises:
        RuntimeError: If called before lifespan has initialized the checkpointer
    """
    if _checkpointer is None:
        raise RuntimeError(
            "Checkpointer not initialized. Ensure the FastAPI app lifespan has started."
        )
    return _checkpointer


@asynccontextmanager
async def checkpointer_lifespan():
    """
    Async context manager for initializing and closing the SQLite checkpointer.

    Use this in the FastAPI lifespan. The checkpointer remains open for the
    duration of the app and is closed on shutdown.

    Example:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with checkpointer_lifespan():
                yield
    """
    global _checkpointer

    db_path = global_config.CHECKPOINT_DB_PATH
    logger.info("Initializing AsyncSqliteSaver at: %s", db_path)

    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        _checkpointer = saver
        logger.info("AsyncSqliteSaver initialized successfully")
        try:
            yield saver
        finally:
            _checkpointer = None
            logger.info("AsyncSqliteSaver closed")
