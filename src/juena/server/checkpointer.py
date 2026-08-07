"""Lifecycle-managed Postgres checkpointer for thread-scoped agent state."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from juena.core.log import get_logger
from juena.server.database import get_pool

logger = get_logger(__name__)

_checkpointer: AsyncPostgresSaver | None = None


def get_checkpointer() -> AsyncPostgresSaver:
    if _checkpointer is None:
        raise RuntimeError(
            "Checkpointer not initialized. Ensure the FastAPI lifespan has started."
        )
    return _checkpointer


@asynccontextmanager
async def checkpointer_lifespan() -> AsyncGenerator[AsyncPostgresSaver, None]:
    """Migrate and expose the checkpointer over the shared psycopg pool.

    Must be entered inside ``database_lifespan``, which owns the pool.
    """

    global _checkpointer

    logger.info("Initializing AsyncPostgresSaver")
    saver = AsyncPostgresSaver(get_pool())
    await saver.setup()
    _checkpointer = saver
    try:
        yield saver
    finally:
        _checkpointer = None
        logger.info("AsyncPostgresSaver released")
