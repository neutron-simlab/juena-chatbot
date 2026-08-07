"""Lifecycle-managed Postgres Store for cross-thread agent files."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from langgraph.store.postgres.aio import AsyncPostgresStore

from juena.core.log import get_logger
from juena.server.database import get_pool

logger = get_logger(__name__)

_store: AsyncPostgresStore | None = None


def get_store() -> AsyncPostgresStore:
    if _store is None:
        raise RuntimeError("Postgres Store is not initialized")
    return _store


@asynccontextmanager
async def store_lifespan() -> AsyncGenerator[AsyncPostgresStore, None]:
    """Expose the Store backing Deep Agents' per-user AGENTS.md files.

    Shares the psycopg pool opened by ``database_lifespan``.
    """

    global _store

    logger.info("Initializing AsyncPostgresStore")
    store = AsyncPostgresStore(get_pool())
    await store.setup()
    _store = store
    try:
        yield store
    finally:
        _store = None
        logger.info("AsyncPostgresStore released")
