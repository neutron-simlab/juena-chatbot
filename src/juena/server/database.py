"""Async Postgres engine and request-session lifecycle."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from juena.core.config import global_config
from juena.core.log import get_logger
from juena.server.models import Base

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_pool: AsyncConnectionPool | None = None

_DSN_PREFIXES = ("postgresql+psycopg://", "postgresql://", "postgres://")


def _strip_dsn_scheme(database_url: str) -> str:
    for prefix in _DSN_PREFIXES:
        if database_url.startswith(prefix):
            return database_url[len(prefix):]
    raise ValueError("DATABASE_URL must use postgres:// or postgresql://")


def psycopg_database_url(database_url: str) -> str:
    """Return the DSN psycopg accepts, for the checkpointer and the Store."""

    return f"postgresql://{_strip_dsn_scheme(database_url)}"


def sqlalchemy_database_url(database_url: str) -> str:
    """Convert a standard Postgres DSN to SQLAlchemy's async psycopg DSN."""

    return f"postgresql+psycopg://{_strip_dsn_scheme(database_url)}"


def get_pool() -> AsyncConnectionPool:
    """Return the shared psycopg pool used by the checkpointer and the Store."""

    if _pool is None:
        raise RuntimeError("Database not initialized. Ensure the FastAPI lifespan has started.")
    return _pool


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Ensure the FastAPI lifespan has started.")
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that supplies one transactional database session."""

    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def database_lifespan() -> AsyncGenerator[AsyncEngine, None]:
    """Open the application database pools and verify connectivity.

    Two pools, not three: SQLAlchemy for the application tables, and one
    psycopg pool shared by the LangGraph checkpointer and Store.  Both are
    derived from the same normalized ``DATABASE_URL``.
    """

    global _engine, _session_factory, _pool

    database_url = global_config.DATABASE_URL
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for Postgres persistence.")

    engine = create_async_engine(
        sqlalchemy_database_url(database_url),
        pool_pre_ping=True,
    )
    # LangGraph's Postgres backends require these connection settings.
    pool = AsyncConnectionPool(
        conninfo=psycopg_database_url(database_url),
        min_size=1,
        max_size=global_config.DATABASE_POOL_MAX_SIZE,
        open=False,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        },
    )
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
            await connection.run_sync(Base.metadata.create_all)
            await connection.commit()
        await pool.open(wait=True)
        _engine = engine
        _pool = pool
        _session_factory = async_sessionmaker(engine, expire_on_commit=False)
        logger.info("Postgres application database initialized")
        yield engine
    finally:
        _session_factory = None
        _engine = None
        _pool = None
        await pool.close()
        await engine.dispose()
        logger.info("Postgres application database closed")
