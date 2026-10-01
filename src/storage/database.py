"""
Async database engine and session management.

Uses asyncpg driver for high-performance async PostgreSQL access.
Provides a singleton engine and session factory with proper
connection pooling configuration.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.config import get_settings

logger = structlog.get_logger(__name__)

# Module-level singletons (initialized lazily)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """
    Get or create the async SQLAlchemy engine (singleton).

    Pool configuration:
    - pool_size=10: Maintain 10 persistent connections
    - max_overflow=20: Allow up to 20 additional connections under load
    - pool_pre_ping=True: Verify connections before use (handle stale connections)
    - pool_recycle=3600: Recycle connections every hour
    """
    global _engine

    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.log_level == "DEBUG",
        )
        logger.info("database_engine_created", url=settings.database_url.split("@")[-1])

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory (singleton)."""
    global _session_factory

    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,  # Prevent lazy-load issues in async context
        )

    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager that provides an async database session.

    Usage:
        async with get_session() as session:
            result = await session.execute(select(Model))

    The session auto-commits on success and rolls back on exception.
    """
    factory = get_session_factory()
    session = factory()

    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def close_engine() -> None:
    """Dispose of the engine and close all connections. Call on shutdown."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        logger.info("database_engine_closed")
        _engine = None
        _session_factory = None
