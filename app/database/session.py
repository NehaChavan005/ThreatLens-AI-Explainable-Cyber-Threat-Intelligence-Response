"""Async SQLAlchemy engine + session factory with graceful degradation.

The API keeps serving predictions even when the database is unreachable —
alerts are simply skipped and `/health` reports a degraded status.
"""

import logging
from typing import AsyncGenerator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..core.config import settings
from .models import Base

logger = logging.getLogger(__name__)

engine = None
session_maker: Optional[async_sessionmaker] = None
db_available = False


async def init_db() -> bool:
    """Create the engine, verify connectivity, and create tables."""
    global engine, session_maker, db_available

    try:
        engine = create_async_engine(
            settings.DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
        )
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

        session_maker = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
        )

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        db_available = True
        logger.info("Database initialized at %s", settings.DATABASE_URL)
        return True

    except Exception as exc:  # pragma: no cover - depends on infra
        logger.error("Database init failed: %s — running degraded", exc)
        db_available = False
        return False


async def close_db() -> None:
    global engine
    if engine is not None:
        await engine.dispose()
        logger.info("Database connections closed")


async def get_db() -> AsyncGenerator[Optional[AsyncSession], None]:
    """FastAPI dependency yielding an async session or None."""
    if not db_available or session_maker is None:
        yield None
        return

    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def is_db_available() -> bool:
    return db_available


async def health_check() -> dict:
    """Health report used by the /health endpoint."""
    if not db_available or engine is None:
        return {"database": "unavailable", "status": "degraded"}
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return {"database": "healthy", "status": "ok"}
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return {"database": "unhealthy", "status": "error", "error": str(exc)}