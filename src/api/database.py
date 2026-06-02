"""
Async database layer using SQLAlchemy 2.x + aiosqlite.
Stores all ingested events with full schema support.
Designed to swap SQLite → PostgreSQL via env var.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Index, Integer,
    String, Text, create_engine, event, text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ──────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────

# Neon / Render / Vercel give a postgres:// URL — convert to async driver
_raw_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL", "sqlite+aiosqlite:///./data/store_intelligence.db")

if _raw_url.startswith("postgres://"):
    # Neon uses postgres://, SQLAlchemy needs postgresql+asyncpg://
    DATABASE_URL = _raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
elif _raw_url.startswith("postgresql://") and "+" not in _raw_url:
    DATABASE_URL = _raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    DATABASE_URL = _raw_url

_is_sqlite = "sqlite" in DATABASE_URL

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


# ──────────────────────────────────────────────
# Base & ORM Models
# ──────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class EventRecord(Base):
    """Persistent event store — single source of truth for all analytics."""
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    store_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(32), nullable=False)
    visitor_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    zone_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    zone_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    zone_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    dwell_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_staff: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_face_hidden: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    gender: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    age_bucket: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    group_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    group_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Queue fields
    wait_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    abandoned: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    queue_position_at_join: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    queue_join_ts: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    queue_served_ts: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    queue_exit_ts: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Spatial
    zone_hotspot_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    zone_hotspot_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Ingestion bookkeeping
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_events_store_ts", "store_id", "timestamp"),
        Index("ix_events_store_type", "store_id", "event_type"),
    )


class DailyMetricsSnapshot(Base):
    """Pre-aggregated daily metrics for trend / baseline comparisons."""
    __tablename__ = "daily_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    date: Mapped[str] = mapped_column(String(10), nullable=False)            # YYYY-MM-DD
    unique_visitors: Mapped[int] = mapped_column(Integer, default=0)
    conversion_rate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_dwell_ms: Mapped[float] = mapped_column(Float, default=0.0)
    avg_wait_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    abandonment_rate: Mapped[float] = mapped_column(Float, default=0.0)
    revenue_inr: Mapped[float] = mapped_column(Float, default=0.0)

    __table_args__ = (
        Index("ix_daily_store_date", "store_id", "date", unique=True),
    )


# ──────────────────────────────────────────────
# Database Init
# ──────────────────────────────────────────────

async def init_db() -> None:
    """Create all tables and enable WAL mode for SQLite."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Enable WAL for concurrent reads on SQLite
        if _is_sqlite:
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA synchronous=NORMAL"))
            await conn.execute(text("PRAGMA cache_size=10000"))


async def get_session() -> AsyncSession:
    """Dependency-injected async session."""
    async with AsyncSessionLocal() as session:
        yield session
