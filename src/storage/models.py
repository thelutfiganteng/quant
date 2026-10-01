"""
SQLAlchemy 2.0 ORM models for PostgreSQL + TimescaleDB.

Tables:
- economic_releases: Macro data releases with actual/consensus/surprise
- market_prices: OHLCV time series for equities/ETFs
- nowcast_estimates: Model predictions before official releases
- calendar_events: Upcoming economic calendar

The three time-series tables (releases, prices, estimates) are converted
to TimescaleDB hypertables via Alembic migration for automatic partitioning
and optimized time-range queries.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    JSON,
    Numeric,
    String,
    Text,
    event,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# ---------------------------------------------------------------------------
# Economic Releases
# ---------------------------------------------------------------------------


class EconomicReleaseModel(Base):
    """
    Stores actual economic data releases with consensus and surprise values.

    TimescaleDB hypertable on release_date for fast time-range queries.
    Unique constraint on (indicator, release_date, source) prevents duplicates
    when scheduler re-runs ingestion (idempotency requirement).
    """

    __tablename__ = "economic_releases"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    indicator: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    release_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    actual: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    consensus: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    previous: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    surprise: Mapped[float | None] = mapped_column(Numeric(12, 6), nullable=True)
    surprise_zscore: Mapped[float | None] = mapped_column(Numeric(8, 4), nullable=True)
    revision: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_releases_indicator_date", "indicator", "release_date"),
        Index("ix_releases_date", "release_date"),
        # Unique constraint for idempotent upserts
        Index(
            "uq_releases_indicator_date_source",
            "indicator",
            "release_date",
            "source",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<EconomicRelease(indicator={self.indicator}, "
            f"date={self.release_date}, actual={self.actual})>"
        )


# ---------------------------------------------------------------------------
# Market Prices
# ---------------------------------------------------------------------------


class MarketPriceModel(Base):
    """
    OHLCV time series for market instruments.

    TimescaleDB hypertable on timestamp.
    Unique constraint on (ticker, timestamp, timeframe) prevents duplicate bars.
    """

    __tablename__ = "market_prices"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    ticker: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    open: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    vwap: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    timeframe: Mapped[str] = mapped_column(String(10), nullable=False, default="1d")

    __table_args__ = (
        Index("ix_prices_ticker_timestamp", "ticker", "timestamp"),
        Index("ix_prices_timestamp", "timestamp"),
        Index(
            "uq_prices_ticker_ts_tf",
            "ticker",
            "timestamp",
            "timeframe",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<MarketPrice(ticker={self.ticker}, "
            f"ts={self.timestamp}, close={self.close})>"
        )


# ---------------------------------------------------------------------------
# Nowcast Estimates
# ---------------------------------------------------------------------------


class NowcastEstimateModel(Base):
    """
    Stores nowcast predictions generated by models.

    Each row represents one model's prediction for one future release,
    generated at a specific point in time. This allows tracking how
    estimates evolve as new proxy data arrives (estimate revision history).

    TimescaleDB hypertable on estimated_at.
    """

    __tablename__ = "nowcast_estimates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    indicator: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    target_release_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    estimated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    point_estimate: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False)
    confidence_lower: Mapped[float | None] = mapped_column(
        Numeric(12, 4), nullable=True
    )
    confidence_upper: Mapped[float | None] = mapped_column(
        Numeric(12, 4), nullable=True
    )
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, default="")
    features_used: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_nowcast_indicator_target", "indicator", "target_release_date"),
        Index("ix_nowcast_estimated_at", "estimated_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<NowcastEstimate(indicator={self.indicator}, "
            f"target={self.target_release_date}, estimate={self.point_estimate})>"
        )


# ---------------------------------------------------------------------------
# Calendar Events
# ---------------------------------------------------------------------------


class CalendarEventModel(Base):
    """
    Upcoming and past economic calendar events.

    Not a hypertable — standard PostgreSQL table since event count
    is relatively small (hundreds per year, not millions of rows).
    """

    __tablename__ = "calendar_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    indicator: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    country: Mapped[str] = mapped_column(String(3), nullable=False, default="US")
    scheduled_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    importance: Mapped[str] = mapped_column(String(10), nullable=False, default="MEDIUM")
    actual: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    forecast: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    previous: Mapped[float | None] = mapped_column(Numeric(12, 4), nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_calendar_date", "scheduled_date"),
        Index(
            "uq_calendar_indicator_date",
            "indicator",
            "scheduled_date",
            "country",
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<CalendarEvent(indicator={self.indicator}, "
            f"date={self.scheduled_date}, importance={self.importance})>"
        )
