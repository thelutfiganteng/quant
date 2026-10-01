"""
Repository pattern for data access.

All database queries are encapsulated here — business logic
should NEVER contain raw SQL or direct ORM queries.

Each repository takes an AsyncSession in its constructor,
allowing dependency injection and easy testing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd
import structlog
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.data_ingestion.schemas import (
    EconomicRelease,
    MarketBar,
    NowcastEstimate,
)
from src.storage.models import (
    CalendarEventModel,
    EconomicReleaseModel,
    MarketPriceModel,
    NowcastEstimateModel,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Economic Releases Repository
# ---------------------------------------------------------------------------


class EconomicReleaseRepository:
    """Data access for economic_releases table."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def upsert_release(self, release: EconomicRelease) -> None:
        """
        Insert or update an economic release (idempotent).

        Uses PostgreSQL ON CONFLICT ... DO UPDATE to handle duplicates.
        The unique constraint is on (indicator, release_date, source).
        """
        stmt = pg_insert(EconomicReleaseModel).values(
            indicator=release.indicator.value,
            release_date=release.release_date,
            actual=release.actual,
            consensus=release.consensus,
            previous=release.previous,
            surprise=release.compute_surprise(),
            surprise_zscore=release.surprise_zscore,
            revision=release.revision,
            source=release.source,
        )

        # On conflict, update the values (data may be revised after initial release)
        stmt = stmt.on_conflict_do_update(
            index_elements=["indicator", "release_date", "source"],
            set_={
                "actual": stmt.excluded.actual,
                "consensus": stmt.excluded.consensus,
                "previous": stmt.excluded.previous,
                "surprise": stmt.excluded.surprise,
                "surprise_zscore": stmt.excluded.surprise_zscore,
                "revision": stmt.excluded.revision,
            },
        )

        await self._session.execute(stmt)
        logger.debug(
            "release_upserted",
            indicator=release.indicator.value,
            date=str(release.release_date),
        )

    async def bulk_upsert_releases(self, releases: list[EconomicRelease]) -> int:
        """
        Bulk upsert multiple releases. Returns count of rows affected.
        """
        if not releases:
            return 0

        values = [
            {
                "indicator": r.indicator.value,
                "release_date": r.release_date,
                "actual": r.actual,
                "consensus": r.consensus,
                "previous": r.previous,
                "surprise": r.compute_surprise(),
                "surprise_zscore": r.surprise_zscore,
                "revision": r.revision,
                "source": r.source,
            }
            for r in releases
        ]

        stmt = pg_insert(EconomicReleaseModel).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["indicator", "release_date", "source"],
            set_={
                "actual": stmt.excluded.actual,
                "consensus": stmt.excluded.consensus,
                "previous": stmt.excluded.previous,
                "surprise": stmt.excluded.surprise,
                "surprise_zscore": stmt.excluded.surprise_zscore,
                "revision": stmt.excluded.revision,
            },
        )

        result = await self._session.execute(stmt)
        count = result.rowcount if result.rowcount else len(values)
        logger.info("releases_bulk_upserted", count=count)
        return count

    async def get_releases(
        self,
        indicator: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> Sequence[EconomicReleaseModel]:
        """Fetch releases for an indicator within a date range."""
        query = (
            select(EconomicReleaseModel)
            .where(EconomicReleaseModel.indicator == indicator)
            .order_by(EconomicReleaseModel.release_date.asc())
        )

        if start:
            query = query.where(EconomicReleaseModel.release_date >= start)
        if end:
            query = query.where(EconomicReleaseModel.release_date <= end)

        result = await self._session.execute(query)
        return result.scalars().all()

    async def get_latest_release(
        self, indicator: str
    ) -> EconomicReleaseModel | None:
        """Get the most recent release for an indicator."""
        query = (
            select(EconomicReleaseModel)
            .where(EconomicReleaseModel.indicator == indicator)
            .order_by(EconomicReleaseModel.release_date.desc())
            .limit(1)
        )
        result = await self._session.execute(query)
        return result.scalar_one_or_none()

    async def get_surprise_history(
        self,
        indicator: str,
        n: int = 60,
    ) -> pd.DataFrame:
        """
        Get the last N surprises for an indicator as a DataFrame.

        Useful for computing rolling surprise z-scores and
        historical surprise distributions.
        """
        query = (
            select(
                EconomicReleaseModel.release_date,
                EconomicReleaseModel.actual,
                EconomicReleaseModel.consensus,
                EconomicReleaseModel.surprise,
                EconomicReleaseModel.surprise_zscore,
            )
            .where(EconomicReleaseModel.indicator == indicator)
            .where(EconomicReleaseModel.surprise.isnot(None))
            .order_by(EconomicReleaseModel.release_date.desc())
            .limit(n)
        )
        result = await self._session.execute(query)
        rows = result.all()

        if not rows:
            return pd.DataFrame(
                columns=["release_date", "actual", "consensus", "surprise", "surprise_zscore"]
            )

        df = pd.DataFrame(rows, columns=["release_date", "actual", "consensus", "surprise", "surprise_zscore"])
        df = df.sort_values("release_date").reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# Market Price Repository
# ---------------------------------------------------------------------------


class MarketPriceRepository:
    """Data access for market_prices table."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def bulk_insert_bars(self, bars: list[MarketBar]) -> int:
        """
        Bulk upsert OHLCV bars (idempotent).

        Uses ON CONFLICT on (ticker, timestamp, timeframe) to skip duplicates.
        """
        if not bars:
            return 0

        values = [
            {
                "ticker": b.ticker,
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "vwap": b.vwap,
                "timeframe": "1d",  # Default; override in caller if needed
            }
            for b in bars
        ]

        stmt = pg_insert(MarketPriceModel).values(values)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["ticker", "timestamp", "timeframe"],
        )

        result = await self._session.execute(stmt)
        count = result.rowcount if result.rowcount else 0
        logger.info("bars_bulk_inserted", ticker=bars[0].ticker if bars else "", count=count)
        return count

    async def get_bars(
        self,
        ticker: str,
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for a ticker within a date range."""
        query = (
            select(MarketPriceModel)
            .where(MarketPriceModel.ticker == ticker)
            .where(MarketPriceModel.timeframe == timeframe)
            .order_by(MarketPriceModel.timestamp.asc())
        )

        if start:
            query = query.where(MarketPriceModel.timestamp >= start)
        if end:
            query = query.where(MarketPriceModel.timestamp <= end)

        result = await self._session.execute(query)
        rows = result.scalars().all()

        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume", "vwap"]
            )

        data = [
            {
                "timestamp": r.timestamp,
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": r.volume,
                "vwap": float(r.vwap) if r.vwap else None,
            }
            for r in rows
        ]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        return df

    async def get_bars_around_event(
        self,
        ticker: str,
        event_time: datetime,
        window_minutes: int = 60,
        timeframe: str = "1m",
    ) -> pd.DataFrame:
        """
        Fetch bars in a window around an event time.

        Args:
            ticker: Instrument ticker.
            event_time: Event datetime.
            window_minutes: Minutes before and after.
            timeframe: Bar timeframe.

        Returns:
            DataFrame centered around the event.
        """
        from datetime import timedelta

        start = event_time - timedelta(minutes=window_minutes)
        end = event_time + timedelta(minutes=window_minutes)
        return await self.get_bars(ticker, start=start, end=end, timeframe=timeframe)


# ---------------------------------------------------------------------------
# Nowcast Repository
# ---------------------------------------------------------------------------


class NowcastRepository:
    """Data access for nowcast_estimates table."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def save_estimate(self, estimate: NowcastEstimate) -> None:
        """Save a new nowcast estimate."""
        model = NowcastEstimateModel(
            indicator=estimate.indicator.value,
            target_release_date=estimate.target_release_date,
            estimated_at=estimate.estimated_at,
            point_estimate=estimate.point_estimate,
            confidence_lower=estimate.confidence_lower,
            confidence_upper=estimate.confidence_upper,
            model_name=estimate.model_name,
            features_used=estimate.features_used,
        )
        self._session.add(model)
        logger.debug(
            "nowcast_saved",
            indicator=estimate.indicator.value,
            estimate=estimate.point_estimate,
        )

    async def get_estimates(
        self,
        indicator: str,
        target_release_date: datetime | None = None,
    ) -> Sequence[NowcastEstimateModel]:
        """
        Get all estimates for an indicator, optionally filtered by target date.

        Returns estimates ordered by estimated_at (ascending) to show
        how the forecast evolved over time.
        """
        query = (
            select(NowcastEstimateModel)
            .where(NowcastEstimateModel.indicator == indicator)
            .order_by(NowcastEstimateModel.estimated_at.asc())
        )

        if target_release_date:
            query = query.where(
                NowcastEstimateModel.target_release_date == target_release_date
            )

        result = await self._session.execute(query)
        return result.scalars().all()

    async def get_latest_estimate(
        self, indicator: str
    ) -> NowcastEstimateModel | None:
        """Get the most recent estimate for an indicator."""
        query = (
            select(NowcastEstimateModel)
            .where(NowcastEstimateModel.indicator == indicator)
            .order_by(NowcastEstimateModel.estimated_at.desc())
            .limit(1)
        )
        result = await self._session.execute(query)
        return result.scalar_one_or_none()
