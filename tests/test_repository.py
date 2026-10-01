"""
Tests for the repository layer.

These tests verify the repository pattern logic using in-memory
SQLite for unit tests. Since SQLite doesn't support ON CONFLICT
the same way as PostgreSQL, we test the core query logic here.
Full PostgreSQL integration tests are marked with @pytest.mark.integration.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.data_ingestion.schemas import (
    EconomicIndicator,
    EconomicRelease,
    MarketBar,
    NowcastEstimate,
)
from src.storage.models import (
    Base,
    EconomicReleaseModel,
    MarketPriceModel,
    NowcastEstimateModel,
)
from src.storage.repository import (
    EconomicReleaseRepository,
    MarketPriceRepository,
    NowcastRepository,
)


# ---------------------------------------------------------------------------
# Fixtures: In-memory async SQLite for testing
# ---------------------------------------------------------------------------


@pytest.fixture
async def async_engine():
    """Create an async SQLite engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def session(async_engine) -> AsyncSession:
    """Create an async session for testing."""
    factory = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with factory() as session:
        yield session


# ---------------------------------------------------------------------------
# Helper: direct insert for SQLite (bypasses PostgreSQL ON CONFLICT)
# ---------------------------------------------------------------------------

async def _insert_release(session: AsyncSession, release: EconomicRelease) -> None:
    """Insert an economic release directly (SQLite-compatible)."""
    model = EconomicReleaseModel(
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
    session.add(model)
    await session.flush()


async def _insert_bar(session: AsyncSession, bar: MarketBar, timeframe: str = "1d") -> None:
    """Insert a market price bar directly (SQLite-compatible)."""
    model = MarketPriceModel(
        ticker=bar.ticker,
        timestamp=bar.timestamp,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        vwap=bar.vwap,
        timeframe=timeframe,
    )
    session.add(model)
    await session.flush()


# ---------------------------------------------------------------------------
# Test: EconomicReleaseRepository
# ---------------------------------------------------------------------------


class TestEconomicReleaseRepository:
    """Tests for EconomicReleaseRepository query methods."""

    async def test_insert_and_get_releases(self, session):
        """Should insert and retrieve releases for an indicator."""
        repo = EconomicReleaseRepository(session)
        release = EconomicRelease(
            indicator=EconomicIndicator.CPI,
            release_date=datetime(2024, 7, 11, 8, 30),
            actual=3.0,
            consensus=3.1,
            previous=3.3,
            source="FRED",
        )

        await _insert_release(session, release)
        await session.commit()

        releases = await repo.get_releases("CPI")
        assert len(releases) == 1
        assert float(releases[0].actual) == pytest.approx(3.0)
        assert float(releases[0].consensus) == pytest.approx(3.1)

    async def test_get_releases_with_date_range(self, session):
        """Should filter releases by date range."""
        repo = EconomicReleaseRepository(session)

        for month in range(1, 7):
            release = EconomicRelease(
                indicator=EconomicIndicator.CPI,
                release_date=datetime(2024, month, 11, 8, 30),
                actual=3.0 + month * 0.1,
                consensus=3.0,
                source="FRED",
            )
            await _insert_release(session, release)
        await session.commit()

        releases = await repo.get_releases(
            "CPI",
            start=datetime(2024, 3, 1),
            end=datetime(2024, 5, 31),
        )

        assert len(releases) == 3  # March, April, May

    async def test_get_latest_release(self, session):
        """Should return the most recent release."""
        repo = EconomicReleaseRepository(session)

        for month in range(1, 4):
            release = EconomicRelease(
                indicator=EconomicIndicator.NFP,
                release_date=datetime(2024, month, 5, 8, 30),
                actual=200.0 + month,
                source="BLS",
            )
            await _insert_release(session, release)
        await session.commit()

        latest = await repo.get_latest_release("NFP")
        assert latest is not None
        assert float(latest.actual) == pytest.approx(203.0)

    async def test_get_latest_release_none(self, session):
        """Should return None when no releases exist."""
        repo = EconomicReleaseRepository(session)
        latest = await repo.get_latest_release("NONEXISTENT")
        assert latest is None

    async def test_get_surprise_history(self, session):
        """Should return surprise history as a DataFrame."""
        repo = EconomicReleaseRepository(session)

        for i in range(5):
            release = EconomicRelease(
                indicator=EconomicIndicator.CPI,
                release_date=datetime(2024, 1 + i, 11, 8, 30),
                actual=3.0 + i * 0.1,
                consensus=3.05 + i * 0.1,
                surprise=-0.05,  # actual - consensus
                source="FRED",
            )
            await _insert_release(session, release)
        await session.commit()

        df = await repo.get_surprise_history("CPI", n=5)
        assert len(df) == 5
        assert "surprise" in df.columns

    async def test_multiple_indicators(self, session):
        """Releases for different indicators should be isolated."""
        repo = EconomicReleaseRepository(session)

        for indicator in [EconomicIndicator.CPI, EconomicIndicator.NFP]:
            release = EconomicRelease(
                indicator=indicator,
                release_date=datetime(2024, 7, 11, 8, 30),
                actual=3.0,
                source="FRED",
            )
            await _insert_release(session, release)
        await session.commit()

        cpi_releases = await repo.get_releases("CPI")
        nfp_releases = await repo.get_releases("NFP")
        assert len(cpi_releases) == 1
        assert len(nfp_releases) == 1


# ---------------------------------------------------------------------------
# Test: MarketPriceRepository
# ---------------------------------------------------------------------------


class TestMarketPriceRepository:
    """Tests for MarketPriceRepository query methods."""

    async def test_insert_and_get_bars(self, session):
        """Should insert and retrieve bars."""
        repo = MarketPriceRepository(session)
        bars = [
            MarketBar(
                ticker="SPY",
                timestamp=datetime(2024, 7, 11, 9, 30) + timedelta(minutes=i),
                open=475.0 + i,
                high=476.0 + i,
                low=474.0 + i,
                close=475.5 + i,
                volume=100000,
            )
            for i in range(5)
        ]

        for bar in bars:
            await _insert_bar(session, bar)
        await session.commit()

        df = await repo.get_bars("SPY")
        assert len(df) == 5

    async def test_get_bars_with_date_range(self, session):
        """Should filter bars by date range."""
        repo = MarketPriceRepository(session)
        base = datetime(2024, 7, 11, 9, 30)
        bars = [
            MarketBar(
                ticker="SPY",
                timestamp=base + timedelta(hours=i),
                open=475.0,
                high=476.0,
                low=474.0,
                close=475.5,
                volume=100000,
            )
            for i in range(10)
        ]

        for bar in bars:
            await _insert_bar(session, bar)
        await session.commit()

        df = await repo.get_bars(
            "SPY",
            start=base + timedelta(hours=2),
            end=base + timedelta(hours=5),
        )
        assert len(df) == 4

    async def test_get_bars_empty(self, session):
        """Should return empty DataFrame for non-existent ticker."""
        repo = MarketPriceRepository(session)
        df = await repo.get_bars("NONEXISTENT")
        assert df.empty

    async def test_get_bars_around_event(self, session):
        """Should fetch bars within a time window around an event."""
        repo = MarketPriceRepository(session)
        event_time = datetime(2024, 7, 11, 8, 30)

        # Insert bars from 7:00 to 10:00
        for minutes in range(0, 180, 1):
            bar = MarketBar(
                ticker="SPY",
                timestamp=datetime(2024, 7, 11, 7, 0) + timedelta(minutes=minutes),
                open=475.0,
                high=476.0,
                low=474.0,
                close=475.5,
                volume=100000,
            )
            await _insert_bar(session, bar, timeframe="1m")
        await session.commit()

        df = await repo.get_bars_around_event("SPY", event_time, window_minutes=30, timeframe="1m")
        # Should have bars from 8:00 to 9:00 (60 minutes)
        assert len(df) == 61  # inclusive


# ---------------------------------------------------------------------------
# Test: NowcastRepository
# ---------------------------------------------------------------------------


class TestNowcastRepository:
    """Tests for NowcastRepository."""

    async def test_save_and_retrieve_estimate(self, session):
        """Should save and retrieve a nowcast estimate."""
        repo = NowcastRepository(session)
        estimate = NowcastEstimate(
            indicator=EconomicIndicator.CPI,
            target_release_date=datetime(2024, 8, 14, 8, 30),
            estimated_at=datetime(2024, 8, 13, 12, 0),
            point_estimate=2.9,
            confidence_lower=2.7,
            confidence_upper=3.1,
            model_name="cpi_bridge_v1",
            features_used={"gas": 3.45},
        )

        await repo.save_estimate(estimate)
        await session.commit()

        estimates = await repo.get_estimates("CPI")
        assert len(estimates) == 1
        assert float(estimates[0].point_estimate) == pytest.approx(2.9)
        assert estimates[0].model_name == "cpi_bridge_v1"

    async def test_get_estimates_filtered_by_target_date(self, session):
        """Should filter estimates by target release date."""
        repo = NowcastRepository(session)

        for day in [14, 15]:
            estimate = NowcastEstimate(
                indicator=EconomicIndicator.CPI,
                target_release_date=datetime(2024, 8, day, 8, 30),
                estimated_at=datetime(2024, 8, day - 1, 12, 0),
                point_estimate=2.9,
                model_name="test",
            )
            await repo.save_estimate(estimate)
        await session.commit()

        estimates = await repo.get_estimates(
            "CPI", target_release_date=datetime(2024, 8, 14, 8, 30)
        )
        assert len(estimates) == 1

    async def test_get_latest_estimate(self, session):
        """Should return the most recent estimate."""
        repo = NowcastRepository(session)

        for hour in range(3):
            estimate = NowcastEstimate(
                indicator=EconomicIndicator.CPI,
                target_release_date=datetime(2024, 8, 14, 8, 30),
                estimated_at=datetime(2024, 8, 13, 10 + hour, 0),
                point_estimate=2.9 + hour * 0.01,
                model_name="test",
            )
            await repo.save_estimate(estimate)
        await session.commit()

        latest = await repo.get_latest_estimate("CPI")
        assert latest is not None
        assert float(latest.point_estimate) == pytest.approx(2.92)

    async def test_get_latest_estimate_none(self, session):
        """Should return None when no estimates exist."""
        repo = NowcastRepository(session)
        latest = await repo.get_latest_estimate("NONEXISTENT")
        assert latest is None
