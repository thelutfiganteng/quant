"""
Tests for the Polygon.io market data client.

Uses respx to mock httpx transport — no real API calls are made.
Tests cover: bar fetching, event window bars, snapshots, and empty results.
"""

from __future__ import annotations

from datetime import datetime

import httpx
import pytest
import respx

from src.data_ingestion.market_data_client import MarketDataClient
from src.data_ingestion.schemas import Timeframe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def market_client():
    """Create a market data client with a test API key."""
    client = MarketDataClient(api_key="test_polygon_key", rate_limit=100.0)
    yield client


# ---------------------------------------------------------------------------
# Test: get_bars
# ---------------------------------------------------------------------------


class TestGetBars:
    """Tests for MarketDataClient.get_bars()."""

    @respx.mock
    async def test_get_bars_success(self, market_client, sample_polygon_response):
        """Successful bar fetch returns a DataFrame with correct shape."""
        respx.get(url__startswith="https://api.polygon.io/v2/aggs/ticker/SPY").mock(
            return_value=httpx.Response(200, json=sample_polygon_response)
        )

        df = await market_client.get_bars(
            "SPY",
            timeframe=Timeframe.MINUTE_1,
            start=datetime(2024, 1, 15),
            end=datetime(2024, 1, 15, 23, 59),
        )

        assert not df.empty
        assert len(df) == 3
        assert df.index.name == "timestamp"
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

        # Check first bar values
        assert df.iloc[0]["open"] == pytest.approx(475.0)
        assert df.iloc[0]["close"] == pytest.approx(475.8)
        assert df.iloc[0]["volume"] == 1500000

    @respx.mock
    async def test_get_bars_empty_results(self, market_client):
        """Empty results should return an empty DataFrame."""
        respx.get(url__startswith="https://api.polygon.io/v2/aggs/ticker/").mock(
            return_value=httpx.Response(200, json={"results": []})
        )

        df = await market_client.get_bars("INVALID")
        assert df.empty

    @respx.mock
    async def test_get_bars_no_results_key(self, market_client):
        """Response without 'results' key should return empty DataFrame."""
        respx.get(url__startswith="https://api.polygon.io/v2/aggs/ticker/").mock(
            return_value=httpx.Response(200, json={"status": "OK", "queryCount": 0})
        )

        df = await market_client.get_bars("SPY")
        assert df.empty

    @respx.mock
    async def test_get_bars_daily_timeframe(self, market_client):
        """Daily timeframe should use correct API path."""
        route = respx.get(url__startswith="https://api.polygon.io/v2/aggs/ticker/SPY").mock(
            return_value=httpx.Response(200, json={"results": []})
        )

        await market_client.get_bars("SPY", timeframe=Timeframe.DAY_1)

        # Verify the URL contains /1/day/
        assert "/1/day/" in str(route.calls[0].request.url)

    @respx.mock
    async def test_get_bars_auth_header(self, market_client):
        """API key should be sent as Bearer token in Authorization header."""
        route = respx.get(url__startswith="https://api.polygon.io/v2/aggs/ticker/SPY").mock(
            return_value=httpx.Response(200, json={"results": []})
        )

        await market_client.get_bars("SPY")

        auth_header = route.calls[0].request.headers.get("authorization")
        assert auth_header == "Bearer test_polygon_key"

    @respx.mock
    async def test_get_bars_sorted_ascending(self, market_client, sample_polygon_response):
        """Bars should be sorted by timestamp ascending."""
        respx.get(url__startswith="https://api.polygon.io/v2/aggs/ticker/SPY").mock(
            return_value=httpx.Response(200, json=sample_polygon_response)
        )

        df = await market_client.get_bars("SPY", timeframe=Timeframe.MINUTE_1)

        timestamps = df.index.tolist()
        assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# Test: get_bars_around_event
# ---------------------------------------------------------------------------


class TestGetBarsAroundEvent:
    """Tests for MarketDataClient.get_bars_around_event()."""

    @respx.mock
    async def test_event_bars_success(self, market_client, sample_polygon_response):
        """Event window bars should be fetched correctly."""
        respx.get(url__startswith="https://api.polygon.io/v2/aggs/ticker/SPY").mock(
            return_value=httpx.Response(200, json=sample_polygon_response)
        )

        event_time = datetime(2024, 1, 15, 8, 30)
        df = await market_client.get_bars_around_event(
            "SPY", event_time, window_minutes=60
        )

        assert not df.empty

    @respx.mock
    async def test_event_bars_uses_minute_timeframe(self, market_client):
        """Event bars should default to 1-minute timeframe."""
        route = respx.get(url__startswith="https://api.polygon.io/v2/aggs/ticker/SPY").mock(
            return_value=httpx.Response(200, json={"results": []})
        )

        event_time = datetime(2024, 1, 15, 8, 30)
        await market_client.get_bars_around_event("SPY", event_time)

        # URL should contain /1/minute/
        assert "/1/minute/" in str(route.calls[0].request.url)


# ---------------------------------------------------------------------------
# Test: get_snapshot
# ---------------------------------------------------------------------------


class TestGetSnapshot:
    """Tests for MarketDataClient.get_snapshot()."""

    @respx.mock
    async def test_get_snapshot_success(self, market_client):
        """Snapshot should return correct market data."""
        response = {
            "ticker": {
                "ticker": "SPY",
                "day": {"c": 476.50, "v": 50000000},
                "prevDay": {"c": 475.00},
                "updated": 1705334400000000000,
            }
        }
        respx.get(url__startswith="https://api.polygon.io/v2/snapshot").mock(
            return_value=httpx.Response(200, json=response)
        )

        snapshot = await market_client.get_snapshot("SPY")

        assert snapshot is not None
        assert snapshot.ticker == "SPY"
        assert snapshot.last_price == pytest.approx(476.50)
        assert snapshot.change_pct == pytest.approx(0.3158, rel=1e-2)

    @respx.mock
    async def test_get_snapshot_empty_ticker(self, market_client):
        """Empty ticker data should return None."""
        respx.get(url__startswith="https://api.polygon.io/v2/snapshot").mock(
            return_value=httpx.Response(200, json={"ticker": {}})
        )

        snapshot = await market_client.get_snapshot("SPY")
        assert snapshot is None

    @respx.mock
    async def test_get_snapshot_api_error(self, market_client):
        """API errors should return None (not crash)."""
        respx.get(url__startswith="https://api.polygon.io/v2/snapshot").mock(
            return_value=httpx.Response(403, json={"error": "forbidden"})
        )

        snapshot = await market_client.get_snapshot("SPY")
        assert snapshot is None
