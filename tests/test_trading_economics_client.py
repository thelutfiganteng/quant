"""
Tests for the Trading Economics API client.

Uses respx to mock httpx transport — no real API calls are made.
Tests cover: calendar fetch, fallback mode behavior, consensus fetch,
and helper methods.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.data_ingestion.schemas import Importance
from src.data_ingestion.trading_economics_client import TradingEconomicsClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def te_client():
    """Create a TE client with a test API key."""
    client = TradingEconomicsClient(api_key="test_te_key", rate_limit=100.0)
    yield client


@pytest.fixture
def te_client_no_key():
    """Create a TE client WITHOUT an API key (fallback mode)."""
    client = TradingEconomicsClient(api_key=None, rate_limit=100.0)
    yield client


# ---------------------------------------------------------------------------
# Test: Fallback Mode
# ---------------------------------------------------------------------------


class TestFallbackMode:
    """Tests for graceful degradation when no API key is configured."""

    async def test_calendar_returns_empty_in_fallback(self, te_client_no_key):
        """Without API key, get_calendar should return empty list."""
        result = await te_client_no_key.get_calendar()
        assert result == []

    async def test_consensus_returns_empty_in_fallback(self, te_client_no_key):
        """Without API key, get_consensus should return empty list."""
        result = await te_client_no_key.get_consensus()
        assert result == []

    async def test_historical_returns_empty_in_fallback(self, te_client_no_key):
        """Without API key, get_historical_indicator should return empty list."""
        result = await te_client_no_key.get_historical_indicator()
        assert result == []

    def test_fallback_flag_is_set(self, te_client_no_key):
        """Fallback mode flag should be True when no key."""
        assert te_client_no_key._fallback_mode is True

    def test_normal_mode_flag(self, te_client):
        """Fallback mode flag should be False with valid key."""
        assert te_client._fallback_mode is False


# ---------------------------------------------------------------------------
# Test: get_calendar
# ---------------------------------------------------------------------------


class TestGetCalendar:
    """Tests for TradingEconomicsClient.get_calendar()."""

    @respx.mock
    async def test_get_calendar_success(self, te_client, sample_te_calendar_response):
        """Successful calendar fetch returns parsed events."""
        respx.get("https://api.tradingeconomics.com/calendar/country/united states").mock(
            return_value=httpx.Response(200, json=sample_te_calendar_response)
        )

        events = await te_client.get_calendar(country="united states")

        assert len(events) == 2

        # CPI event
        cpi_event = events[0]
        assert cpi_event.indicator == "CPI"
        assert cpi_event.actual == 3.0
        assert cpi_event.forecast == 3.1
        assert cpi_event.surprise == pytest.approx(-0.1)
        assert cpi_event.importance == Importance.HIGH

        # NFP event
        nfp_event = events[1]
        assert nfp_event.indicator == "Non Farm Payrolls"
        assert nfp_event.actual == 206.0
        assert nfp_event.surprise == pytest.approx(16.0)

    @respx.mock
    async def test_get_calendar_empty_response(self, te_client):
        """Empty API response returns empty list."""
        respx.get("https://api.tradingeconomics.com/calendar/country/united states").mock(
            return_value=httpx.Response(200, json=[])
        )

        events = await te_client.get_calendar()
        assert events == []

    @respx.mock
    async def test_get_calendar_non_list_response(self, te_client):
        """Non-list response (error object) should return empty list."""
        respx.get("https://api.tradingeconomics.com/calendar/country/united states").mock(
            return_value=httpx.Response(200, json={"error": "unauthorized"})
        )

        events = await te_client.get_calendar()
        assert events == []


# ---------------------------------------------------------------------------
# Test: get_consensus
# ---------------------------------------------------------------------------


class TestGetConsensus:
    """Tests for TradingEconomicsClient.get_consensus()."""

    @respx.mock
    async def test_get_consensus_success(self, te_client):
        """Consensus data should be parsed correctly."""
        response = [
            {
                "Category": "Inflation Rate",
                "Date": "2024-08-14T08:30:00",
                "ForecastValue": 2.9,
                "ForecastValue1": 3.1,
                "ForecastValue2": 2.7,
                "LatestValue": 3.0,
            }
        ]
        respx.get("https://api.tradingeconomics.com/forecast/country/united states").mock(
            return_value=httpx.Response(200, json=response)
        )

        results = await te_client.get_consensus()

        assert len(results) == 1
        assert results[0].consensus == 2.9
        assert results[0].high_estimate == 3.1
        assert results[0].low_estimate == 2.7
        assert results[0].previous == 3.0

    @respx.mock
    async def test_get_consensus_filters_null_forecast(self, te_client):
        """Items without ForecastValue should be filtered out."""
        response = [
            {
                "Category": "GDP",
                "Date": "2024-08-28T08:30:00",
                "ForecastValue": None,
                "LatestValue": 2.8,
            },
            {
                "Category": "Inflation Rate",
                "Date": "2024-08-14T08:30:00",
                "ForecastValue": 2.9,
                "LatestValue": 3.0,
            },
        ]
        respx.get("https://api.tradingeconomics.com/forecast/country/united states").mock(
            return_value=httpx.Response(200, json=response)
        )

        results = await te_client.get_consensus()
        assert len(results) == 1
        assert results[0].indicator == "Inflation Rate"


# ---------------------------------------------------------------------------
# Test: Helper Methods
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for static helper methods."""

    def test_safe_float_valid(self):
        assert TradingEconomicsClient._safe_float(3.14) == pytest.approx(3.14)
        assert TradingEconomicsClient._safe_float("2.5") == pytest.approx(2.5)
        assert TradingEconomicsClient._safe_float(0) == 0.0

    def test_safe_float_invalid(self):
        assert TradingEconomicsClient._safe_float(None) is None
        assert TradingEconomicsClient._safe_float("") is None
        assert TradingEconomicsClient._safe_float("N/A") is None

    def test_map_importance(self):
        assert TradingEconomicsClient._map_importance(3) == Importance.HIGH
        assert TradingEconomicsClient._map_importance(2) == Importance.MEDIUM
        assert TradingEconomicsClient._map_importance(1) == Importance.LOW
        assert TradingEconomicsClient._map_importance("invalid") == Importance.MEDIUM
