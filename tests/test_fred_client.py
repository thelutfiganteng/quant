"""
Tests for the FRED API client.

Uses respx to mock httpx transport — no real API calls are made.
Tests cover: successful data fetch, missing values handling,
error responses, retry behavior, and series info retrieval.
"""

from __future__ import annotations

from datetime import datetime

import httpx
import pandas as pd
import pytest
import respx

from src.data_ingestion.fred_client import FREDClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fred_client():
    """Create a FRED client with a test API key."""
    client = FREDClient(api_key="test_fred_key", rate_limit=100.0)
    yield client


# ---------------------------------------------------------------------------
# Test: get_series
# ---------------------------------------------------------------------------


class TestGetSeries:
    """Tests for FREDClient.get_series()."""

    @respx.mock
    async def test_get_series_success(self, fred_client, sample_fred_response):
        """Successful fetch returns a DataFrame with correct shape and values."""
        respx.get("https://api.stlouisfed.org/fred/series/observations").mock(
            return_value=httpx.Response(200, json=sample_fred_response)
        )

        df = await fred_client.get_series(
            "CPIAUCSL",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 30),
        )

        assert not df.empty
        assert len(df) == 6
        assert "value" in df.columns
        assert df.index.name == "date"

        # Check first value
        assert df.iloc[0]["value"] == pytest.approx(308.417)
        # Check last value
        assert df.iloc[-1]["value"] == pytest.approx(313.049)

    @respx.mock
    async def test_get_series_with_missing_values(self, fred_client):
        """FRED returns '.' for missing values — should become NaN in DataFrame."""
        response = {
            "observations": [
                {"date": "2024-01-01", "value": "100.0"},
                {"date": "2024-02-01", "value": "."},
                {"date": "2024-03-01", "value": "102.0"},
            ]
        }
        respx.get("https://api.stlouisfed.org/fred/series/observations").mock(
            return_value=httpx.Response(200, json=response)
        )

        df = await fred_client.get_series("TEST")

        assert len(df) == 3
        assert df.iloc[0]["value"] == 100.0
        assert pd.isna(df.iloc[1]["value"])  # None becomes NaN in float column
        assert df.iloc[2]["value"] == 102.0

    @respx.mock
    async def test_get_series_empty_response(self, fred_client):
        """Empty observations should return an empty DataFrame."""
        respx.get("https://api.stlouisfed.org/fred/series/observations").mock(
            return_value=httpx.Response(200, json={"observations": []})
        )

        df = await fred_client.get_series("NONEXISTENT")

        assert df.empty
        assert list(df.columns) == ["date", "value"]

    @respx.mock
    async def test_get_series_api_error(self, fred_client):
        """Non-retryable API errors should raise immediately."""
        respx.get("https://api.stlouisfed.org/fred/series/observations").mock(
            return_value=httpx.Response(403, json={"error_message": "Bad API key"})
        )

        from src.data_ingestion.base_client import AuthenticationError

        with pytest.raises(AuthenticationError):
            await fred_client.get_series("CPIAUCSL")

    @respx.mock
    async def test_get_series_retry_on_500(self, fred_client):
        """Server errors should be retried, then succeed."""
        route = respx.get("https://api.stlouisfed.org/fred/series/observations")
        route.side_effect = [
            httpx.Response(500, json={"error": "Internal Server Error"}),
            httpx.Response(200, json={"observations": [{"date": "2024-01-01", "value": "100.0"}]}),
        ]

        df = await fred_client.get_series("CPIAUCSL")
        assert len(df) == 1
        assert route.call_count == 2

    @respx.mock
    async def test_get_series_with_frequency(self, fred_client):
        """Frequency parameter should be passed to the API."""
        route = respx.get("https://api.stlouisfed.org/fred/series/observations").mock(
            return_value=httpx.Response(200, json={"observations": [{"date": "2024-01-01", "value": "100.0"}]})
        )

        await fred_client.get_series("GDP", frequency="q")

        # Verify the frequency param was sent
        assert route.calls[0].request.url.params["frequency"] == "q"


# ---------------------------------------------------------------------------
# Test: get_series_info
# ---------------------------------------------------------------------------


class TestGetSeriesInfo:
    """Tests for FREDClient.get_series_info()."""

    @respx.mock
    async def test_get_series_info_success(self, fred_client, sample_fred_series_info_response):
        """Successful metadata fetch returns correct FREDSeriesInfo."""
        respx.get("https://api.stlouisfed.org/fred/series").mock(
            return_value=httpx.Response(200, json=sample_fred_series_info_response)
        )

        info = await fred_client.get_series_info("CPIAUCSL")

        assert info.series_id == "CPIAUCSL"
        assert "Consumer Price Index" in info.title
        assert info.frequency == "Monthly"
        assert info.seasonal_adjustment == "Seasonally Adjusted"

    @respx.mock
    async def test_get_series_info_not_found(self, fred_client):
        """Missing series should raise ValueError."""
        respx.get("https://api.stlouisfed.org/fred/series").mock(
            return_value=httpx.Response(200, json={"serieses": []})
        )

        with pytest.raises(ValueError, match="No series found"):
            await fred_client.get_series_info("NONEXISTENT")


# ---------------------------------------------------------------------------
# Test: get_release_dates
# ---------------------------------------------------------------------------


class TestGetReleaseDates:
    """Tests for FREDClient.get_release_dates()."""

    @respx.mock
    async def test_get_release_dates_success(self, fred_client):
        """Release dates should be fetched and parsed correctly."""
        # Mock the series/release endpoint
        respx.get("https://api.stlouisfed.org/fred/series/release").mock(
            return_value=httpx.Response(
                200,
                json={
                    "releases": [
                        {"id": 10, "name": "Consumer Price Index"}
                    ]
                },
            )
        )

        # Mock the release/dates endpoint
        respx.get("https://api.stlouisfed.org/fred/release/dates").mock(
            return_value=httpx.Response(
                200,
                json={
                    "release_dates": [
                        {"release_id": 10, "date": "2024-06-12"},
                        {"release_id": 10, "date": "2024-07-11"},
                    ]
                },
            )
        )

        dates = await fred_client.get_release_dates("CPIAUCSL")

        assert len(dates) == 2
        assert dates[0].release_name == "Consumer Price Index"
        assert dates[1].date == datetime(2024, 7, 11)


# ---------------------------------------------------------------------------
# Test: search_series
# ---------------------------------------------------------------------------


class TestSearchSeries:
    """Tests for FREDClient.search_series()."""

    @respx.mock
    async def test_search_series_success(self, fred_client):
        """Search should return matching series."""
        respx.get("https://api.stlouisfed.org/fred/series/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "serieses": [
                        {
                            "id": "CPIAUCSL",
                            "title": "Consumer Price Index",
                            "frequency": "Monthly",
                            "units": "Index",
                            "seasonal_adjustment": "SA",
                        },
                        {
                            "id": "CPILFESL",
                            "title": "Core CPI",
                            "frequency": "Monthly",
                            "units": "Index",
                            "seasonal_adjustment": "SA",
                        },
                    ]
                },
            )
        )

        results = await fred_client.search_series("CPI", limit=5)
        assert len(results) == 2
        assert results[0].series_id == "CPIAUCSL"
