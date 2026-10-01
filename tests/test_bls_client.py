"""
Tests for the BLS API client.

Uses respx to mock httpx transport — no real API calls are made.
Tests cover: successful data fetch, DataFrame conversion,
latest release retrieval, and error handling.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from src.data_ingestion.bls_client import BLSClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bls_client():
    """Create a BLS client with a test API key."""
    client = BLSClient(api_key="test_bls_key", rate_limit=100.0)
    yield client


# ---------------------------------------------------------------------------
# Test: get_series
# ---------------------------------------------------------------------------


class TestGetSeries:
    """Tests for BLSClient.get_series()."""

    @respx.mock
    async def test_get_series_success(self, bls_client, sample_bls_response):
        """Successful fetch returns correctly parsed BLSSeriesData."""
        respx.post("https://api.bls.gov/publicAPI/v2/timeseries/data/").mock(
            return_value=httpx.Response(200, json=sample_bls_response)
        )

        result = await bls_client.get_series(
            ["CUSR0000SA0"], start_year=2024, end_year=2024
        )

        assert "CUSR0000SA0" in result
        series = result["CUSR0000SA0"]
        assert series.series_id == "CUSR0000SA0"
        assert len(series.data) == 3

        # Check most recent data point
        latest = series.data[0]
        assert latest.year == 2024
        assert latest.period == "M06"
        assert latest.value == pytest.approx(313.049)

    @respx.mock
    async def test_get_series_multiple(self, bls_client):
        """Fetching multiple series returns all of them."""
        response = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [
                    {
                        "seriesID": "CUSR0000SA0",
                        "data": [
                            {"year": "2024", "period": "M01", "periodName": "January", "value": "308.0", "footnotes": [{}]},
                        ],
                    },
                    {
                        "seriesID": "CES0000000001",
                        "data": [
                            {"year": "2024", "period": "M01", "periodName": "January", "value": "157500", "footnotes": [{}]},
                        ],
                    },
                ]
            },
        }

        respx.post("https://api.bls.gov/publicAPI/v2/timeseries/data/").mock(
            return_value=httpx.Response(200, json=response)
        )

        result = await bls_client.get_series(["CUSR0000SA0", "CES0000000001"])

        assert len(result) == 2
        assert "CUSR0000SA0" in result
        assert "CES0000000001" in result

    @respx.mock
    async def test_get_series_request_failed(self, bls_client):
        """BLS REQUEST_NOT_SUCCEEDED should return empty results."""
        response = {
            "status": "REQUEST_NOT_SUCCEEDED",
            "message": ["Invalid Series ID"],
            "Results": {},
        }

        respx.post("https://api.bls.gov/publicAPI/v2/timeseries/data/").mock(
            return_value=httpx.Response(200, json=response)
        )

        result = await bls_client.get_series(["INVALID_SERIES"])
        assert result == {}

    async def test_get_series_too_many_ids(self, bls_client):
        """Requesting >50 series should raise ValueError."""
        with pytest.raises(ValueError, match="max 50"):
            await bls_client.get_series([f"SERIES_{i}" for i in range(51)])

    @respx.mock
    async def test_get_series_comma_in_value(self, bls_client):
        """BLS sometimes returns values with commas (e.g., '157,500')."""
        response = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {
                "series": [
                    {
                        "seriesID": "CES0000000001",
                        "data": [
                            {"year": "2024", "period": "M01", "periodName": "January", "value": "157,500", "footnotes": [{}]},
                        ],
                    }
                ]
            },
        }

        respx.post("https://api.bls.gov/publicAPI/v2/timeseries/data/").mock(
            return_value=httpx.Response(200, json=response)
        )

        result = await bls_client.get_series(["CES0000000001"])
        assert result["CES0000000001"].data[0].value == pytest.approx(157500.0)


# ---------------------------------------------------------------------------
# Test: get_series_dataframe
# ---------------------------------------------------------------------------


class TestGetSeriesDataFrame:
    """Tests for BLSClient.get_series_dataframe()."""

    @respx.mock
    async def test_dataframe_conversion(self, bls_client, sample_bls_response):
        """Series data should be converted to a proper DataFrame."""
        respx.post("https://api.bls.gov/publicAPI/v2/timeseries/data/").mock(
            return_value=httpx.Response(200, json=sample_bls_response)
        )

        df = await bls_client.get_series_dataframe("CUSR0000SA0", start_year=2024)

        assert not df.empty
        assert "value" in df.columns
        assert df.index.name == "date"
        assert len(df) == 3

    @respx.mock
    async def test_dataframe_empty_result(self, bls_client):
        """Empty BLS response should produce empty DataFrame."""
        response = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {"series": [{"seriesID": "TEST", "data": []}]},
        }
        respx.post("https://api.bls.gov/publicAPI/v2/timeseries/data/").mock(
            return_value=httpx.Response(200, json=response)
        )

        df = await bls_client.get_series_dataframe("TEST")
        assert df.empty


# ---------------------------------------------------------------------------
# Test: get_latest_release
# ---------------------------------------------------------------------------


class TestGetLatestRelease:
    """Tests for BLSClient.get_latest_release()."""

    @respx.mock
    async def test_latest_release_success(self, bls_client, sample_bls_response):
        """Latest release should return the most recent data point."""
        respx.post("https://api.bls.gov/publicAPI/v2/timeseries/data/").mock(
            return_value=httpx.Response(200, json=sample_bls_response)
        )

        latest = await bls_client.get_latest_release("CUSR0000SA0")

        assert latest is not None
        assert latest.period == "M06"
        assert latest.value == pytest.approx(313.049)

    @respx.mock
    async def test_latest_release_none(self, bls_client):
        """Should return None when no data available."""
        response = {
            "status": "REQUEST_SUCCEEDED",
            "Results": {"series": []},
        }
        respx.post("https://api.bls.gov/publicAPI/v2/timeseries/data/").mock(
            return_value=httpx.Response(200, json=response)
        )

        latest = await bls_client.get_latest_release("NONEXISTENT")
        assert latest is None
