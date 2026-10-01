"""
Shared pytest fixtures for the macro-quant test suite.

Provides:
- Mock settings (no real API keys needed)
- Sample data factories for each schema type
- httpx mock transport setup via respx
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.config import Settings
from src.data_ingestion.schemas import (
    BLSDataPoint,
    CalendarEvent,
    EconomicIndicator,
    EconomicRelease,
    Importance,
    MarketBar,
    NowcastEstimate,
)


# ---------------------------------------------------------------------------
# Settings Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_settings() -> Settings:
    """Create test settings that don't require real API keys."""
    return Settings(
        environment="development",
        log_level="DEBUG",
        database_url="postgresql+asyncpg://test:test@localhost:5432/test_db",
        database_url_sync="postgresql+psycopg2://test:test@localhost:5432/test_db",
        timescaledb_enabled=False,
        fred_api_key="test_fred_key",
        bls_api_key="test_bls_key",
        trading_economics_api_key="test_te_key",
        polygon_api_key="test_polygon_key",
    )


# ---------------------------------------------------------------------------
# Sample Data Factories
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_fred_response() -> dict:
    """Mock FRED API response for observations endpoint."""
    return {
        "realtime_start": "2024-01-01",
        "realtime_end": "2024-12-31",
        "observation_start": "2024-01-01",
        "observation_end": "2024-06-30",
        "units": "lin",
        "output_type": 1,
        "file_type": "json",
        "order_by": "observation_date",
        "sort_order": "asc",
        "count": 6,
        "offset": 0,
        "limit": 100000,
        "observations": [
            {"realtime_start": "2024-01-01", "realtime_end": "2024-12-31", "date": "2024-01-01", "value": "308.417"},
            {"realtime_start": "2024-01-01", "realtime_end": "2024-12-31", "date": "2024-02-01", "value": "310.326"},
            {"realtime_start": "2024-01-01", "realtime_end": "2024-12-31", "date": "2024-03-01", "value": "312.230"},
            {"realtime_start": "2024-01-01", "realtime_end": "2024-12-31", "date": "2024-04-01", "value": "313.207"},
            {"realtime_start": "2024-01-01", "realtime_end": "2024-12-31", "date": "2024-05-01", "value": "313.225"},
            {"realtime_start": "2024-01-01", "realtime_end": "2024-12-31", "date": "2024-06-01", "value": "313.049"},
        ],
    }


@pytest.fixture
def sample_fred_series_info_response() -> dict:
    """Mock FRED API response for series info."""
    return {
        "serieses": [
            {
                "id": "CPIAUCSL",
                "title": "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
                "frequency": "Monthly",
                "units": "Index 1982-1984=100",
                "seasonal_adjustment": "Seasonally Adjusted",
                "last_updated": "2024-07-11 07:36:03-05",
            }
        ]
    }


@pytest.fixture
def sample_bls_response() -> dict:
    """Mock BLS API v2 response."""
    return {
        "status": "REQUEST_SUCCEEDED",
        "responseTime": 50,
        "message": [],
        "Results": {
            "series": [
                {
                    "seriesID": "CUSR0000SA0",
                    "data": [
                        {
                            "year": "2024",
                            "period": "M06",
                            "periodName": "June",
                            "value": "313.049",
                            "footnotes": [{}],
                        },
                        {
                            "year": "2024",
                            "period": "M05",
                            "periodName": "May",
                            "value": "313.225",
                            "footnotes": [{}],
                        },
                        {
                            "year": "2024",
                            "period": "M04",
                            "periodName": "April",
                            "value": "313.207",
                            "footnotes": [{}],
                        },
                    ],
                }
            ]
        },
    }


@pytest.fixture
def sample_polygon_response() -> dict:
    """Mock Polygon.io aggregates response."""
    base_ts = int(datetime(2024, 1, 15, 9, 30).timestamp() * 1000)
    return {
        "ticker": "SPY",
        "queryCount": 3,
        "resultsCount": 3,
        "adjusted": True,
        "results": [
            {
                "v": 1500000,
                "vw": 475.50,
                "o": 475.00,
                "c": 475.80,
                "h": 476.00,
                "l": 474.50,
                "t": base_ts,
                "n": 5000,
            },
            {
                "v": 1800000,
                "vw": 476.20,
                "o": 475.80,
                "c": 476.50,
                "h": 477.00,
                "l": 475.60,
                "t": base_ts + 60000,
                "n": 6000,
            },
            {
                "v": 2000000,
                "vw": 476.80,
                "o": 476.50,
                "c": 477.20,
                "h": 477.50,
                "l": 476.30,
                "t": base_ts + 120000,
                "n": 7000,
            },
        ],
    }


@pytest.fixture
def sample_te_calendar_response() -> list:
    """Mock Trading Economics calendar response."""
    return [
        {
            "Date": "2024-07-11T08:30:00",
            "Country": "United States",
            "Category": "CPI",
            "Actual": 3.0,
            "Forecast": 3.1,
            "Previous": 3.3,
            "Importance": 3,
            "Currency": "USD",
        },
        {
            "Date": "2024-07-05T08:30:00",
            "Country": "United States",
            "Category": "Non Farm Payrolls",
            "Actual": 206.0,
            "Forecast": 190.0,
            "Previous": 218.0,
            "Importance": 3,
            "Currency": "USD",
        },
    ]


@pytest.fixture
def sample_economic_release() -> EconomicRelease:
    """A single sample economic release for testing."""
    return EconomicRelease(
        indicator=EconomicIndicator.CPI,
        release_date=datetime(2024, 7, 11, 8, 30),
        actual=3.0,
        consensus=3.1,
        previous=3.3,
        source="FRED",
    )


@pytest.fixture
def sample_market_bars() -> list[MarketBar]:
    """Sample OHLCV bars for testing."""
    base_time = datetime(2024, 7, 11, 8, 30)
    return [
        MarketBar(
            ticker="SPY",
            timestamp=base_time + timedelta(minutes=i),
            open=475.0 + i * 0.1,
            high=475.5 + i * 0.1,
            low=474.5 + i * 0.1,
            close=475.3 + i * 0.1,
            volume=100000 + i * 10000,
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_nowcast_estimate() -> NowcastEstimate:
    """A sample nowcast estimate for testing."""
    return NowcastEstimate(
        indicator=EconomicIndicator.CPI,
        target_release_date=datetime(2024, 8, 14, 8, 30),
        estimated_at=datetime(2024, 8, 13, 12, 0),
        point_estimate=2.9,
        confidence_lower=2.7,
        confidence_upper=3.1,
        model_name="cpi_bridge_v1",
        features_used={"gasoline_prices": 3.45, "shelter_cpi": 5.2},
    )
