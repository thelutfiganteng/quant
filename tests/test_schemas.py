"""
Tests for Pydantic schemas and data validation.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.data_ingestion.schemas import (
    BLSDataPoint,
    CalendarEvent,
    EconomicIndicator,
    EconomicRelease,
    FREDObservation,
    Importance,
    MarketBar,
    NowcastEstimate,
    FRED_SERIES,
    BLS_SERIES,
)


class TestEconomicIndicator:
    """Tests for the EconomicIndicator enum."""

    def test_all_values_are_strings(self):
        for indicator in EconomicIndicator:
            assert isinstance(indicator.value, str)

    def test_common_indicators_exist(self):
        assert EconomicIndicator.CPI
        assert EconomicIndicator.NFP
        assert EconomicIndicator.GDP
        assert EconomicIndicator.UNEMPLOYMENT_RATE


class TestFREDObservation:
    """Tests for FREDObservation schema."""

    def test_valid_observation(self):
        obs = FREDObservation(date=datetime(2024, 1, 1), value=308.417)
        assert obs.value == pytest.approx(308.417)

    def test_missing_value(self):
        obs = FREDObservation(date=datetime(2024, 1, 1), value=None)
        assert obs.value is None

    def test_frozen_model(self):
        obs = FREDObservation(date=datetime(2024, 1, 1), value=100.0)
        with pytest.raises(Exception):
            obs.value = 200.0


class TestBLSDataPoint:
    """Tests for BLSDataPoint schema."""

    def test_date_conversion(self):
        dp = BLSDataPoint(year=2024, period="M06", period_name="June", value=313.049)
        assert dp.date == datetime(2024, 6, 1)

    def test_annual_period(self):
        dp = BLSDataPoint(year=2024, period="A01", period_name="Annual", value=100.0)
        assert dp.date == datetime(2024, 1, 1)


class TestCalendarEvent:
    """Tests for CalendarEvent schema."""

    def test_surprise_calculation(self):
        event = CalendarEvent(
            date=datetime(2024, 7, 11),
            indicator="CPI",
            actual=3.0,
            forecast=3.1,
        )
        assert event.surprise == pytest.approx(-0.1)

    def test_surprise_with_missing_values(self):
        event = CalendarEvent(
            date=datetime(2024, 7, 11),
            indicator="CPI",
            actual=3.0,
            forecast=None,
        )
        assert event.surprise is None

    def test_default_importance(self):
        event = CalendarEvent(date=datetime(2024, 7, 11), indicator="CPI")
        assert event.importance == Importance.MEDIUM


class TestEconomicRelease:
    """Tests for EconomicRelease schema."""

    def test_compute_surprise(self):
        release = EconomicRelease(
            indicator=EconomicIndicator.CPI,
            release_date=datetime(2024, 7, 11),
            actual=3.0,
            consensus=3.1,
        )
        assert release.compute_surprise() == pytest.approx(-0.1)

    def test_compute_surprise_missing(self):
        release = EconomicRelease(
            indicator=EconomicIndicator.CPI,
            release_date=datetime(2024, 7, 11),
            actual=None,
            consensus=3.1,
        )
        assert release.compute_surprise() is None


class TestMarketBar:
    """Tests for MarketBar schema."""

    def test_valid_bar(self):
        bar = MarketBar(
            ticker="SPY",
            timestamp=datetime(2024, 1, 1, 9, 30),
            open=475.0,
            high=476.0,
            low=474.0,
            close=475.5,
            volume=1000000,
        )
        assert bar.ticker == "SPY"
        assert bar.volume == 1000000


class TestSeriesConstants:
    """Tests for FRED and BLS series ID mappings."""

    def test_fred_series_has_cpi(self):
        assert EconomicIndicator.CPI in FRED_SERIES
        assert FRED_SERIES[EconomicIndicator.CPI] == "CPIAUCSL"

    def test_fred_series_has_nfp(self):
        assert EconomicIndicator.NFP in FRED_SERIES
        assert FRED_SERIES[EconomicIndicator.NFP] == "PAYEMS"

    def test_bls_series_has_cpi(self):
        assert EconomicIndicator.CPI in BLS_SERIES
        assert BLS_SERIES[EconomicIndicator.CPI] == "CUSR0000SA0"
