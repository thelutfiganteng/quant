"""
Shared Pydantic v2 schemas for the data ingestion layer.

All data flowing between clients, storage, and models uses these
validated schemas — no raw dicts floating around.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EconomicIndicator(str, Enum):
    """Supported macroeconomic indicators."""

    CPI = "CPI"
    CORE_CPI = "CORE_CPI"
    NFP = "NFP"
    GDP = "GDP"
    PCE = "PCE"
    CORE_PCE = "CORE_PCE"
    UNEMPLOYMENT_RATE = "UNEMPLOYMENT_RATE"
    INITIAL_CLAIMS = "INITIAL_CLAIMS"
    RETAIL_SALES = "RETAIL_SALES"
    ISM_MANUFACTURING = "ISM_MANUFACTURING"
    ISM_SERVICES = "ISM_SERVICES"
    PPI = "PPI"
    HOUSING_STARTS = "HOUSING_STARTS"
    CONSUMER_CONFIDENCE = "CONSUMER_CONFIDENCE"
    DURABLE_GOODS = "DURABLE_GOODS"


class Importance(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class Timeframe(str, Enum):
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1mo"


# ---------------------------------------------------------------------------
# FRED Schemas
# ---------------------------------------------------------------------------


class FREDObservation(BaseModel):
    """Single observation from a FRED time series."""

    model_config = ConfigDict(frozen=True)

    date: datetime
    value: float | None = None  # FRED returns "." for missing values


class FREDSeriesInfo(BaseModel):
    """Metadata about a FRED series."""

    series_id: str
    title: str
    frequency: str
    units: str
    seasonal_adjustment: str
    last_updated: datetime | None = None


class FREDReleaseDate(BaseModel):
    """A release date entry from FRED."""

    release_id: int
    release_name: str
    date: datetime


# ---------------------------------------------------------------------------
# BLS Schemas
# ---------------------------------------------------------------------------


class BLSDataPoint(BaseModel):
    """Single data point from BLS API."""

    year: int
    period: str  # e.g., "M01" for January
    period_name: str  # e.g., "January"
    value: float
    footnotes: list[str] = Field(default_factory=list)

    @property
    def date(self) -> datetime:
        """Convert BLS period format to datetime."""
        # BLS monthly periods: M01-M12, annual: A01
        if self.period.startswith("M"):
            month = int(self.period[1:])
            return datetime(self.year, month, 1)
        return datetime(self.year, 1, 1)


class BLSSeriesData(BaseModel):
    """Response from BLS API for a single series."""

    series_id: str
    data: list[BLSDataPoint] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Trading Economics / Calendar Schemas
# ---------------------------------------------------------------------------


class CalendarEvent(BaseModel):
    """Economic calendar event with consensus and actual values."""

    date: datetime
    country: str = "US"
    indicator: str
    actual: float | None = None
    forecast: float | None = None
    previous: float | None = None
    importance: Importance = Importance.MEDIUM
    currency: str = "USD"
    source: str = ""

    @property
    def surprise(self) -> float | None:
        """Raw surprise: actual - forecast."""
        if self.actual is not None and self.forecast is not None:
            return self.actual - self.forecast
        return None


class ConsensusData(BaseModel):
    """Consensus estimates for an upcoming release."""

    indicator: str
    release_date: datetime
    consensus: float
    high_estimate: float | None = None
    low_estimate: float | None = None
    num_estimates: int | None = None
    previous: float | None = None


# ---------------------------------------------------------------------------
# Market Data Schemas
# ---------------------------------------------------------------------------


class MarketBar(BaseModel):
    """Single OHLCV bar."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    vwap: float | None = None
    num_trades: int | None = None


class MarketSnapshot(BaseModel):
    """Latest market quote/snapshot."""

    ticker: str
    timestamp: datetime
    last_price: float
    bid: float | None = None
    ask: float | None = None
    volume: int = 0
    change_pct: float | None = None


# ---------------------------------------------------------------------------
# Unified Release Schema (used by storage & models)
# ---------------------------------------------------------------------------


class EconomicRelease(BaseModel):
    """
    Unified representation of an economic data release.
    This is the canonical schema used across storage, nowcasting, and backtesting.
    """

    indicator: EconomicIndicator
    release_date: datetime
    actual: float | None = None
    consensus: float | None = None
    previous: float | None = None
    surprise: float | None = None
    surprise_zscore: float | None = None
    revision: float | None = None
    source: str = ""

    def compute_surprise(self) -> float | None:
        """Compute raw surprise (actual - consensus)."""
        if self.actual is not None and self.consensus is not None:
            return self.actual - self.consensus
        return None


# ---------------------------------------------------------------------------
# Nowcast Schema
# ---------------------------------------------------------------------------


class NowcastEstimate(BaseModel):
    """A single nowcast prediction for an upcoming release."""

    indicator: EconomicIndicator
    target_release_date: datetime
    estimated_at: datetime
    point_estimate: float
    confidence_lower: float | None = None
    confidence_upper: float | None = None
    model_name: str = ""
    features_used: dict | None = None


# ---------------------------------------------------------------------------
# Constants: Common FRED Series IDs
# ---------------------------------------------------------------------------

FRED_SERIES = {
    EconomicIndicator.CPI: "CPIAUCSL",
    EconomicIndicator.CORE_CPI: "CPILFESL",
    EconomicIndicator.NFP: "PAYEMS",
    EconomicIndicator.GDP: "GDP",
    EconomicIndicator.PCE: "PCE",
    EconomicIndicator.CORE_PCE: "PCEPILFE",
    EconomicIndicator.UNEMPLOYMENT_RATE: "UNRATE",
    EconomicIndicator.INITIAL_CLAIMS: "ICSA",
    EconomicIndicator.RETAIL_SALES: "RSAFS",
    EconomicIndicator.ISM_MANUFACTURING: "MANEMP",
    EconomicIndicator.PPI: "PPIACO",
}

# BLS Series IDs
BLS_SERIES = {
    EconomicIndicator.CPI: "CUSR0000SA0",
    EconomicIndicator.CORE_CPI: "CUSR0000SA0L1E",
    EconomicIndicator.NFP: "CES0000000001",
    EconomicIndicator.UNEMPLOYMENT_RATE: "LNS14000000",
    EconomicIndicator.PPI: "WPSFD4",
}
