"""
FRED (Federal Reserve Economic Data) API client.

API Docs: https://fred.stlouisfed.org/docs/api/fred/
Covers: CPI (CPIAUCSL), NFP (PAYEMS), GDP, PCE, Unemployment Rate, etc.
Free tier: unlimited requests with API key, ~120 req/min soft limit.
"""

from __future__ import annotations

from datetime import datetime
import re

import pandas as pd
import structlog

from src.data_ingestion.base_client import BaseClient
from src.data_ingestion.schemas import FREDObservation, FREDReleaseDate, FREDSeriesInfo

logger = structlog.get_logger(__name__)

_FRED_BASE_URL = "https://api.stlouisfed.org"


class FREDClient(BaseClient):
    """Async client for the FRED API."""

    def __init__(self, api_key: str, rate_limit: float = 2.0):
        super().__init__(
            base_url=_FRED_BASE_URL,
            api_key=api_key,
            rate_limit=rate_limit,
            max_retries=3,
            base_delay=1.0,
            timeout=30.0,
        )

    def _get_default_headers(self) -> dict[str, str]:
        return {"Accept": "application/json"}

    def _get_auth_params(self) -> dict[str, str]:
        return {"api_key": self._api_key or "", "file_type": "json"}

    # -----------------------------------------------------------------
    # Public Methods
    # -----------------------------------------------------------------

    async def get_series(
        self,
        series_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        frequency: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch observation data for a FRED series.

        Args:
            series_id: FRED series ID (e.g., "CPIAUCSL").
            start_date: Start of date range (inclusive).
            end_date: End of date range (inclusive).
            frequency: Aggregation frequency (d, w, bw, m, q, sa, a).

        Returns:
            DataFrame with columns: ["date", "value"], indexed by date.
        """
        params: dict[str, str] = {"series_id": series_id}
        if start_date:
            params["observation_start"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["observation_end"] = end_date.strftime("%Y-%m-%d")
        if frequency:
            params["frequency"] = frequency

        data = await self._request("GET", "/fred/series/observations", params=params)

        observations = []
        for obs in data.get("observations", []):
            # FRED returns "." for missing/unavailable values
            raw_value = obs.get("value", ".")
            value = float(raw_value) if raw_value != "." else None
            observations.append(
                FREDObservation(
                    date=datetime.strptime(obs["date"], "%Y-%m-%d"),
                    value=value,
                )
            )

        if not observations:
            logger.warning("fred_empty_series", series_id=series_id)
            return pd.DataFrame(columns=["date", "value"])

        df = pd.DataFrame([o.model_dump() for o in observations])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        logger.info(
            "fred_series_fetched",
            series_id=series_id,
            rows=len(df),
            start=str(df.index.min()),
            end=str(df.index.max()),
        )
        return df

    async def get_series_info(self, series_id: str) -> FREDSeriesInfo:
        """
        Fetch metadata about a FRED series.

        Args:
            series_id: FRED series ID.

        Returns:
            FREDSeriesInfo with title, frequency, units, etc.
        """
        data = await self._request(
            "GET", "/fred/series", params={"series_id": series_id}
        )

        serieses = data.get("serieses", [])
        if not serieses:
            raise ValueError(f"No series found for {series_id}")

        s = serieses[0]
        return FREDSeriesInfo(
            series_id=s["id"],
            title=s["title"],
            frequency=s.get("frequency", ""),
            units=s.get("units", ""),
            seasonal_adjustment=s.get("seasonal_adjustment", ""),
            last_updated=self._parse_fred_datetime(s.get("last_updated")),
        )

    async def get_release_dates(
        self,
        series_id: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[FREDReleaseDate]:
        """
        Fetch release dates for a given series.
        Useful for building the economic calendar from FRED data.

        Args:
            series_id: FRED series ID.
            start_date: Filter releases after this date.
            end_date: Filter releases before this date.

        Returns:
            List of FREDReleaseDate objects.
        """
        # First get the release ID for this series
        release_data = await self._request(
            "GET", "/fred/series/release", params={"series_id": series_id}
        )
        releases = release_data.get("releases", [])
        if not releases:
            return []

        release_id = releases[0]["id"]
        release_name = releases[0]["name"]

        # Then get dates for that release
        params: dict[str, str | int] = {"release_id": release_id}
        if start_date:
            params["realtime_start"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["realtime_end"] = end_date.strftime("%Y-%m-%d")

        dates_data = await self._request(
            "GET", "/fred/release/dates", params=params
        )

        results = []
        for rd in dates_data.get("release_dates", []):
            results.append(
                FREDReleaseDate(
                    release_id=release_id,
                    release_name=release_name,
                    date=datetime.strptime(rd["date"], "%Y-%m-%d"),
                )
            )

        return results

    async def search_series(
        self, query: str, limit: int = 20
    ) -> list[FREDSeriesInfo]:
        """
        Search FRED for series matching a query string.

        Args:
            query: Search text.
            limit: Max results to return.

        Returns:
            List of matching FREDSeriesInfo.
        """
        data = await self._request(
            "GET",
            "/fred/series/search",
            params={"search_text": query, "limit": limit},
        )

        results = []
        for s in data.get("serieses", []):
            results.append(
                FREDSeriesInfo(
                    series_id=s["id"],
                    title=s["title"],
                    frequency=s.get("frequency", ""),
                    units=s.get("units", ""),
                    seasonal_adjustment=s.get("seasonal_adjustment", ""),
                    last_updated=self._parse_fred_datetime(s.get("last_updated")),
                )
            )
        return results

    @staticmethod
    def _parse_fred_datetime(dt_str: str | None) -> datetime | None:
        """
        Parse FRED datetime strings which can have non-standard timezone offsets.
        Examples: '2024-07-11 07:36:03-05', '2024-07-11 07:36:03-05:00'
        """
        if not dt_str:
            return None
        try:
            # Normalize timezone: '-05' -> '-05:00'
            normalized = re.sub(r'([+-]\d{2})$', r'\g<1>:00', dt_str)
            return datetime.fromisoformat(normalized)
        except (ValueError, AttributeError):
            try:
                return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S%z")
            except ValueError:
                return None
