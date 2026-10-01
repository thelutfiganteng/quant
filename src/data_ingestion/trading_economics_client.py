"""
Trading Economics API client.

API Docs: https://docs.tradingeconomics.com/
Covers: Economic calendar, consensus estimates, historical indicators.
Free tier: 50 requests/day — quite limited.

If no API key is configured, the client enters "fallback mode" and returns
empty results with a warning log. This allows the system to run without
Trading Economics for development/testing.
"""

from __future__ import annotations

from datetime import datetime

import structlog

from src.data_ingestion.base_client import BaseClient
from src.data_ingestion.schemas import CalendarEvent, ConsensusData, Importance

logger = structlog.get_logger(__name__)

_TE_BASE_URL = "https://api.tradingeconomics.com"


class TradingEconomicsClient(BaseClient):
    """
    Async client for the Trading Economics API.

    Gracefully degrades when API key is not available.
    """

    def __init__(self, api_key: str | None = None, rate_limit: float = 1.0):
        self._fallback_mode = not api_key
        super().__init__(
            base_url=_TE_BASE_URL,
            api_key=api_key,
            rate_limit=rate_limit,
            max_retries=2,
            base_delay=2.0,
            timeout=30.0,
        )
        if self._fallback_mode:
            logger.warning(
                "trading_economics_fallback_mode",
                message="No API key configured — calendar and consensus data will be unavailable",
            )

    def _get_default_headers(self) -> dict[str, str]:
        return {"Accept": "application/json"}

    def _get_auth_params(self) -> dict[str, str]:
        if self._api_key:
            return {"c": self._api_key}
        return {}

    # -----------------------------------------------------------------
    # Public Methods
    # -----------------------------------------------------------------

    async def get_calendar(
        self,
        country: str = "united states",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        importance: int | None = None,
    ) -> list[CalendarEvent]:
        """
        Fetch economic calendar events.

        Args:
            country: Country name (e.g., "united states").
            start_date: Start of date range.
            end_date: End of date range.
            importance: Filter by importance (1=Low, 2=Medium, 3=High).

        Returns:
            List of CalendarEvent objects. Empty list in fallback mode.
        """
        if self._fallback_mode:
            logger.info("trading_economics_calendar_skipped", reason="fallback_mode")
            return []

        # Build path
        path = f"/calendar/country/{country}"
        if start_date and end_date:
            path += f"/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

        params: dict[str, str | int] = {}
        if importance is not None:
            params["importance"] = importance

        data = await self._request("GET", path, params=params)

        events = []
        for item in data if isinstance(data, list) else []:
            imp = self._map_importance(item.get("Importance", 2))
            events.append(
                CalendarEvent(
                    date=self._parse_te_date(item.get("Date", "")),
                    country=item.get("Country", country),
                    indicator=item.get("Category", ""),
                    actual=self._safe_float(item.get("Actual")),
                    forecast=self._safe_float(item.get("Forecast")),
                    previous=self._safe_float(item.get("Previous")),
                    importance=imp,
                    currency=item.get("Currency", "USD"),
                    source="TradingEconomics",
                )
            )

        logger.info(
            "trading_economics_calendar_fetched",
            country=country,
            events=len(events),
        )
        return events

    async def get_historical_indicator(
        self,
        country: str = "united states",
        indicator: str = "inflation rate",
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict]:
        """
        Fetch historical values for a specific indicator.

        Args:
            country: Country name.
            indicator: Indicator name (e.g., "inflation rate", "non farm payrolls").
            start_date: Start date.
            end_date: End date.

        Returns:
            List of dicts with date/value pairs. Empty in fallback mode.
        """
        if self._fallback_mode:
            return []

        path = f"/historical/country/{country}/indicator/{indicator}"
        params: dict[str, str] = {}
        if start_date:
            params["d1"] = start_date.strftime("%Y-%m-%d")
        if end_date:
            params["d2"] = end_date.strftime("%Y-%m-%d")

        data = await self._request("GET", path, params=params)
        return data if isinstance(data, list) else []

    async def get_consensus(
        self, country: str = "united states", indicator: str | None = None
    ) -> list[ConsensusData]:
        """
        Fetch consensus forecasts for upcoming releases.

        Args:
            country: Country name.
            indicator: Optional specific indicator filter.

        Returns:
            List of ConsensusData. Empty in fallback mode.
        """
        if self._fallback_mode:
            return []

        path = f"/forecast/country/{country}"
        if indicator:
            path += f"/indicator/{indicator}"

        data = await self._request("GET", path)

        results = []
        for item in data if isinstance(data, list) else []:
            forecast_val = self._safe_float(item.get("ForecastValue"))
            if forecast_val is None:
                continue

            results.append(
                ConsensusData(
                    indicator=item.get("Category", ""),
                    release_date=self._parse_te_date(item.get("Date", "")),
                    consensus=forecast_val,
                    high_estimate=self._safe_float(item.get("ForecastValue1")),
                    low_estimate=self._safe_float(item.get("ForecastValue2")),
                    previous=self._safe_float(item.get("LatestValue")),
                )
            )

        return results

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_te_date(date_str: str) -> datetime:
        """Parse Trading Economics date format."""
        if not date_str:
            return datetime.now()
        try:
            # TE format: "2024-01-10T13:30:00"
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return datetime.now()

    @staticmethod
    def _safe_float(value: object) -> float | None:
        """Safely convert a value to float, returning None on failure."""
        if value is None or value == "":
            return None
        try:
            return float(value)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _map_importance(value: int | str) -> Importance:
        """Map Trading Economics importance levels to our enum."""
        try:
            level = int(value)
        except (ValueError, TypeError):
            return Importance.MEDIUM
        if level >= 3:
            return Importance.HIGH
        elif level == 2:
            return Importance.MEDIUM
        return Importance.LOW
