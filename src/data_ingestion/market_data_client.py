"""
Market data client using Polygon.io REST API.

API Docs: https://polygon.io/docs
Covers: OHLCV bars (stocks, ETFs, indices), snapshots.
Free tier: 5 API calls/min, 2 years historical data.

Primary use: fetch price data around macro event releases
to measure market reaction (e.g., SPY, TLT, DX around CPI release).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import structlog

from src.data_ingestion.base_client import BaseClient
from src.data_ingestion.schemas import MarketBar, MarketSnapshot, Timeframe

logger = structlog.get_logger(__name__)

_POLYGON_BASE_URL = "https://api.polygon.io"

# Timeframe mapping to Polygon API format
_TIMEFRAME_MAP = {
    Timeframe.MINUTE_1: ("1", "minute"),
    Timeframe.MINUTE_5: ("5", "minute"),
    Timeframe.MINUTE_15: ("15", "minute"),
    Timeframe.HOUR_1: ("1", "hour"),
    Timeframe.DAY_1: ("1", "day"),
    Timeframe.WEEK_1: ("1", "week"),
    Timeframe.MONTH_1: ("1", "month"),
}


class MarketDataClient(BaseClient):
    """
    Async client for Polygon.io market data API.

    Provides OHLCV bars and snapshots for equities and ETFs.
    Key tickers for macro analysis: SPY, QQQ, TLT, GLD, UUP.
    """

    def __init__(self, api_key: str, rate_limit: float = 0.08):
        """
        Args:
            api_key: Polygon.io API key.
            rate_limit: Requests per second (free tier = 5/min ≈ 0.08/sec).
        """
        super().__init__(
            base_url=_POLYGON_BASE_URL,
            api_key=api_key,
            rate_limit=rate_limit,
            max_retries=3,
            base_delay=2.0,
            timeout=30.0,
        )

    def _get_default_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key or ''}",
        }

    def _get_auth_params(self) -> dict[str, str]:
        # Polygon uses Authorization header, not query params
        return {}

    # -----------------------------------------------------------------
    # Public Methods
    # -----------------------------------------------------------------

    async def get_bars(
        self,
        ticker: str,
        timeframe: Timeframe = Timeframe.DAY_1,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a ticker.

        Args:
            ticker: Ticker symbol (e.g., "SPY").
            timeframe: Bar timeframe (1m, 5m, 1h, 1d, etc.).
            start: Start datetime.
            end: End datetime.
            limit: Max results per request (max 50000).

        Returns:
            DataFrame with columns: [open, high, low, close, volume, vwap],
            indexed by timestamp.
        """
        multiplier, span = _TIMEFRAME_MAP.get(timeframe, ("1", "day"))

        start_str = (start or datetime(2020, 1, 1)).strftime("%Y-%m-%d")
        end_str = (end or datetime.now()).strftime("%Y-%m-%d")

        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{span}/{start_str}/{end_str}"
        params: dict[str, str | int] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": limit,
        }

        data = await self._request("GET", path, params=params)

        results = data.get("results", [])
        if not results:
            logger.warning("polygon_no_bars", ticker=ticker, timeframe=timeframe.value)
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap"])

        bars = []
        for r in results:
            bars.append(
                MarketBar(
                    ticker=ticker,
                    timestamp=datetime.fromtimestamp(r["t"] / 1000),
                    open=r["o"],
                    high=r["h"],
                    low=r["l"],
                    close=r["c"],
                    volume=r.get("v", 0),
                    vwap=r.get("vw"),
                    num_trades=r.get("n"),
                )
            )

        df = pd.DataFrame([b.model_dump() for b in bars])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        logger.info(
            "polygon_bars_fetched",
            ticker=ticker,
            timeframe=timeframe.value,
            bars=len(df),
        )
        return df

    async def get_bars_around_event(
        self,
        ticker: str,
        event_time: datetime,
        window_minutes: int = 60,
        timeframe: Timeframe = Timeframe.MINUTE_1,
    ) -> pd.DataFrame:
        """
        Fetch minute-level bars around a macro event release.

        This is the core method for measuring market reaction to data releases.

        Args:
            ticker: Ticker symbol.
            event_time: Exact time of the data release.
            window_minutes: Minutes before AND after the event to fetch.
            timeframe: Bar granularity (default: 1-minute).

        Returns:
            DataFrame of bars centered around the event time.
        """
        start = event_time - timedelta(minutes=window_minutes)
        end = event_time + timedelta(minutes=window_minutes)

        df = await self.get_bars(ticker, timeframe=timeframe, start=start, end=end)

        if not df.empty:
            logger.info(
                "polygon_event_bars",
                ticker=ticker,
                event_time=str(event_time),
                window=window_minutes,
                bars=len(df),
            )
        return df

    async def get_snapshot(self, ticker: str) -> MarketSnapshot | None:
        """
        Get the latest snapshot (quote) for a ticker.

        Args:
            ticker: Ticker symbol.

        Returns:
            MarketSnapshot or None if unavailable.
        """
        path = f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"

        try:
            data = await self._request("GET", path)
        except Exception:
            logger.warning("polygon_snapshot_failed", ticker=ticker)
            return None

        ticker_data = data.get("ticker", {})
        if not ticker_data:
            return None

        day = ticker_data.get("day", {})
        prev_day = ticker_data.get("prevDay", {})

        current_close = day.get("c", 0)
        prev_close = prev_day.get("c", 0)
        change_pct = ((current_close - prev_close) / prev_close * 100) if prev_close else None

        return MarketSnapshot(
            ticker=ticker,
            timestamp=datetime.fromtimestamp(
                ticker_data.get("updated", 0) / 1e9
            )
            if ticker_data.get("updated")
            else datetime.now(),
            last_price=current_close,
            volume=day.get("v", 0),
            change_pct=change_pct,
        )
