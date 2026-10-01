"""
Bureau of Labor Statistics (BLS) API v2 client.

API Docs: https://www.bls.gov/developers/api_signature_v2.htm
Covers: CPI (CUSR0000SA0), NFP (CES0000000001), Unemployment, PPI.
Free tier: 500 requests/day with API key, 25/day without.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import structlog

from src.data_ingestion.base_client import BaseClient
from src.data_ingestion.schemas import BLSDataPoint, BLSSeriesData

logger = structlog.get_logger(__name__)

_BLS_BASE_URL = "https://api.bls.gov/publicAPI/v2"


class BLSClient(BaseClient):
    """
    Async client for the BLS API v2.

    Note: BLS API v2 uses POST requests with JSON body
    (unlike most REST APIs that use GET with query params).
    """

    def __init__(self, api_key: str | None = None, rate_limit: float = 1.0):
        super().__init__(
            base_url=_BLS_BASE_URL,
            api_key=api_key,
            rate_limit=rate_limit,
            max_retries=3,
            base_delay=2.0,  # BLS is slower, use longer backoff
            timeout=45.0,
        )

    def _get_default_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get_auth_params(self) -> dict[str, str]:
        # BLS uses registration key in the POST body, not query params
        return {}

    # -----------------------------------------------------------------
    # Public Methods
    # -----------------------------------------------------------------

    async def get_series(
        self,
        series_ids: list[str],
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> dict[str, BLSSeriesData]:
        """
        Fetch data for one or more BLS series.

        BLS API v2 supports up to 50 series per request.

        Args:
            series_ids: List of BLS series IDs (max 50).
            start_year: Start year (e.g., 2020).
            end_year: End year (e.g., 2024).

        Returns:
            Dict mapping series_id -> BLSSeriesData.
        """
        if len(series_ids) > 50:
            raise ValueError("BLS API supports max 50 series per request")

        # Build POST body
        payload: dict = {"seriesid": series_ids}
        if start_year:
            payload["startyear"] = str(start_year)
        if end_year:
            payload["endyear"] = str(end_year)
        if self._api_key:
            payload["registrationkey"] = self._api_key

        data = await self._request_post("/timeseries/data/", payload)

        results: dict[str, BLSSeriesData] = {}

        if data.get("status") != "REQUEST_SUCCEEDED":
            logger.error(
                "bls_request_failed",
                status=data.get("status"),
                message=data.get("message", []),
            )
            return results

        for series in data.get("Results", {}).get("series", []):
            sid = series["seriesID"]
            data_points = []
            for dp in series.get("data", []):
                try:
                    value = float(dp["value"].replace(",", ""))
                except (ValueError, AttributeError):
                    continue

                footnotes = [
                    fn.get("text", "") for fn in dp.get("footnotes", []) if fn.get("text")
                ]
                data_points.append(
                    BLSDataPoint(
                        year=int(dp["year"]),
                        period=dp["period"],
                        period_name=dp.get("periodName", ""),
                        value=value,
                        footnotes=footnotes,
                    )
                )

            results[sid] = BLSSeriesData(series_id=sid, data=data_points)
            logger.info(
                "bls_series_fetched",
                series_id=sid,
                data_points=len(data_points),
            )

        return results

    async def get_series_dataframe(
        self,
        series_id: str,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> pd.DataFrame:
        """
        Convenience method: fetch a single series as a DataFrame.

        Returns:
            DataFrame with columns: ["date", "value"], indexed by date.
        """
        result = await self.get_series([series_id], start_year, end_year)

        if series_id not in result or not result[series_id].data:
            return pd.DataFrame(columns=["date", "value"])

        rows = [
            {"date": dp.date, "value": dp.value} for dp in result[series_id].data
        ]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    async def get_latest_release(self, series_id: str) -> BLSDataPoint | None:
        """
        Get the most recent data point for a series.

        Uses the current year and previous year to ensure
        we capture the latest available data.
        """
        current_year = datetime.now().year
        result = await self.get_series(
            [series_id],
            start_year=current_year - 1,
            end_year=current_year,
        )

        if series_id not in result or not result[series_id].data:
            return None

        # BLS returns data in reverse chronological order
        return result[series_id].data[0]

    # -----------------------------------------------------------------
    # Internal: BLS uses POST, not GET
    # -----------------------------------------------------------------

    async def _request_post(self, path: str, payload: dict) -> dict:
        """
        BLS API uses POST with JSON body instead of GET with query params.
        We override the base _request flow here.
        """
        # Remove the registrationkey from payload before logging
        log_payload = {k: v for k, v in payload.items() if k != "registrationkey"}
        self._log.debug("bls_post_request", path=path, payload=log_payload)

        # Use the base _request method but pass the body as json
        # We need to bypass _get_auth_params since BLS auth goes in the body
        await self._rate_limiter.acquire()

        response = await self._client.post(path, json=payload)

        if response.status_code == 200:
            return response.json()

        error = self._classify_error(response)
        raise error
