"""
Tests for the base client's retry logic, rate limiting, and error classification.

These tests verify the core infrastructure that all API clients depend on.
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import respx

from src.data_ingestion.base_client import (
    APIError,
    AuthenticationError,
    BaseClient,
    DataNotFoundError,
    RateLimitError,
    ServerError,
    TokenBucketRateLimiter,
)


# ---------------------------------------------------------------------------
# Concrete implementation for testing the abstract BaseClient
# ---------------------------------------------------------------------------


class _TestClient(BaseClient):
    """Minimal concrete implementation for testing BaseClient."""

    def __init__(self, **kwargs):
        super().__init__(base_url="https://test.api.com", **kwargs)

    def _get_default_headers(self) -> dict[str, str]:
        return {"Accept": "application/json"}

    def _get_auth_params(self) -> dict[str, str]:
        if self._api_key:
            return {"api_key": self._api_key}
        return {}


# ---------------------------------------------------------------------------
# Test: TokenBucketRateLimiter
# ---------------------------------------------------------------------------


class TestTokenBucketRateLimiter:
    """Tests for the token bucket rate limiter."""

    async def test_initial_capacity(self):
        """Rate limiter should start with full capacity."""
        rl = TokenBucketRateLimiter(rate=10.0, capacity=5)
        # Should be able to acquire 5 tokens immediately
        for _ in range(5):
            await rl.acquire()

    async def test_refill_over_time(self):
        """Tokens should refill after waiting."""
        rl = TokenBucketRateLimiter(rate=100.0, capacity=1)
        # Drain the single token
        await rl.acquire()
        # Wait for refill
        await asyncio.sleep(0.02)
        # Should be able to acquire again
        await rl.acquire()

    async def test_default_capacity(self):
        """Default capacity should be max(1, int(rate * 2))."""
        rl = TokenBucketRateLimiter(rate=5.0)
        assert rl.capacity == 10

        rl_small = TokenBucketRateLimiter(rate=0.1)
        assert rl_small.capacity == 1


# ---------------------------------------------------------------------------
# Test: Error Classification
# ---------------------------------------------------------------------------


class TestErrorClassification:
    """Tests for HTTP error code classification."""

    @respx.mock
    async def test_401_raises_auth_error(self):
        """HTTP 401 should raise AuthenticationError."""
        client = _TestClient(rate_limit=100.0)
        respx.get("https://test.api.com/endpoint").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )
        with pytest.raises(AuthenticationError):
            await client._request("GET", "/endpoint")

    @respx.mock
    async def test_403_raises_auth_error(self):
        """HTTP 403 should raise AuthenticationError."""
        client = _TestClient(rate_limit=100.0)
        respx.get("https://test.api.com/endpoint").mock(
            return_value=httpx.Response(403, text="Forbidden")
        )
        with pytest.raises(AuthenticationError):
            await client._request("GET", "/endpoint")

    @respx.mock
    async def test_404_raises_not_found(self):
        """HTTP 404 should raise DataNotFoundError."""
        client = _TestClient(rate_limit=100.0)
        respx.get("https://test.api.com/endpoint").mock(
            return_value=httpx.Response(404, text="Not Found")
        )
        with pytest.raises(DataNotFoundError):
            await client._request("GET", "/endpoint")

    @respx.mock
    async def test_400_raises_generic_api_error(self):
        """HTTP 400 should raise generic APIError."""
        client = _TestClient(rate_limit=100.0)
        respx.get("https://test.api.com/endpoint").mock(
            return_value=httpx.Response(400, text="Bad Request")
        )
        with pytest.raises(APIError):
            await client._request("GET", "/endpoint")


# ---------------------------------------------------------------------------
# Test: Retry Logic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """Tests for exponential backoff retry behavior."""

    @respx.mock
    async def test_retry_on_429(self):
        """HTTP 429 should trigger retries."""
        client = _TestClient(rate_limit=100.0, max_retries=2, base_delay=0.01)
        route = respx.get("https://test.api.com/endpoint")
        route.side_effect = [
            httpx.Response(429, text="Rate limited"),
            httpx.Response(200, json={"data": "ok"}),
        ]

        result = await client._request("GET", "/endpoint")
        assert result == {"data": "ok"}
        assert route.call_count == 2

    @respx.mock
    async def test_retry_on_500(self):
        """HTTP 500 should trigger retries."""
        client = _TestClient(rate_limit=100.0, max_retries=2, base_delay=0.01)
        route = respx.get("https://test.api.com/endpoint")
        route.side_effect = [
            httpx.Response(500, text="Internal Server Error"),
            httpx.Response(200, json={"data": "ok"}),
        ]

        result = await client._request("GET", "/endpoint")
        assert result == {"data": "ok"}
        assert route.call_count == 2

    @respx.mock
    async def test_retries_exhausted_raises(self):
        """After all retries exhausted, should raise ServerError."""
        client = _TestClient(rate_limit=100.0, max_retries=2, base_delay=0.01)
        respx.get("https://test.api.com/endpoint").mock(
            return_value=httpx.Response(500, text="Server Error")
        )

        with pytest.raises(ServerError):
            await client._request("GET", "/endpoint")

    @respx.mock
    async def test_no_retry_on_auth_error(self):
        """Non-retryable errors (401) should NOT be retried."""
        client = _TestClient(rate_limit=100.0, max_retries=3, base_delay=0.01)
        route = respx.get("https://test.api.com/endpoint").mock(
            return_value=httpx.Response(401, text="Unauthorized")
        )

        with pytest.raises(AuthenticationError):
            await client._request("GET", "/endpoint")

        # Should only be called once (no retries)
        assert route.call_count == 1


# ---------------------------------------------------------------------------
# Test: Auth Params
# ---------------------------------------------------------------------------


class TestAuthParams:
    """Tests for authentication parameter injection."""

    @respx.mock
    async def test_api_key_in_query_params(self):
        """API key should be added to query parameters."""
        client = _TestClient(api_key="my_secret_key", rate_limit=100.0)
        route = respx.get("https://test.api.com/endpoint").mock(
            return_value=httpx.Response(200, json={"data": "ok"})
        )

        await client._request("GET", "/endpoint")

        assert route.calls[0].request.url.params["api_key"] == "my_secret_key"

    @respx.mock
    async def test_no_api_key_when_none(self):
        """No api_key param should be added when key is None."""
        client = _TestClient(api_key=None, rate_limit=100.0)
        route = respx.get("https://test.api.com/endpoint").mock(
            return_value=httpx.Response(200, json={"data": "ok"})
        )

        await client._request("GET", "/endpoint")

        assert "api_key" not in dict(route.calls[0].request.url.params)


# ---------------------------------------------------------------------------
# Test: Context Manager
# ---------------------------------------------------------------------------


class TestContextManager:
    """Tests for async context manager protocol."""

    async def test_context_manager_closes_client(self):
        """Exiting context manager should close the httpx client."""
        async with _TestClient(rate_limit=100.0) as client:
            assert client._client is not None

        # After context exit, client should be closed
        assert client._client.is_closed
