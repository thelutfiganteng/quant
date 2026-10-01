"""
Abstract base class for all external API clients.

Provides:
- Persistent httpx.AsyncClient with connection pooling
- Token bucket rate limiting (configurable per provider)
- Exponential backoff retry with jitter
- Structured logging via structlog
- Custom exception hierarchy for clean error handling
"""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class APIError(Exception):
    """Base exception for all API errors."""

    def __init__(self, message: str, status_code: int | None = None, response_body: str = ""):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded (HTTP 429)."""

    pass


class AuthenticationError(APIError):
    """Raised when API key is invalid or missing (HTTP 401/403)."""

    pass


class DataNotFoundError(APIError):
    """Raised when requested data does not exist (HTTP 404)."""

    pass


class ServerError(APIError):
    """Raised on server-side errors (HTTP 5xx)."""

    pass


# ---------------------------------------------------------------------------
# Token Bucket Rate Limiter
# ---------------------------------------------------------------------------


class TokenBucketRateLimiter:
    """
    Simple token bucket rate limiter.

    Tokens are refilled at `rate` tokens/second up to `capacity`.
    Each request consumes one token. If no tokens are available,
    the caller awaits until a token is refilled.
    """

    def __init__(self, rate: float, capacity: int | None = None):
        """
        Args:
            rate: Tokens refilled per second (e.g., 2.0 = 2 requests/sec).
            capacity: Max burst size. Defaults to max(1, int(rate * 2)).
        """
        self.rate = rate
        self.capacity = capacity or max(1, int(rate * 2))
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time for next token
                wait = (1.0 - self._tokens) / self.rate
                await asyncio.sleep(wait)

    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now


# ---------------------------------------------------------------------------
# Base Client
# ---------------------------------------------------------------------------

# Status codes that trigger automatic retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class BaseClient(ABC):
    """
    Abstract base class for all API clients.

    Subclasses must implement:
        - _build_url(path) -> full URL
        - _parse_response(response_data) -> parsed result
        - _get_default_headers() -> dict of default headers
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        rate_limit: float = 1.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
        timeout: float = 30.0,
    ):
        """
        Args:
            base_url: Base URL for the API (e.g., "https://api.stlouisfed.org").
            api_key: API key (None if not required).
            rate_limit: Max requests per second.
            max_retries: Number of retry attempts for retryable errors.
            base_delay: Base delay in seconds for exponential backoff.
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._rate_limiter = TokenBucketRateLimiter(rate=rate_limit)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers=self._get_default_headers(),
        )
        self._log = logger.bind(client=self.__class__.__name__)

    # --- Abstract methods for subclasses ---

    @abstractmethod
    def _get_default_headers(self) -> dict[str, str]:
        """Return default headers for all requests."""
        ...

    @abstractmethod
    def _get_auth_params(self) -> dict[str, str]:
        """Return query parameters for authentication (e.g., api_key=XXX)."""
        ...

    # --- Core request method with retry + rate limiting ---

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Make an HTTP request with rate limiting and retry logic.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: URL path (appended to base_url).
            params: Query parameters.
            json_body: JSON request body (for POST).
            extra_headers: Additional headers for this request.

        Returns:
            Parsed JSON response as dict.

        Raises:
            APIError: On non-retryable errors.
            RateLimitError: If retries exhausted on 429.
            AuthenticationError: On 401/403.
            DataNotFoundError: On 404.
            ServerError: If retries exhausted on 5xx.
        """
        # Merge auth params into query params
        all_params = {**(params or {}), **self._get_auth_params()}

        headers = None
        if extra_headers:
            headers = extra_headers

        last_exception: APIError | None = None

        for attempt in range(self._max_retries + 1):
            # Wait for rate limiter
            await self._rate_limiter.acquire()

            try:
                self._log.debug(
                    "api_request",
                    method=method,
                    path=path,
                    attempt=attempt + 1,
                    params={k: v for k, v in all_params.items() if k != "api_key"},
                )

                response = await self._client.request(
                    method=method,
                    url=path,
                    params=all_params,
                    json=json_body,
                    headers=headers,
                )

                # Success
                if response.status_code == 200:
                    data = response.json()
                    self._log.debug(
                        "api_response",
                        status=200,
                        path=path,
                    )
                    return data

                # Handle specific error codes
                error = self._classify_error(response)

                # Non-retryable errors: fail immediately
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    raise error

                # Retryable error: log and retry
                last_exception = error
                delay = self._compute_backoff(attempt)
                self._log.warning(
                    "api_retry",
                    status=response.status_code,
                    path=path,
                    attempt=attempt + 1,
                    delay_seconds=round(delay, 2),
                )
                await asyncio.sleep(delay)

            except httpx.TimeoutException as exc:
                last_exception = ServerError(
                    f"Request timed out: {path}", status_code=None
                )
                delay = self._compute_backoff(attempt)
                self._log.warning(
                    "api_timeout",
                    path=path,
                    attempt=attempt + 1,
                    delay_seconds=round(delay, 2),
                    error=str(exc),
                )
                await asyncio.sleep(delay)

            except httpx.ConnectError as exc:
                last_exception = ServerError(
                    f"Connection error: {path}", status_code=None
                )
                delay = self._compute_backoff(attempt)
                self._log.warning(
                    "api_connect_error",
                    path=path,
                    attempt=attempt + 1,
                    delay_seconds=round(delay, 2),
                    error=str(exc),
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        self._log.error(
            "api_retries_exhausted",
            path=path,
            max_retries=self._max_retries,
        )
        raise last_exception or ServerError(f"Request failed after {self._max_retries} retries")

    def _classify_error(self, response: httpx.Response) -> APIError:
        """Classify HTTP error response into specific exception type."""
        status = response.status_code
        body = response.text[:500]  # Truncate long error bodies

        if status == 429:
            return RateLimitError(
                f"Rate limit exceeded (429)", status_code=429, response_body=body
            )
        elif status in (401, 403):
            return AuthenticationError(
                f"Authentication failed ({status})", status_code=status, response_body=body
            )
        elif status == 404:
            return DataNotFoundError(
                f"Resource not found (404)", status_code=404, response_body=body
            )
        elif status >= 500:
            return ServerError(
                f"Server error ({status})", status_code=status, response_body=body
            )
        else:
            return APIError(
                f"HTTP {status} error", status_code=status, response_body=body
            )

    def _compute_backoff(self, attempt: int) -> float:
        """
        Exponential backoff with full jitter.

        delay = random(0, base_delay * 2^attempt)
        Capped at 60 seconds.
        """
        max_delay = min(60.0, self._base_delay * (2**attempt))
        return random.uniform(0, max_delay)

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        await self._client.aclose()
        self._log.info("client_closed")

    async def __aenter__(self) -> BaseClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
