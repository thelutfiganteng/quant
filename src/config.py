"""
Application configuration using pydantic-settings.

Loads all settings from environment variables (or .env file).
Validates that required values are present at import time — fail fast,
don't silently run with missing API keys.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Central configuration — all values loaded from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"

    # --- Database ---
    database_url: str = "postgresql+asyncpg://mquser:mqpass@localhost:5432/macro_quant"
    database_url_sync: str = "postgresql+psycopg2://mquser:mqpass@localhost:5432/macro_quant"
    timescaledb_enabled: bool = True

    # --- API Keys ---
    fred_api_key: str = ""
    bls_api_key: str = ""
    trading_economics_api_key: str = ""
    polygon_api_key: str = ""

    # --- Rate Limits (requests per second) ---
    fred_rate_limit: float = 2.0  # FRED: ~120 req/min
    bls_rate_limit: float = 1.0  # BLS: conservative
    trading_economics_rate_limit: float = 1.0
    polygon_rate_limit: float = 0.08  # Free tier: 5 req/min

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return upper

    @model_validator(mode="after")
    def check_required_keys(self) -> Settings:
        """
        Fail fast if critical API keys are missing in non-development mode.
        In development, we allow empty keys but log warnings at startup.
        """
        if self.environment != Environment.DEVELOPMENT:
            missing = []
            if not self.fred_api_key:
                missing.append("FRED_API_KEY")
            if not self.polygon_api_key:
                missing.append("POLYGON_API_KEY")
            if missing:
                raise ValueError(
                    f"Missing required API keys for {self.environment.value}: {', '.join(missing)}"
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Singleton settings instance.
    Cached so .env is only read once per process.
    """
    return Settings()
