"""Initial schema - create all tables and TimescaleDB hypertables

Revision ID: 001_initial
Revises: None
Create Date: 2026-09-10
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Enable TimescaleDB extension ---
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

    # --- economic_releases ---
    op.create_table(
        "economic_releases",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("indicator", sa.String(50), nullable=False),
        sa.Column("release_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actual", sa.Numeric(12, 4), nullable=True),
        sa.Column("consensus", sa.Numeric(12, 4), nullable=True),
        sa.Column("previous", sa.Numeric(12, 4), nullable=True),
        sa.Column("surprise", sa.Numeric(12, 6), nullable=True),
        sa.Column("surprise_zscore", sa.Numeric(8, 4), nullable=True),
        sa.Column("revision", sa.Numeric(12, 4), nullable=True),
        sa.Column("source", sa.String(50), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_releases_indicator", "economic_releases", ["indicator"])
    op.create_index("ix_releases_indicator_date", "economic_releases", ["indicator", "release_date"])
    op.create_index("ix_releases_date", "economic_releases", ["release_date"])
    op.create_index(
        "uq_releases_indicator_date_source",
        "economic_releases",
        ["indicator", "release_date", "source"],
        unique=True,
    )

    # Convert to TimescaleDB hypertable
    op.execute(
        "SELECT create_hypertable('economic_releases', 'release_date', "
        "migrate_data => true, if_not_exists => true);"
    )

    # --- market_prices ---
    op.create_table(
        "market_prices",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(12, 4), nullable=False),
        sa.Column("high", sa.Numeric(12, 4), nullable=False),
        sa.Column("low", sa.Numeric(12, 4), nullable=False),
        sa.Column("close", sa.Numeric(12, 4), nullable=False),
        sa.Column("volume", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("vwap", sa.Numeric(12, 4), nullable=True),
        sa.Column("timeframe", sa.String(10), nullable=False, server_default="1d"),
    )
    op.create_index("ix_prices_ticker", "market_prices", ["ticker"])
    op.create_index("ix_prices_ticker_timestamp", "market_prices", ["ticker", "timestamp"])
    op.create_index("ix_prices_timestamp", "market_prices", ["timestamp"])
    op.create_index(
        "uq_prices_ticker_ts_tf",
        "market_prices",
        ["ticker", "timestamp", "timeframe"],
        unique=True,
    )

    op.execute(
        "SELECT create_hypertable('market_prices', 'timestamp', "
        "migrate_data => true, if_not_exists => true);"
    )

    # --- nowcast_estimates ---
    op.create_table(
        "nowcast_estimates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("indicator", sa.String(50), nullable=False),
        sa.Column("target_release_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("estimated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("point_estimate", sa.Numeric(12, 4), nullable=False),
        sa.Column("confidence_lower", sa.Numeric(12, 4), nullable=True),
        sa.Column("confidence_upper", sa.Numeric(12, 4), nullable=True),
        sa.Column("model_name", sa.String(100), nullable=False, server_default=""),
        sa.Column("features_used", sa.JSON, nullable=True),
    )
    op.create_index("ix_nowcast_indicator", "nowcast_estimates", ["indicator"])
    op.create_index("ix_nowcast_indicator_target", "nowcast_estimates", ["indicator", "target_release_date"])
    op.create_index("ix_nowcast_estimated_at", "nowcast_estimates", ["estimated_at"])

    op.execute(
        "SELECT create_hypertable('nowcast_estimates', 'estimated_at', "
        "migrate_data => true, if_not_exists => true);"
    )

    # --- calendar_events (regular table, not hypertable) ---
    op.create_table(
        "calendar_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("indicator", sa.String(50), nullable=False),
        sa.Column("country", sa.String(3), nullable=False, server_default="US"),
        sa.Column("scheduled_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("importance", sa.String(10), nullable=False, server_default="MEDIUM"),
        sa.Column("actual", sa.Numeric(12, 4), nullable=True),
        sa.Column("forecast", sa.Numeric(12, 4), nullable=True),
        sa.Column("previous", sa.Numeric(12, 4), nullable=True),
        sa.Column("source", sa.String(50), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_calendar_indicator", "calendar_events", ["indicator"])
    op.create_index("ix_calendar_date", "calendar_events", ["scheduled_date"])
    op.create_index(
        "uq_calendar_indicator_date",
        "calendar_events",
        ["indicator", "scheduled_date", "country"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("calendar_events")
    op.drop_table("nowcast_estimates")
    op.drop_table("market_prices")
    op.drop_table("economic_releases")
