"""
Signal Generation Engine.

Combines nowcast estimates, surprise analysis, and strategy rules
to produce actionable trading signals with confidence scoring.

Signal Pipeline:
    1. Data Collection: gather latest nowcasts + release schedule
    2. Signal Generation: run strategies against upcoming events
    3. Scoring: rank signals by confidence × historical edge
    4. Risk Filter: apply position limits, correlation checks
    5. Emission: output final signals for dashboard + execution

Design:
    - Each signal has a unique ID and full audit trail
    - Signals are immutable once generated (append-only log)
    - Confidence is calibrated against backtest accuracy
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums & Types
# ---------------------------------------------------------------------------


class SignalDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class SignalStatus(str, Enum):
    PENDING = "PENDING"      # Generated, awaiting confirmation
    ACTIVE = "ACTIVE"        # Confirmed, ready for execution
    EXECUTED = "EXECUTED"    # Position opened
    CLOSED = "CLOSED"        # Position closed
    EXPIRED = "EXPIRED"      # Not acted on before expiry
    CANCELLED = "CANCELLED"  # Manually cancelled


class SignalSource(str, Enum):
    NOWCAST = "NOWCAST"          # Pre-release nowcast divergence
    SURPRISE = "SURPRISE"        # Post-release surprise reaction
    COMPOSITE = "COMPOSITE"      # Multi-factor composite
    MANUAL = "MANUAL"            # Manual override


# ---------------------------------------------------------------------------
# Signal Data Model
# ---------------------------------------------------------------------------


@dataclass
class Signal:
    """
    Trading signal with full metadata and audit trail.

    A signal is the atomic output of the system: a recommendation
    to go LONG/SHORT a specific asset around a macro event.
    """

    signal_id: str
    indicator: str
    direction: SignalDirection
    asset: str
    confidence: float  # 0.0 to 1.0
    source: SignalSource
    status: SignalStatus

    # Timing
    generated_at: datetime
    valid_from: datetime
    valid_until: datetime
    event_time: datetime  # Scheduled release time

    # Sizing
    suggested_size_pct: float = 0.02  # % of portfolio

    # Context
    nowcast_estimate: float | None = None
    consensus: float | None = None
    actual: float | None = None
    surprise_zscore: float | None = None

    # Strategy info
    strategy_name: str = ""
    model_name: str = ""

    # Risk
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if signal is still within its validity window."""
        now = datetime.now(timezone.utc)
        return self.valid_from <= now <= self.valid_until

    @property
    def is_actionable(self) -> bool:
        """Signal is actionable if valid and in PENDING/ACTIVE status."""
        return self.is_valid and self.status in (
            SignalStatus.PENDING, SignalStatus.ACTIVE
        )

    @property
    def edge_estimate(self) -> float:
        """Estimated edge = confidence × direction factor."""
        return self.confidence * (1 if self.direction == SignalDirection.LONG else -1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API/dashboard."""
        return {
            "signal_id": self.signal_id,
            "indicator": self.indicator,
            "direction": self.direction.value,
            "asset": self.asset,
            "confidence": round(self.confidence, 4),
            "source": self.source.value,
            "status": self.status.value,
            "generated_at": self.generated_at.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "event_time": self.event_time.isoformat(),
            "suggested_size_pct": self.suggested_size_pct,
            "nowcast_estimate": self.nowcast_estimate,
            "consensus": self.consensus,
            "surprise_zscore": self.surprise_zscore,
            "strategy_name": self.strategy_name,
            "model_name": self.model_name,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Signal Scorer
# ---------------------------------------------------------------------------


class SignalScorer:
    """
    Scores and ranks signals using multi-factor confidence model.

    Factors:
        1. Nowcast confidence (model R² or walk-forward accuracy)
        2. Surprise magnitude (|z-score|)
        3. Historical hit rate for this indicator
        4. Indicator importance (CPI > Housing Starts)
        5. Time decay (confidence drops as event approaches)
    """

    # Indicator importance weights (higher = more market-moving)
    INDICATOR_WEIGHTS: dict[str, float] = {
        "CPI": 1.0,
        "CORE_CPI": 0.95,
        "NFP": 1.0,
        "GDP": 0.85,
        "PCE": 0.90,
        "CORE_PCE": 0.90,
        "UNEMPLOYMENT_RATE": 0.80,
        "PPI": 0.70,
        "INITIAL_CLAIMS": 0.50,
        "RETAIL_SALES": 0.75,
        "ISM_MANUFACTURING": 0.70,
        "ISM_SERVICES": 0.65,
        "HOUSING_STARTS": 0.40,
        "CONSUMER_CONFIDENCE": 0.55,
        "DURABLE_GOODS": 0.50,
    }

    def __init__(
        self,
        backtest_hit_rates: dict[str, float] | None = None,
        confidence_floor: float = 0.1,
        confidence_ceiling: float = 0.95,
    ):
        """
        Args:
            backtest_hit_rates: Historical win rates per indicator
                                from backtesting. Keys = indicator name.
            confidence_floor: Minimum confidence score.
            confidence_ceiling: Maximum confidence score.
        """
        self.backtest_hit_rates = backtest_hit_rates or {}
        self.confidence_floor = confidence_floor
        self.confidence_ceiling = confidence_ceiling

    def score(
        self,
        indicator: str,
        nowcast_confidence: float = 0.5,
        surprise_zscore: float | None = None,
        hours_to_event: float | None = None,
    ) -> float:
        """
        Compute composite confidence score.

        Args:
            indicator: Economic indicator name.
            nowcast_confidence: Model confidence (0-1).
            surprise_zscore: Surprise z-score (if post-release).
            hours_to_event: Hours until the release event.

        Returns:
            Composite confidence score between floor and ceiling.
        """
        factors = []
        weights = []

        # Factor 1: Nowcast model confidence
        factors.append(min(nowcast_confidence, 1.0))
        weights.append(0.30)

        # Factor 2: Indicator importance
        importance = self.INDICATOR_WEIGHTS.get(indicator, 0.5)
        factors.append(importance)
        weights.append(0.20)

        # Factor 3: Historical backtest hit rate
        hit_rate = self.backtest_hit_rates.get(indicator, 0.50)
        factors.append(hit_rate)
        weights.append(0.25)

        # Factor 4: Surprise magnitude (if available)
        if surprise_zscore is not None:
            surprise_factor = min(abs(surprise_zscore) / 3.0, 1.0)
            factors.append(surprise_factor)
            weights.append(0.15)
        else:
            # Redistribute weight
            weights[0] += 0.05
            weights[1] += 0.05
            weights[2] += 0.05

        # Factor 5: Time decay (optional)
        if hours_to_event is not None and hours_to_event > 0:
            # Confidence peaks 2-6 hours before event
            if hours_to_event <= 2:
                time_factor = 0.95  # Close to event — high confidence
            elif hours_to_event <= 6:
                time_factor = 1.0   # Sweet spot
            elif hours_to_event <= 24:
                time_factor = 0.85
            else:
                time_factor = 0.7   # Far out — lower confidence
            factors.append(time_factor)
            weights.append(0.10)
        else:
            weights[0] += 0.05
            weights[2] += 0.05

        # Normalize weights
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        # Weighted average
        score = sum(f * w for f, w in zip(factors, weights))

        # Clamp to floor/ceiling
        score = max(self.confidence_floor, min(self.confidence_ceiling, score))

        return round(score, 4)


# ---------------------------------------------------------------------------
# Signal Generator
# ---------------------------------------------------------------------------


class SignalGenerator:
    """
    Main signal generation engine.

    Orchestrates the full signal pipeline:
        1. Evaluate upcoming events
        2. Run nowcast models
        3. Generate directional signals
        4. Score and rank
        5. Apply risk filters
        6. Emit final signals
    """

    def __init__(
        self,
        scorer: SignalScorer | None = None,
        min_confidence: float = 0.3,
        max_signals_per_day: int = 5,
        default_stop_loss_pct: float = 0.5,
        default_take_profit_pct: float = 1.0,
        position_size_pct: float = 0.02,
    ):
        self.scorer = scorer or SignalScorer()
        self.min_confidence = min_confidence
        self.max_signals_per_day = max_signals_per_day
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct
        self.position_size_pct = position_size_pct

        # Signal log (append-only)
        self._signal_log: list[Signal] = []

    @property
    def signal_count(self) -> int:
        return len(self._signal_log)

    @property
    def active_signals(self) -> list[Signal]:
        return [s for s in self._signal_log if s.is_actionable]

    def generate_from_nowcast(
        self,
        indicator: str,
        nowcast_estimate: float,
        consensus: float,
        event_time: datetime,
        model_name: str = "bridge_v1",
        asset: str = "SPY",
        asset_map: dict[str, str] | None = None,
        nowcast_confidence: float = 0.5,
    ) -> Signal | None:
        """
        Generate a pre-release signal from nowcast vs consensus divergence.

        Args:
            indicator: Economic indicator.
            nowcast_estimate: Model's estimate of the upcoming release.
            consensus: Market consensus estimate.
            event_time: Scheduled release datetime.
            model_name: Name of the nowcast model used.
            asset: Default trading asset.
            asset_map: Optional indicator → asset mapping.
            nowcast_confidence: Model's confidence level (0-1).

        Returns:
            Signal if divergence exceeds threshold, None otherwise.
        """
        divergence = nowcast_estimate - consensus

        # Determine direction
        if abs(divergence) < 0.01:  # Too small to trade
            return None

        direction = SignalDirection.LONG if divergence > 0 else SignalDirection.SHORT

        # Inflation indicators: invert for bonds
        target_asset = asset
        if asset_map:
            target_asset = asset_map.get(indicator, asset)

        # Score the signal
        hours_to_event = max(
            (event_time - datetime.now(timezone.utc)).total_seconds() / 3600, 0
        )
        confidence = self.scorer.score(
            indicator=indicator,
            nowcast_confidence=nowcast_confidence,
            hours_to_event=hours_to_event,
        )

        if confidence < self.min_confidence:
            logger.debug(
                "signal_below_threshold",
                indicator=indicator,
                confidence=confidence,
                min_required=self.min_confidence,
            )
            return None

        # Create signal
        now = datetime.now(timezone.utc)
        signal = Signal(
            signal_id=self._generate_id(indicator, event_time),
            indicator=indicator,
            direction=direction,
            asset=target_asset,
            confidence=confidence,
            source=SignalSource.NOWCAST,
            status=SignalStatus.PENDING,
            generated_at=now,
            valid_from=now,
            valid_until=event_time,
            event_time=event_time,
            suggested_size_pct=self._compute_position_size(confidence),
            nowcast_estimate=round(nowcast_estimate, 4),
            consensus=round(consensus, 4),
            strategy_name="nowcast_divergence",
            model_name=model_name,
            stop_loss_pct=self.default_stop_loss_pct,
            take_profit_pct=self.default_take_profit_pct,
            metadata={
                "divergence": round(divergence, 4),
                "divergence_pct": round(divergence / abs(consensus) * 100, 2) if consensus != 0 else 0,
            },
        )

        self._signal_log.append(signal)

        logger.info(
            "signal_generated",
            signal_id=signal.signal_id,
            indicator=indicator,
            direction=direction.value,
            confidence=confidence,
            asset=target_asset,
        )

        return signal

    def generate_from_surprise(
        self,
        indicator: str,
        actual: float,
        consensus: float,
        surprise_zscore: float,
        event_time: datetime,
        asset: str = "SPY",
    ) -> Signal | None:
        """
        Generate a post-release signal from actual surprise data.

        Args:
            indicator: Economic indicator.
            actual: Actual released value.
            consensus: Market consensus.
            surprise_zscore: Standardized surprise.
            event_time: Release datetime.
            asset: Trading asset.

        Returns:
            Signal if surprise is significant, None otherwise.
        """
        if abs(surprise_zscore) < 0.5:  # Not significant
            return None

        direction = SignalDirection.LONG if surprise_zscore > 0 else SignalDirection.SHORT

        confidence = self.scorer.score(
            indicator=indicator,
            surprise_zscore=surprise_zscore,
        )

        if confidence < self.min_confidence:
            return None

        now = datetime.now(timezone.utc)
        from datetime import timedelta
        signal = Signal(
            signal_id=self._generate_id(indicator, event_time, suffix="surprise"),
            indicator=indicator,
            direction=direction,
            asset=asset,
            confidence=confidence,
            source=SignalSource.SURPRISE,
            status=SignalStatus.ACTIVE,  # Immediate action
            generated_at=now,
            valid_from=now,
            valid_until=now + timedelta(hours=2),
            event_time=event_time,
            suggested_size_pct=self._compute_position_size(confidence),
            actual=actual,
            consensus=round(consensus, 4),
            surprise_zscore=round(surprise_zscore, 4),
            strategy_name="surprise_reaction",
            stop_loss_pct=self.default_stop_loss_pct,
            take_profit_pct=self.default_take_profit_pct,
            metadata={
                "surprise_raw": round(actual - consensus, 4),
            },
        )

        self._signal_log.append(signal)

        logger.info(
            "surprise_signal_generated",
            signal_id=signal.signal_id,
            indicator=indicator,
            direction=direction.value,
            zscore=surprise_zscore,
        )

        return signal

    def generate_batch(
        self,
        upcoming_events: pd.DataFrame,
        nowcast_estimates: dict[str, float],
        consensus_estimates: dict[str, float],
        model_confidences: dict[str, float] | None = None,
        asset_map: dict[str, str] | None = None,
    ) -> list[Signal]:
        """
        Generate signals for a batch of upcoming events.

        Args:
            upcoming_events: DataFrame with columns:
                - indicator: str
                - event_time: datetime
            nowcast_estimates: indicator → point estimate.
            consensus_estimates: indicator → consensus.
            model_confidences: indicator → model confidence.
            asset_map: indicator → asset mapping.

        Returns:
            List of generated signals, ranked by confidence.
        """
        signals: list[Signal] = []

        for _, event in upcoming_events.iterrows():
            indicator = event["indicator"]
            event_time = pd.Timestamp(event["event_time"]).to_pydatetime()
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=timezone.utc)

            nowcast = nowcast_estimates.get(indicator)
            consensus = consensus_estimates.get(indicator)

            if nowcast is None or consensus is None:
                continue

            model_conf = (model_confidences or {}).get(indicator, 0.5)

            signal = self.generate_from_nowcast(
                indicator=indicator,
                nowcast_estimate=nowcast,
                consensus=consensus,
                event_time=event_time,
                nowcast_confidence=model_conf,
                asset_map=asset_map,
            )
            if signal:
                signals.append(signal)

        # Rank by confidence (highest first)
        signals.sort(key=lambda s: s.confidence, reverse=True)

        # Apply daily limit
        signals = signals[: self.max_signals_per_day]

        logger.info(
            "batch_signals_generated",
            n_events=len(upcoming_events),
            n_signals=len(signals),
        )

        return signals

    def get_signal_history(
        self,
        indicator: str | None = None,
        status: SignalStatus | None = None,
        limit: int = 50,
    ) -> list[Signal]:
        """Get historical signals with optional filters."""
        filtered = self._signal_log

        if indicator:
            filtered = [s for s in filtered if s.indicator == indicator]
        if status:
            filtered = [s for s in filtered if s.status == status]

        return sorted(filtered, key=lambda s: s.generated_at, reverse=True)[:limit]

    def update_signal_status(
        self, signal_id: str, new_status: SignalStatus
    ) -> bool:
        """Update a signal's status. Returns True if found."""
        for signal in self._signal_log:
            if signal.signal_id == signal_id:
                signal.status = new_status
                logger.info(
                    "signal_status_updated",
                    signal_id=signal_id,
                    new_status=new_status.value,
                )
                return True
        return False

    def expire_stale_signals(self) -> int:
        """Mark expired signals. Returns count of expired."""
        count = 0
        now = datetime.now(timezone.utc)
        for signal in self._signal_log:
            if signal.status in (SignalStatus.PENDING, SignalStatus.ACTIVE):
                if now > signal.valid_until:
                    signal.status = SignalStatus.EXPIRED
                    count += 1
        if count:
            logger.info("signals_expired", count=count)
        return count

    def _compute_position_size(self, confidence: float) -> float:
        """Scale position size by confidence (higher conf → larger size)."""
        base = self.position_size_pct
        # Scale: 50% at min confidence, 100% at max confidence
        scale = 0.5 + 0.5 * (confidence / 0.95)
        return round(base * min(scale, 1.5), 4)

    @staticmethod
    def _generate_id(
        indicator: str, event_time: datetime, suffix: str = ""
    ) -> str:
        """Generate a deterministic signal ID."""
        seed = f"{indicator}:{event_time.isoformat()}:{suffix}"
        hash_val = hashlib.md5(seed.encode()).hexdigest()[:8]
        return f"SIG-{indicator[:3]}-{hash_val}"
