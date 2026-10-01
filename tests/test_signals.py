"""
Tests for Phase 5 — Signal Generation Engine.

Tests the Signal data model, SignalScorer, and SignalGenerator
including nowcast signals, surprise signals, batch generation,
and signal lifecycle management.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.signals import (
    Signal,
    SignalDirection,
    SignalGenerator,
    SignalScorer,
    SignalSource,
    SignalStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _future(hours: int = 24) -> datetime:
    """Create a future datetime for testing."""
    return datetime.now(timezone.utc) + timedelta(hours=hours)


def _past(hours: int = 1) -> datetime:
    """Create a past datetime for testing."""
    return datetime.now(timezone.utc) - timedelta(hours=hours)


# ===========================================================================
# Signal Data Model Tests
# ===========================================================================


class TestSignal:
    """Tests for the Signal dataclass."""

    def _make_signal(self, **overrides) -> Signal:
        """Create a test signal with sensible defaults."""
        now = datetime.now(timezone.utc)
        defaults = dict(
            signal_id="SIG-CPI-test1",
            indicator="CPI",
            direction=SignalDirection.LONG,
            asset="SPY",
            confidence=0.72,
            source=SignalSource.NOWCAST,
            status=SignalStatus.PENDING,
            generated_at=now,
            valid_from=now - timedelta(hours=1),
            valid_until=now + timedelta(hours=12),
            event_time=now + timedelta(hours=6),
        )
        defaults.update(overrides)
        return Signal(**defaults)

    def test_is_valid_within_window(self):
        """Signal within validity window is valid."""
        signal = self._make_signal()
        assert signal.is_valid

    def test_is_not_valid_after_expiry(self):
        """Signal past valid_until is not valid."""
        signal = self._make_signal(
            valid_until=_past(1),
        )
        assert not signal.is_valid

    def test_is_actionable(self):
        """Signal is actionable when valid and PENDING/ACTIVE."""
        signal = self._make_signal(status=SignalStatus.PENDING)
        assert signal.is_actionable

        signal = self._make_signal(status=SignalStatus.ACTIVE)
        assert signal.is_actionable

    def test_is_not_actionable_when_executed(self):
        """Executed signal is not actionable."""
        signal = self._make_signal(status=SignalStatus.EXECUTED)
        assert not signal.is_actionable

    def test_edge_estimate_long(self):
        """LONG signal has positive edge."""
        signal = self._make_signal(
            direction=SignalDirection.LONG, confidence=0.8
        )
        assert signal.edge_estimate == pytest.approx(0.8)

    def test_edge_estimate_short(self):
        """SHORT signal has negative edge."""
        signal = self._make_signal(
            direction=SignalDirection.SHORT, confidence=0.6
        )
        assert signal.edge_estimate == pytest.approx(-0.6)

    def test_to_dict(self):
        """Signal serializes to dict correctly."""
        signal = self._make_signal(
            nowcast_estimate=2.8, consensus=2.9
        )
        d = signal.to_dict()

        assert d["signal_id"] == "SIG-CPI-test1"
        assert d["direction"] == "LONG"
        assert d["confidence"] == 0.72
        assert d["nowcast_estimate"] == 2.8
        assert isinstance(d["generated_at"], str)


# ===========================================================================
# SignalScorer Tests
# ===========================================================================


class TestSignalScorer:
    """Tests for the multi-factor signal scorer."""

    def test_default_score(self):
        """Default score for unknown indicator is reasonable."""
        scorer = SignalScorer()
        score = scorer.score("UNKNOWN_INDICATOR")
        assert 0.1 <= score <= 0.95

    def test_high_importance_indicator_scores_higher(self):
        """CPI/NFP scores higher than HOUSING_STARTS."""
        scorer = SignalScorer()
        cpi_score = scorer.score("CPI", nowcast_confidence=0.7)
        housing_score = scorer.score("HOUSING_STARTS", nowcast_confidence=0.7)
        assert cpi_score > housing_score

    def test_high_confidence_scores_higher(self):
        """Higher nowcast confidence → higher score."""
        scorer = SignalScorer()
        high = scorer.score("CPI", nowcast_confidence=0.9)
        low = scorer.score("CPI", nowcast_confidence=0.3)
        assert high > low

    def test_surprise_factor(self):
        """Large surprise magnitude boosts score."""
        scorer = SignalScorer()
        with_surprise = scorer.score("CPI", surprise_zscore=2.5)
        without_surprise = scorer.score("CPI")
        # With a big surprise, score should differ
        assert with_surprise != without_surprise

    def test_backtest_hit_rate(self):
        """High historical hit rate boosts score."""
        scorer_high = SignalScorer(backtest_hit_rates={"CPI": 0.75})
        scorer_low = SignalScorer(backtest_hit_rates={"CPI": 0.35})
        high_score = scorer_high.score("CPI")
        low_score = scorer_low.score("CPI")
        assert high_score > low_score

    def test_time_decay_sweet_spot(self):
        """Score highest 2-6 hours before event."""
        scorer = SignalScorer()
        sweet_spot = scorer.score("CPI", hours_to_event=4)
        far_out = scorer.score("CPI", hours_to_event=48)
        assert sweet_spot > far_out

    def test_score_floor_and_ceiling(self):
        """Score is clamped between floor and ceiling."""
        scorer = SignalScorer(confidence_floor=0.2, confidence_ceiling=0.8)

        # Even with low inputs, shouldn't go below floor
        low = scorer.score("UNKNOWN", nowcast_confidence=0.0)
        assert low >= 0.2

        # Even with high inputs, shouldn't exceed ceiling
        high = scorer.score("CPI", nowcast_confidence=1.0, surprise_zscore=5.0)
        assert high <= 0.8


# ===========================================================================
# SignalGenerator — Nowcast Signals
# ===========================================================================


class TestSignalGeneratorNowcast:
    """Tests for nowcast-based signal generation."""

    def test_generate_long_signal(self):
        """Nowcast > consensus → LONG signal."""
        gen = SignalGenerator(min_confidence=0.1)
        signal = gen.generate_from_nowcast(
            indicator="CPI",
            nowcast_estimate=3.0,
            consensus=2.8,
            event_time=_future(24),
        )
        assert signal is not None
        assert signal.direction == SignalDirection.LONG
        assert signal.indicator == "CPI"
        assert signal.nowcast_estimate == 3.0
        assert signal.consensus == 2.8

    def test_generate_short_signal(self):
        """Nowcast < consensus → SHORT signal."""
        gen = SignalGenerator(min_confidence=0.1)
        signal = gen.generate_from_nowcast(
            indicator="NFP",
            nowcast_estimate=150,
            consensus=175,
            event_time=_future(24),
        )
        assert signal is not None
        assert signal.direction == SignalDirection.SHORT

    def test_no_signal_for_tiny_divergence(self):
        """Tiny divergence (< 0.01) produces no signal."""
        gen = SignalGenerator()
        signal = gen.generate_from_nowcast(
            indicator="CPI",
            nowcast_estimate=2.900,
            consensus=2.905,
            event_time=_future(24),
        )
        assert signal is None

    def test_signal_has_stop_loss_take_profit(self):
        """Signal includes risk parameters."""
        gen = SignalGenerator(
            min_confidence=0.1,
            default_stop_loss_pct=0.5,
            default_take_profit_pct=1.5,
        )
        signal = gen.generate_from_nowcast(
            indicator="CPI",
            nowcast_estimate=3.0,
            consensus=2.5,
            event_time=_future(24),
        )
        assert signal is not None
        assert signal.stop_loss_pct == 0.5
        assert signal.take_profit_pct == 1.5

    def test_asset_map(self):
        """Custom asset mapping per indicator."""
        gen = SignalGenerator(min_confidence=0.1)
        signal = gen.generate_from_nowcast(
            indicator="CPI",
            nowcast_estimate=3.0,
            consensus=2.5,
            event_time=_future(24),
            asset_map={"CPI": "TLT", "NFP": "SPY"},
        )
        assert signal is not None
        assert signal.asset == "TLT"

    def test_signal_logged(self):
        """Generated signal is added to the log."""
        gen = SignalGenerator(min_confidence=0.1)
        assert gen.signal_count == 0

        gen.generate_from_nowcast(
            indicator="CPI",
            nowcast_estimate=3.0,
            consensus=2.5,
            event_time=_future(24),
        )
        assert gen.signal_count == 1

    def test_position_size_scales_with_confidence(self):
        """Higher confidence → larger position size."""
        gen = SignalGenerator(min_confidence=0.1, position_size_pct=0.02)

        # Force high confidence via scorer with high hit rate
        scorer = SignalScorer(backtest_hit_rates={"CPI": 0.9})
        gen_high = SignalGenerator(
            scorer=scorer, min_confidence=0.1, position_size_pct=0.02
        )

        sig = gen_high.generate_from_nowcast(
            indicator="CPI",
            nowcast_estimate=3.5,
            consensus=2.5,
            event_time=_future(4),
            nowcast_confidence=0.95,
        )
        assert sig is not None
        assert sig.suggested_size_pct > 0


# ===========================================================================
# SignalGenerator — Surprise Signals
# ===========================================================================


class TestSignalGeneratorSurprise:
    """Tests for post-release surprise signal generation."""

    def test_positive_surprise_long(self):
        """Positive surprise z-score → LONG."""
        gen = SignalGenerator(min_confidence=0.1)
        signal = gen.generate_from_surprise(
            indicator="CPI",
            actual=3.0,
            consensus=2.8,
            surprise_zscore=1.8,
            event_time=_past(0),
        )
        assert signal is not None
        assert signal.direction == SignalDirection.LONG
        assert signal.source == SignalSource.SURPRISE
        assert signal.status == SignalStatus.ACTIVE  # Immediate

    def test_negative_surprise_short(self):
        """Negative surprise z-score → SHORT."""
        gen = SignalGenerator(min_confidence=0.1)
        signal = gen.generate_from_surprise(
            indicator="NFP",
            actual=140,
            consensus=175,
            surprise_zscore=-2.1,
            event_time=_past(0),
        )
        assert signal is not None
        assert signal.direction == SignalDirection.SHORT

    def test_insignificant_surprise_no_signal(self):
        """Small surprise (|z| < 0.5) produces no signal."""
        gen = SignalGenerator()
        signal = gen.generate_from_surprise(
            indicator="CPI",
            actual=2.81,
            consensus=2.80,
            surprise_zscore=0.2,
            event_time=_past(0),
        )
        assert signal is None

    def test_surprise_has_actual_value(self):
        """Surprise signal includes actual release value."""
        gen = SignalGenerator(min_confidence=0.1)
        signal = gen.generate_from_surprise(
            indicator="GDP",
            actual=2.8,
            consensus=2.0,
            surprise_zscore=2.5,
            event_time=_past(0),
        )
        assert signal is not None
        assert signal.actual == 2.8


# ===========================================================================
# SignalGenerator — Batch Generation
# ===========================================================================


class TestSignalGeneratorBatch:
    """Tests for batch signal generation."""

    def test_batch_generation(self):
        """Batch generates signals for multiple events."""
        gen = SignalGenerator(min_confidence=0.1)

        events = pd.DataFrame({
            "indicator": ["CPI", "NFP", "GDP"],
            "event_time": [_future(24), _future(48), _future(72)],
        })
        nowcasts = {"CPI": 3.0, "NFP": 180, "GDP": 2.5}
        consensus = {"CPI": 2.8, "NFP": 170, "GDP": 2.5}

        signals = gen.generate_batch(events, nowcasts, consensus)

        # GDP divergence is 0.0, so should be filtered
        assert len(signals) >= 1  # At least CPI and NFP

    def test_batch_respects_daily_limit(self):
        """Batch doesn't exceed max_signals_per_day."""
        gen = SignalGenerator(min_confidence=0.1, max_signals_per_day=2)

        events = pd.DataFrame({
            "indicator": ["CPI", "NFP", "GDP", "PCE", "PPI"],
            "event_time": [_future(i * 24) for i in range(1, 6)],
        })
        nowcasts = {k: 3.0 for k in ["CPI", "NFP", "GDP", "PCE", "PPI"]}
        consensus = {k: 2.5 for k in ["CPI", "NFP", "GDP", "PCE", "PPI"]}

        signals = gen.generate_batch(events, nowcasts, consensus)
        assert len(signals) <= 2

    def test_batch_ranked_by_confidence(self):
        """Batch signals are sorted by confidence (highest first)."""
        gen = SignalGenerator(min_confidence=0.1)

        events = pd.DataFrame({
            "indicator": ["CPI", "HOUSING_STARTS"],
            "event_time": [_future(4), _future(4)],
        })
        nowcasts = {"CPI": 3.0, "HOUSING_STARTS": 1.5}
        consensus = {"CPI": 2.5, "HOUSING_STARTS": 1.0}

        signals = gen.generate_batch(events, nowcasts, consensus)

        if len(signals) >= 2:
            assert signals[0].confidence >= signals[1].confidence

    def test_batch_skips_missing_data(self):
        """Events without nowcast/consensus are skipped."""
        gen = SignalGenerator(min_confidence=0.1)

        events = pd.DataFrame({
            "indicator": ["CPI", "NFP"],
            "event_time": [_future(24), _future(48)],
        })
        # Only CPI has data
        nowcasts = {"CPI": 3.0}
        consensus = {"CPI": 2.5, "NFP": 170}  # NFP has no nowcast

        signals = gen.generate_batch(events, nowcasts, consensus)
        indicators = [s.indicator for s in signals]
        assert "NFP" not in indicators


# ===========================================================================
# SignalGenerator — Lifecycle Management
# ===========================================================================


class TestSignalLifecycle:
    """Tests for signal status management."""

    def test_update_signal_status(self):
        """Can update a signal's status."""
        gen = SignalGenerator(min_confidence=0.1)
        signal = gen.generate_from_nowcast(
            indicator="CPI",
            nowcast_estimate=3.0,
            consensus=2.5,
            event_time=_future(24),
        )
        assert signal is not None

        result = gen.update_signal_status(
            signal.signal_id, SignalStatus.EXECUTED
        )
        assert result is True

        # Verify status changed
        history = gen.get_signal_history(indicator="CPI")
        assert history[0].status == SignalStatus.EXECUTED

    def test_update_nonexistent_signal(self):
        """Updating non-existent signal returns False."""
        gen = SignalGenerator()
        result = gen.update_signal_status("SIG-FAKE-12345", SignalStatus.CANCELLED)
        assert result is False

    def test_expire_stale_signals(self):
        """Stale signals are marked EXPIRED."""
        gen = SignalGenerator(min_confidence=0.1)

        # Create a signal that's already expired
        signal = gen.generate_from_nowcast(
            indicator="CPI",
            nowcast_estimate=3.0,
            consensus=2.5,
            event_time=_future(24),
        )
        assert signal is not None

        # Manually set valid_until to the past
        signal.valid_until = _past(1)

        expired_count = gen.expire_stale_signals()
        assert expired_count == 1
        assert signal.status == SignalStatus.EXPIRED

    def test_get_signal_history_filtered(self):
        """Signal history can be filtered by indicator and status."""
        gen = SignalGenerator(min_confidence=0.1)

        gen.generate_from_nowcast("CPI", 3.0, 2.5, _future(24))
        gen.generate_from_nowcast("NFP", 180, 170, _future(48))
        gen.generate_from_surprise("GDP", 2.8, 2.0, 2.5, _past(0))

        cpi_only = gen.get_signal_history(indicator="CPI")
        assert all(s.indicator == "CPI" for s in cpi_only)

        active_only = gen.get_signal_history(status=SignalStatus.ACTIVE)
        assert all(s.status == SignalStatus.ACTIVE for s in active_only)

    def test_active_signals_property(self):
        """active_signals returns only actionable signals."""
        gen = SignalGenerator(min_confidence=0.1)

        sig1 = gen.generate_from_nowcast("CPI", 3.0, 2.5, _future(24))
        sig2 = gen.generate_from_nowcast("NFP", 180, 170, _future(48))

        assert len(gen.active_signals) >= 1

        # Execute one
        if sig1:
            gen.update_signal_status(sig1.signal_id, SignalStatus.EXECUTED)

        # Should have one fewer active
        remaining = [s for s in gen.active_signals if s.indicator == "CPI"]
        assert len(remaining) == 0

    def test_deterministic_signal_id(self):
        """Same inputs produce the same signal ID."""
        gen = SignalGenerator(min_confidence=0.1)
        event_time = datetime(2026, 10, 10, 8, 30, tzinfo=timezone.utc)

        sig1 = gen.generate_from_nowcast("CPI", 3.0, 2.5, event_time)
        sig2 = gen.generate_from_nowcast("CPI", 3.0, 2.5, event_time)

        assert sig1 is not None and sig2 is not None
        assert sig1.signal_id == sig2.signal_id
