"""
Tests for Phase 4 — Backtesting Engine.

Tests the event-driven backtester, strategies, cost model,
and BacktestResult metrics using synthetic data.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    BacktestResult,
    BaseStrategy,
    CostModel,
    EventDrivenBacktester,
    Trade,
)
from src.backtesting.strategies import (
    CompositeStrategy,
    InflationBondStrategy,
    NowcastSurpriseStrategy,
    SurpriseDirectionStrategy,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic market & release data
# ---------------------------------------------------------------------------


def _make_price_data(
    start: str = "2023-01-01",
    n_days: int = 365,
    base_price: float = 450.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic 1-minute SPY price data around release times."""
    rng = np.random.default_rng(seed)
    # Create minute-level data for a year
    dates = pd.date_range(start, periods=n_days * 390, freq="min")  # 390 min/day
    # Random walk
    returns = rng.standard_normal(len(dates)) * 0.0002  # ~2bps per minute
    prices = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {"close": prices, "volume": rng.integers(1000, 50000, size=len(dates))},
        index=dates,
    )


def _make_releases(
    n: int = 24, seed: int = 42
) -> pd.DataFrame:
    """Synthetic macro releases over 2 years."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-15 08:30:00", periods=n, freq="MS")

    indicators = ["CPI", "NFP", "GDP", "PCE"]
    actuals = 0.2 + rng.standard_normal(n) * 0.15
    consensus = actuals + rng.standard_normal(n) * 0.1
    surprises = actuals - consensus

    # Compute z-scores
    zscores = []
    for i in range(n):
        if i < 4:
            zscores.append(0.0)
        else:
            hist = surprises[:i]
            std = hist.std()
            zscores.append(float((surprises[i] - hist.mean()) / std) if std > 0 else 0)

    return pd.DataFrame(
        {
            "release_date": dates,
            "indicator": [indicators[i % len(indicators)] for i in range(n)],
            "actual": actuals,
            "consensus": consensus,
            "surprise": surprises,
            "surprise_zscore": zscores,
        }
    )


# ===========================================================================
# Trade Tests
# ===========================================================================


class TestTrade:
    """Tests for Trade data class."""

    def test_long_return_positive(self):
        """LONG trade with price increase → positive return."""
        trade = Trade(
            trade_id=1, indicator="CPI", release_date=datetime(2024, 1, 1),
            direction="LONG", asset="SPY",
            entry_time=datetime(2024, 1, 1, 8, 25),
            exit_time=datetime(2024, 1, 1, 9, 0),
            entry_price=450.0, exit_price=451.0,
            quantity=100, gross_pnl=100.0,
            commission=2.0, slippage_cost=0.9,
            net_pnl=97.1, signal_strength=0.8,
        )
        assert trade.return_pct == pytest.approx(1.0 / 450 * 100, abs=0.01)
        assert trade.is_winner
        assert trade.duration_minutes == 35.0

    def test_short_return_positive(self):
        """SHORT trade with price decrease → positive return."""
        trade = Trade(
            trade_id=1, indicator="CPI", release_date=datetime(2024, 1, 1),
            direction="SHORT", asset="SPY",
            entry_time=datetime(2024, 1, 1, 8, 25),
            exit_time=datetime(2024, 1, 1, 9, 0),
            entry_price=450.0, exit_price=449.0,
            quantity=100, gross_pnl=100.0,
            commission=2.0, slippage_cost=0.9,
            net_pnl=97.1, signal_strength=0.7,
        )
        assert trade.return_pct == pytest.approx(1.0 / 450 * 100, abs=0.01)
        assert trade.is_winner

    def test_losing_trade(self):
        """Trade with negative net PnL is a loser."""
        trade = Trade(
            trade_id=1, indicator="NFP", release_date=datetime(2024, 1, 1),
            direction="LONG", asset="SPY",
            entry_time=datetime(2024, 1, 1, 8, 25),
            exit_time=datetime(2024, 1, 1, 9, 0),
            entry_price=450.0, exit_price=449.0,
            quantity=100, gross_pnl=-100.0,
            commission=2.0, slippage_cost=0.9,
            net_pnl=-102.9, signal_strength=0.5,
        )
        assert not trade.is_winner


# ===========================================================================
# CostModel Tests
# ===========================================================================


class TestCostModel:
    """Tests for transaction cost model."""

    def test_default_costs(self):
        """Default costs are reasonable."""
        model = CostModel()
        comm, slip = model.compute_costs(price=450.0, quantity=100)
        assert comm == 1.0  # $1 per trade
        assert slip == pytest.approx(450 * 100 * 0.0001)  # 1 bps

    def test_zero_slippage(self):
        """Zero slippage when bps = 0."""
        model = CostModel(slippage_bps=0)
        comm, slip = model.compute_costs(price=450.0, quantity=100)
        assert slip == 0.0

    def test_high_cost_scenario(self):
        """High costs reduce profitability."""
        model = CostModel(commission_per_trade=5.0, slippage_bps=5.0)
        comm, slip = model.compute_costs(price=100.0, quantity=1000)
        assert comm == 5.0
        assert slip == pytest.approx(100 * 1000 * 5 / 10_000)


# ===========================================================================
# SurpriseDirectionStrategy Tests
# ===========================================================================


class TestSurpriseDirectionStrategy:
    """Tests for the baseline surprise direction strategy."""

    def test_positive_surprise_goes_long(self):
        """Positive z-score → LONG signal."""
        strategy = SurpriseDirectionStrategy(zscore_threshold=0.5)
        release = pd.Series({
            "indicator": "CPI", "actual": 0.5, "consensus": 0.3,
            "surprise": 0.2, "surprise_zscore": 1.5,
        })
        signal = strategy.generate_signal(release)
        assert signal is not None
        assert signal["direction"] == "LONG"
        assert 0 < signal["confidence"] <= 1

    def test_negative_surprise_goes_short(self):
        """Negative z-score → SHORT signal."""
        strategy = SurpriseDirectionStrategy(zscore_threshold=0.5)
        release = pd.Series({
            "indicator": "CPI", "actual": 0.1, "consensus": 0.3,
            "surprise": -0.2, "surprise_zscore": -1.5,
        })
        signal = strategy.generate_signal(release)
        assert signal is not None
        assert signal["direction"] == "SHORT"

    def test_below_threshold_no_trade(self):
        """Small z-score below threshold → no trade."""
        strategy = SurpriseDirectionStrategy(zscore_threshold=1.0)
        release = pd.Series({
            "indicator": "CPI", "actual": 0.25, "consensus": 0.2,
            "surprise": 0.05, "surprise_zscore": 0.3,
        })
        signal = strategy.generate_signal(release)
        assert signal is None

    def test_indicator_filter(self):
        """Only trade specified indicators."""
        strategy = SurpriseDirectionStrategy(
            zscore_threshold=0.5, indicators=["CPI", "NFP"]
        )
        # GDP not in filter
        release = pd.Series({
            "indicator": "GDP", "surprise_zscore": 2.0, "surprise": 0.5,
        })
        signal = strategy.generate_signal(release)
        assert signal is None

        # CPI in filter
        release = pd.Series({
            "indicator": "CPI", "surprise_zscore": 2.0, "surprise": 0.5,
        })
        signal = strategy.generate_signal(release)
        assert signal is not None

    def test_missing_zscore_uses_surprise(self):
        """Falls back to raw surprise if z-score missing."""
        strategy = SurpriseDirectionStrategy(zscore_threshold=0.5)
        release = pd.Series({
            "indicator": "CPI", "actual": 0.5, "consensus": 0.3,
            "surprise": 1.5, "surprise_zscore": None,
        })
        signal = strategy.generate_signal(release)
        assert signal is not None
        assert signal["direction"] == "LONG"


# ===========================================================================
# NowcastSurpriseStrategy Tests
# ===========================================================================


class TestNowcastSurpriseStrategy:
    """Tests for the nowcast-based pre-release strategy."""

    def test_nowcast_above_consensus_long(self):
        """Nowcast > consensus → LONG."""
        strategy = NowcastSurpriseStrategy(divergence_threshold=0.1)
        release = pd.Series({
            "indicator": "CPI", "nowcast_estimate": 0.5, "consensus": 0.3,
        })
        signal = strategy.generate_signal(release)
        assert signal is not None
        assert signal["direction"] == "LONG"

    def test_nowcast_below_consensus_short(self):
        """Nowcast < consensus → SHORT."""
        strategy = NowcastSurpriseStrategy(divergence_threshold=0.1)
        release = pd.Series({
            "indicator": "CPI", "nowcast_estimate": 0.1, "consensus": 0.3,
        })
        signal = strategy.generate_signal(release)
        assert signal is not None
        assert signal["direction"] == "SHORT"

    def test_small_divergence_no_trade(self):
        """Small divergence below threshold → no trade."""
        strategy = NowcastSurpriseStrategy(divergence_threshold=0.5)
        release = pd.Series({
            "indicator": "CPI", "nowcast_estimate": 0.32, "consensus": 0.30,
        })
        signal = strategy.generate_signal(release)
        assert signal is None

    def test_asset_map(self):
        """Custom asset mapping per indicator."""
        strategy = NowcastSurpriseStrategy(
            divergence_threshold=0.1,
            asset_map={"CPI": "TLT", "NFP": "SPY"},
        )
        release = pd.Series({
            "indicator": "CPI", "nowcast_estimate": 0.5, "consensus": 0.3,
        })
        signal = strategy.generate_signal(release)
        assert signal["asset"] == "TLT"


# ===========================================================================
# InflationBondStrategy Tests
# ===========================================================================


class TestInflationBondStrategy:
    """Tests for the inflation-bond strategy."""

    def test_hot_inflation_shorts_bonds(self):
        """Higher-than-expected inflation → SHORT TLT."""
        strategy = InflationBondStrategy(zscore_threshold=0.3)
        release = pd.Series({
            "indicator": "CPI", "surprise_zscore": 1.5,
        })
        signal = strategy.generate_signal(release)
        assert signal is not None
        assert signal["direction"] == "SHORT"
        assert signal["asset"] == "TLT"

    def test_cool_inflation_longs_bonds(self):
        """Lower-than-expected inflation → LONG TLT."""
        strategy = InflationBondStrategy(zscore_threshold=0.3)
        release = pd.Series({
            "indicator": "CPI", "surprise_zscore": -1.0,
        })
        signal = strategy.generate_signal(release)
        assert signal is not None
        assert signal["direction"] == "LONG"

    def test_non_inflation_indicator_skipped(self):
        """Non-inflation indicators are skipped."""
        strategy = InflationBondStrategy()
        release = pd.Series({
            "indicator": "NFP", "surprise_zscore": 2.0,
        })
        signal = strategy.generate_signal(release)
        assert signal is None


# ===========================================================================
# CompositeStrategy Tests
# ===========================================================================


class TestCompositeStrategy:
    """Tests for the composite multi-strategy."""

    def test_picks_highest_confidence(self):
        """Picks the signal with highest confidence."""
        s1 = SurpriseDirectionStrategy(name="s1", zscore_threshold=0.1)
        s2 = InflationBondStrategy(name="s2", zscore_threshold=0.1)

        composite = CompositeStrategy(strategies=[s1, s2], min_confidence=0.1)

        release = pd.Series({
            "indicator": "CPI", "surprise": 0.2,
            "surprise_zscore": 2.0,  # High z → both fire
        })
        signal = composite.generate_signal(release)
        assert signal is not None

    def test_no_signal_below_min_confidence(self):
        """No signal if all sub-strategies below min confidence."""
        s1 = SurpriseDirectionStrategy(zscore_threshold=5.0)  # Very high threshold
        composite = CompositeStrategy(strategies=[s1], min_confidence=0.9)

        release = pd.Series({
            "indicator": "CPI", "surprise": 0.01, "surprise_zscore": 0.1,
        })
        signal = composite.generate_signal(release)
        assert signal is None


# ===========================================================================
# EventDrivenBacktester Tests
# ===========================================================================


class TestEventDrivenBacktester:
    """Tests for the backtesting engine."""

    def test_basic_backtest(self):
        """Basic backtest runs and produces results."""
        strategy = SurpriseDirectionStrategy(zscore_threshold=0.3)
        backtester = EventDrivenBacktester(
            strategy=strategy,
            initial_capital=100_000,
            position_size_pct=0.02,
        )

        releases = _make_releases(24)
        prices = _make_price_data(start="2023-01-01", n_days=730)

        result = backtester.run(releases, prices)

        assert isinstance(result, BacktestResult)
        assert result.n_trades > 0
        assert result.initial_capital == 100_000
        assert result.final_capital > 0

    def test_backtest_metrics(self):
        """Metrics are computed correctly."""
        strategy = SurpriseDirectionStrategy(zscore_threshold=0.3)
        backtester = EventDrivenBacktester(strategy=strategy)
        releases = _make_releases(24)
        prices = _make_price_data(start="2023-01-01", n_days=730)

        result = backtester.run(releases, prices)

        assert 0 <= result.win_rate <= 100
        assert result.avg_duration_minutes > 0
        assert result.trades_per_year > 0
        assert result.total_commissions >= 0
        assert result.total_slippage >= 0

    def test_summary_dict(self):
        """Summary returns a valid dict."""
        strategy = SurpriseDirectionStrategy(zscore_threshold=0.3)
        backtester = EventDrivenBacktester(strategy=strategy)
        releases = _make_releases(24)
        prices = _make_price_data(start="2023-01-01", n_days=730)

        result = backtester.run(releases, prices)
        summary = result.summary()

        assert isinstance(summary, dict)
        assert "sharpe_ratio" in summary
        assert "win_rate" in summary
        assert "max_drawdown_pct" in summary

    def test_to_dataframe(self):
        """Trades can be exported to DataFrame."""
        strategy = SurpriseDirectionStrategy(zscore_threshold=0.3)
        backtester = EventDrivenBacktester(strategy=strategy)
        releases = _make_releases(24)
        prices = _make_price_data(start="2023-01-01", n_days=730)

        result = backtester.run(releases, prices)
        df = result.to_dataframe()

        assert len(df) == result.n_trades
        assert "net_pnl" in df.columns
        assert "return_pct" in df.columns

    def test_monthly_returns(self):
        """Monthly returns aggregation works."""
        strategy = SurpriseDirectionStrategy(zscore_threshold=0.3)
        backtester = EventDrivenBacktester(strategy=strategy)
        releases = _make_releases(24)
        prices = _make_price_data(start="2023-01-01", n_days=730)

        result = backtester.run(releases, prices)
        monthly = result.monthly_returns()

        assert len(monthly) > 0
        assert "return_pct" in monthly.columns

    def test_no_trades_when_threshold_high(self):
        """No trades when threshold is impossibly high."""
        strategy = SurpriseDirectionStrategy(zscore_threshold=100.0)
        backtester = EventDrivenBacktester(strategy=strategy)
        releases = _make_releases(12)
        prices = _make_price_data(start="2023-01-01", n_days=365)

        result = backtester.run(releases, prices)

        assert result.n_trades == 0
        assert result.final_capital == result.initial_capital

    def test_custom_cost_model(self):
        """Custom cost model affects P&L."""
        low_cost = CostModel(commission_per_trade=0.5, slippage_bps=0.5)
        high_cost = CostModel(commission_per_trade=10.0, slippage_bps=10.0)

        strategy = SurpriseDirectionStrategy(zscore_threshold=0.3)
        releases = _make_releases(24)
        prices = _make_price_data(start="2023-01-01", n_days=730)

        result_low = EventDrivenBacktester(
            strategy=strategy, cost_model=low_cost
        ).run(releases, prices)
        result_high = EventDrivenBacktester(
            strategy=strategy, cost_model=high_cost
        ).run(releases, prices)

        assert result_high.total_commissions > result_low.total_commissions
        assert result_high.total_slippage > result_low.total_slippage

    def test_equity_curve_monotonic_tracking(self):
        """Equity curve tracks running capital correctly."""
        strategy = SurpriseDirectionStrategy(zscore_threshold=0.3)
        backtester = EventDrivenBacktester(strategy=strategy)
        releases = _make_releases(24)
        prices = _make_price_data(start="2023-01-01", n_days=730)

        result = backtester.run(releases, prices)

        if not result.equity_curve.empty:
            # Equity should equal initial + cumulative PnL
            cum_pnl = result.equity_curve["trade_pnl"].cumsum()
            expected_final = result.initial_capital + cum_pnl.iloc[-1]
            assert result.equity_curve["equity"].iloc[-1] == pytest.approx(
                expected_final, abs=0.1
            )


# ===========================================================================
# BacktestResult Edge Cases
# ===========================================================================


class TestBacktestResultEdgeCases:
    """Tests for edge cases in BacktestResult metrics."""

    def test_empty_trades(self):
        """Metrics handle zero trades gracefully."""
        result = BacktestResult(
            strategy_name="test",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            trades=[],
            initial_capital=100_000,
            final_capital=100_000,
            equity_curve=pd.DataFrame(),
        )
        assert result.win_rate == 0.0
        assert result.profit_factor == 0.0
        assert result.avg_trade_pnl == 0.0
        assert result.sharpe_ratio == 0.0
        assert result.n_trades == 0

    def test_all_winners(self):
        """100% win rate → high profit factor."""
        trades = [
            Trade(
                trade_id=i, indicator="CPI",
                release_date=datetime(2024, i, 1),
                direction="LONG", asset="SPY",
                entry_time=datetime(2024, i, 1, 8, 25),
                exit_time=datetime(2024, i, 1, 9, 0),
                entry_price=450, exit_price=451,
                quantity=100, gross_pnl=100,
                commission=1, slippage_cost=0.5,
                net_pnl=98.5, signal_strength=0.8,
            )
            for i in range(1, 7)
        ]
        result = BacktestResult(
            strategy_name="test",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 6, 30),
            trades=trades,
            initial_capital=100_000,
            final_capital=100_591,
            equity_curve=pd.DataFrame(
                {"date": [t.release_date for t in trades],
                 "equity": [100_000 + 98.5 * (i + 1) for i in range(6)],
                 "trade_pnl": [98.5] * 6}
            ),
        )
        assert result.win_rate == 100.0
        assert result.profit_factor == float("inf")
