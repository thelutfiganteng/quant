"""
Event-Driven Backtesting Engine.

Simulates trading around scheduled macroeconomic data releases.
Supports entry before release, exit after, with configurable timing,
transaction costs, and position sizing.

Architecture:
    1. EventSchedule: defines release dates + indicators
    2. Strategy: decides direction/size from nowcast + surprise signals
    3. Engine: loops through events, executes trades, tracks PnL
    4. Results: computes metrics (Sharpe, drawdown, win rate, etc.)

Key design constraints (from plan):
    - Walk-forward only: strategy sees no future data
    - Realistic costs: slippage + commission per trade
    - No look-ahead: nowcast uses only pre-release information
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Core Types
# ---------------------------------------------------------------------------


class TradeDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class TradeConfig:
    """Transaction cost and timing configuration."""

    slippage_bps: float = 1.0         # 0.01% slippage per trade
    commission_per_trade: float = 1.0  # $1 commission
    entry_minutes_before: int = 5      # Enter T-5min
    exit_minutes_after: int = 30       # Exit T+30min
    max_position_pct: float = 1.0      # Max 100% of capital


@dataclass
class TradeResult:
    """Result of a single trade."""

    event_date: datetime
    indicator: str
    direction: TradeDirection
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    position_size: float       # Number of shares/contracts
    pnl_gross: float           # Before costs
    pnl_net: float             # After costs
    slippage_cost: float
    commission_cost: float
    return_pct: float          # Net return %
    signal_strength: float     # How confident the signal was
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_winner(self) -> bool:
        return self.pnl_net > 0


@dataclass
class BacktestResults:
    """Complete backtest results with performance metrics."""

    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    trades: list[TradeResult]

    # Computed on creation
    _metrics: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.trades and not self._metrics:
            self._metrics = self._compute_metrics()

    def _compute_metrics(self) -> dict[str, Any]:
        """Compute all performance metrics from trade list."""
        if not self.trades:
            return self._empty_metrics()

        returns = np.array([t.return_pct for t in self.trades])
        pnls = np.array([t.pnl_net for t in self.trades])
        winners = [t for t in self.trades if t.is_winner]
        losers = [t for t in self.trades if not t.is_winner]

        # Total return
        total_return_pct = (
            (self.final_capital - self.initial_capital) / self.initial_capital
        ) * 100

        # Time period
        days = (self.end_date - self.start_date).days
        years = max(days / 365.25, 1 / 365.25)

        # Annualized return (CAGR)
        ann_return = ((self.final_capital / self.initial_capital) ** (1 / years) - 1) * 100

        # Sharpe ratio (annualized, assuming ~250 trading days)
        if len(returns) > 1 and np.std(returns) > 0:
            trades_per_year = len(returns) / years
            sharpe = (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0

        # Max drawdown
        equity_curve = self._build_equity_curve()
        max_dd = self._compute_max_drawdown(equity_curve)

        # Win rate
        win_rate = len(winners) / len(self.trades) * 100 if self.trades else 0

        # Profit factor
        gross_profit = sum(t.pnl_net for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl_net for t in losers)) if losers else 1e-10
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average trade metrics
        avg_winner = np.mean([t.pnl_net for t in winners]) if winners else 0
        avg_loser = np.mean([t.pnl_net for t in losers]) if losers else 0

        # Trade duration
        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 60
            for t in self.trades
        ]

        # Total costs
        total_slippage = sum(t.slippage_cost for t in self.trades)
        total_commission = sum(t.commission_cost for t in self.trades)

        return {
            "total_return": round(total_return_pct, 2),
            "annualized_return": round(ann_return, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown": round(max_dd, 2),
            "win_rate": round(win_rate, 1),
            "total_trades": len(self.trades),
            "trades_per_year": round(len(self.trades) / years, 1),
            "avg_trade_duration_minutes": round(float(np.mean(durations)), 1),
            "profit_factor": round(profit_factor, 2),
            "avg_winner": round(float(avg_winner), 2),
            "avg_loser": round(float(avg_loser), 2),
            "max_consecutive_losses": self._max_consecutive(False),
            "max_consecutive_wins": self._max_consecutive(True),
            "total_slippage": round(total_slippage, 2),
            "total_commission": round(total_commission, 2),
            "total_costs": round(total_slippage + total_commission, 2),
        }

    def _empty_metrics(self) -> dict[str, Any]:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "trades_per_year": 0.0,
            "avg_trade_duration_minutes": 0.0,
            "profit_factor": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "max_consecutive_losses": 0,
            "max_consecutive_wins": 0,
            "total_slippage": 0.0,
            "total_commission": 0.0,
            "total_costs": 0.0,
        }

    def _build_equity_curve(self) -> pd.Series:
        """Build equity curve from sequential trades."""
        equity = [self.initial_capital]
        for trade in sorted(self.trades, key=lambda t: t.event_date):
            equity.append(equity[-1] + trade.pnl_net)
        dates = [self.start_date] + [t.event_date for t in sorted(self.trades, key=lambda t: t.event_date)]
        return pd.Series(equity, index=dates)

    @staticmethod
    def _compute_max_drawdown(equity_curve: pd.Series) -> float:
        """Compute maximum drawdown percentage."""
        peak = equity_curve.expanding().max()
        drawdown = ((equity_curve - peak) / peak) * 100
        return round(float(drawdown.min()), 2)

    def _max_consecutive(self, wins: bool) -> int:
        """Count max consecutive wins or losses."""
        max_streak = 0
        current = 0
        for t in sorted(self.trades, key=lambda t: t.event_date):
            if t.is_winner == wins:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @property
    def metrics(self) -> dict[str, Any]:
        return self._metrics

    def monthly_returns(self) -> pd.DataFrame:
        """Aggregate trade returns by month."""
        if not self.trades:
            return pd.DataFrame(columns=["month", "return", "n_trades"])

        data = [
            {"date": t.event_date, "pnl": t.pnl_net, "return_pct": t.return_pct}
            for t in self.trades
        ]
        df = pd.DataFrame(data)
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")

        monthly = df.groupby("month").agg(
            total_return=("return_pct", "sum"),
            n_trades=("pnl", "count"),
            total_pnl=("pnl", "sum"),
        ).reset_index()

        monthly["month"] = monthly["month"].astype(str)
        return monthly

    def summary(self) -> dict[str, Any]:
        """Full summary for API/display."""
        return {
            "strategy": self.strategy_name,
            "period": f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "metrics": self.metrics,
            "monthly_returns": self.monthly_returns().to_dict("records"),
        }


# ---------------------------------------------------------------------------
# Backtesting Engine
# ---------------------------------------------------------------------------


class EventDrivenBacktester:
    """
    Event-driven backtesting engine.

    Simulates trading around macro release events with:
        - Configurable entry/exit timing relative to release
        - Transaction costs (slippage + commission)
        - Walk-forward signal evaluation (no future data)
        - Per-trade analytics

    Usage:
        engine = EventDrivenBacktester(config=TradeConfig(...))
        results = engine.run(
            events=event_schedule,
            price_data=market_data,
            strategy=my_strategy,
            initial_capital=100_000,
        )
    """

    def __init__(self, config: TradeConfig | None = None):
        self.config = config or TradeConfig()

    def run(
        self,
        events: pd.DataFrame,
        price_data: pd.DataFrame,
        strategy: "BaseStrategy",
        initial_capital: float = 100_000.0,
    ) -> BacktestResults:
        """
        Run the backtest over all events.

        Args:
            events: DataFrame with columns:
                - 'date': release datetime
                - 'indicator': string (CPI, NFP, etc.)
                - 'actual': actual released value
                - 'consensus': consensus estimate
                - 'surprise': actual - consensus
                Plus any strategy-specific columns.
            price_data: DataFrame with OHLCV columns:
                - Index: datetime
                - 'open', 'high', 'low', 'close', 'volume'
            strategy: Strategy instance to generate signals.
            initial_capital: Starting capital in USD.

        Returns:
            BacktestResults with all trades and metrics.
        """
        events = events.sort_values("date").reset_index(drop=True)
        capital = initial_capital
        trades: list[TradeResult] = []

        logger.info(
            "backtest_starting",
            strategy=strategy.name,
            n_events=len(events),
            initial_capital=initial_capital,
        )

        for _, event in events.iterrows():
            event_time = pd.Timestamp(event["date"])

            # Get strategy signal
            signal = strategy.generate_signal(event, capital)
            if signal.direction == TradeDirection.FLAT:
                continue

            # Find entry/exit prices
            entry_time = event_time - timedelta(minutes=self.config.entry_minutes_before)
            exit_time = event_time + timedelta(minutes=self.config.exit_minutes_after)

            entry_price = self._get_price(price_data, entry_time, "entry")
            exit_price = self._get_price(price_data, exit_time, "exit")

            if entry_price is None or exit_price is None:
                logger.debug(
                    "trade_skipped_no_price",
                    indicator=event.get("indicator", ""),
                    event_time=str(event_time),
                )
                continue

            # Position sizing
            position_value = capital * signal.position_pct
            position_size = position_value / entry_price

            # Compute PnL
            if signal.direction == TradeDirection.LONG:
                pnl_gross = (exit_price - entry_price) * position_size
            else:  # SHORT
                pnl_gross = (entry_price - exit_price) * position_size

            # Transaction costs
            slippage = position_value * (self.config.slippage_bps / 10_000) * 2  # Entry + exit
            commission = self.config.commission_per_trade * 2  # Entry + exit
            pnl_net = pnl_gross - slippage - commission

            # Return percentage (on capital used)
            return_pct = (pnl_net / position_value) * 100 if position_value > 0 else 0.0

            # Update capital
            capital += pnl_net

            trade = TradeResult(
                event_date=event_time.to_pydatetime() if hasattr(event_time, 'to_pydatetime') else event_time,
                indicator=str(event.get("indicator", "")),
                direction=signal.direction,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=entry_time.to_pydatetime() if hasattr(entry_time, 'to_pydatetime') else entry_time,
                exit_time=exit_time.to_pydatetime() if hasattr(exit_time, 'to_pydatetime') else exit_time,
                position_size=position_size,
                pnl_gross=round(pnl_gross, 2),
                pnl_net=round(pnl_net, 2),
                slippage_cost=round(slippage, 2),
                commission_cost=round(commission, 2),
                return_pct=round(return_pct, 4),
                signal_strength=signal.confidence,
                metadata=signal.metadata,
            )

            trades.append(trade)

        results = BacktestResults(
            strategy_name=strategy.name,
            start_date=events["date"].min(),
            end_date=events["date"].max(),
            initial_capital=initial_capital,
            final_capital=round(capital, 2),
            trades=trades,
        )

        logger.info(
            "backtest_complete",
            strategy=strategy.name,
            n_trades=len(trades),
            total_return=results.metrics.get("total_return"),
            sharpe=results.metrics.get("sharpe_ratio"),
        )

        return results

    def _get_price(
        self, price_data: pd.DataFrame, target_time: pd.Timestamp, label: str
    ) -> float | None:
        """
        Get the price closest to target_time.

        For intraday data, finds the nearest bar.
        For daily data, uses the day's open (entry) or close (exit).
        """
        if price_data.empty:
            return None

        idx = price_data.index

        if not isinstance(idx, pd.DatetimeIndex):
            return None

        # For daily data: use open for entry, close for exit
        target_date = target_time.normalize()
        if target_date in idx:
            row = price_data.loc[target_date]
            return float(row["open"]) if label == "entry" else float(row["close"])

        # Find nearest available bar
        diffs = abs(idx - target_time)
        nearest_idx = diffs.argmin()
        nearest_time = idx[nearest_idx]

        # Don't use data more than 1 day away
        if abs((nearest_time - target_time).total_seconds()) > 86400:
            return None

        row = price_data.iloc[nearest_idx]
        return float(row["open"]) if label == "entry" else float(row["close"])


# ---------------------------------------------------------------------------
# Strategy Signal
# ---------------------------------------------------------------------------


@dataclass
class StrategySignal:
    """Output from a strategy's signal generation."""

    direction: TradeDirection
    confidence: float = 0.5        # 0-1 confidence level
    position_pct: float = 1.0     # Fraction of capital to use
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy:
    """Abstract base for trading strategies."""

    def __init__(self, name: str):
        self.name = name

    def generate_signal(
        self, event: pd.Series, capital: float
    ) -> StrategySignal:
        """Generate a trade signal for an event. Override in subclass."""
        raise NotImplementedError
