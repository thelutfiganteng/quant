"""
Event-Driven Backtesting Engine.

Simulates trading around macroeconomic data releases. Unlike traditional
bar-by-bar backtesting, this engine is event-driven:

    1. Wait for a scheduled macro release (CPI, NFP, etc.)
    2. Evaluate the signal (nowcast vs consensus)
    3. Enter position at T-offset (e.g., 5 min before release)
    4. Exit position at T+offset (e.g., 30 min after release)
    5. Record P&L with transaction costs

Design principles:
    - No look-ahead bias: signals use only pre-release data
    - Realistic costs: slippage + commission
    - Walk-forward: strategy params trained on past data only
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Containers
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """Single trade record."""

    trade_id: int
    indicator: str
    release_date: datetime
    direction: str  # "LONG" or "SHORT"
    asset: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    commission: float
    slippage_cost: float
    net_pnl: float
    signal_strength: float  # |z-score| or confidence
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def return_pct(self) -> float:
        """Return as percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.direction == "LONG":
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def duration_minutes(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 60


@dataclass
class BacktestResult:
    """Complete backtest results and performance metrics."""

    strategy_name: str
    start_date: datetime
    end_date: datetime
    trades: list[Trade]
    initial_capital: float
    final_capital: float
    equity_curve: pd.DataFrame

    # --- Computed Metrics ---

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return ((self.final_capital - self.initial_capital) / self.initial_capital) * 100

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def n_winners(self) -> int:
        return sum(1 for t in self.trades if t.is_winner)

    @property
    def n_losers(self) -> int:
        return self.n_trades - self.n_winners

    @property
    def win_rate(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return (self.n_winners / self.n_trades) * 100

    @property
    def avg_trade_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.net_pnl for t in self.trades) / self.n_trades

    @property
    def avg_winner(self) -> float:
        winners = [t.net_pnl for t in self.trades if t.is_winner]
        return sum(winners) / len(winners) if winners else 0.0

    @property
    def avg_loser(self) -> float:
        losers = [t.net_pnl for t in self.trades if not t.is_winner]
        return sum(losers) / len(losers) if losers else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.net_pnl for t in self.trades if t.is_winner)
        gross_loss = abs(sum(t.net_pnl for t in self.trades if not t.is_winner))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        equity = self.equity_curve["equity"].values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        return float(np.min(drawdown))

    @property
    def sharpe_ratio(self) -> float:
        if self.equity_curve.empty or len(self.equity_curve) < 2:
            return 0.0
        returns = self.equity_curve["equity"].pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        # Annualize assuming ~12 events per year (monthly macro releases)
        return float(returns.mean() / returns.std() * np.sqrt(12))

    @property
    def trades_per_year(self) -> float:
        if not self.trades:
            return 0.0
        days = (self.end_date - self.start_date).days
        if days == 0:
            return 0.0
        return self.n_trades / (days / 365.25)

    @property
    def avg_duration_minutes(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.duration_minutes for t in self.trades) / self.n_trades

    @property
    def total_commissions(self) -> float:
        return sum(t.commission for t in self.trades)

    @property
    def total_slippage(self) -> float:
        return sum(t.slippage_cost for t in self.trades)

    def summary(self) -> dict[str, Any]:
        """Flat summary dict for logging/display."""
        return {
            "strategy": self.strategy_name,
            "period": f"{self.start_date.date()} to {self.end_date.date()}",
            "total_return_pct": round(self.total_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "n_trades": self.n_trades,
            "win_rate": round(self.win_rate, 1),
            "profit_factor": round(self.profit_factor, 2),
            "avg_trade_pnl": round(self.avg_trade_pnl, 2),
            "avg_duration_min": round(self.avg_duration_minutes, 1),
            "trades_per_year": round(self.trades_per_year, 1),
            "total_commissions": round(self.total_commissions, 2),
            "total_slippage": round(self.total_slippage, 2),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(
            [
                {
                    "trade_id": t.trade_id,
                    "indicator": t.indicator,
                    "release_date": t.release_date,
                    "direction": t.direction,
                    "asset": t.asset,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl,
                    "return_pct": t.return_pct,
                    "signal_strength": t.signal_strength,
                    "is_winner": t.is_winner,
                    "duration_min": t.duration_minutes,
                }
                for t in self.trades
            ]
        )

    def monthly_returns(self) -> pd.DataFrame:
        """Aggregate returns by month."""
        trades_df = self.to_dataframe()
        if trades_df.empty:
            return pd.DataFrame(columns=["month", "return", "n_trades"])

        trades_df["month"] = pd.to_datetime(trades_df["release_date"]).dt.to_period("M")
        monthly = (
            trades_df.groupby("month")
            .agg(
                total_return=("net_pnl", "sum"),
                n_trades=("trade_id", "count"),
                win_rate=("is_winner", "mean"),
            )
            .reset_index()
        )
        monthly["return_pct"] = monthly["total_return"] / self.initial_capital * 100
        monthly["month"] = monthly["month"].astype(str)
        return monthly


# ---------------------------------------------------------------------------
# Backtesting Engine
# ---------------------------------------------------------------------------


@dataclass
class CostModel:
    """Transaction cost model."""

    commission_per_trade: float = 1.0  # $ per trade
    slippage_bps: float = 1.0  # basis points (0.01%)

    def compute_costs(
        self, price: float, quantity: float
    ) -> tuple[float, float]:
        """
        Compute commission and slippage for a trade.

        Returns:
            (commission, slippage_cost)
        """
        commission = self.commission_per_trade
        slippage = price * quantity * (self.slippage_bps / 10_000)
        return commission, slippage


class EventDrivenBacktester:
    """
    Backtesting engine for event-driven macro strategies.

    Process:
        1. Iterate through historical macro releases chronologically
        2. For each release, generate a signal from the strategy
        3. Simulate entry/exit with realistic market prices
        4. Track P&L, equity curve, and performance metrics
    """

    def __init__(
        self,
        strategy: "BaseStrategy",
        initial_capital: float = 100_000.0,
        cost_model: CostModel | None = None,
        position_size_pct: float = 0.02,  # 2% of capital per trade
    ):
        """
        Args:
            strategy: Trading strategy implementing BaseStrategy.
            initial_capital: Starting capital in USD.
            cost_model: Transaction cost model.
            position_size_pct: Fraction of capital to risk per trade.
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.cost_model = cost_model or CostModel()
        self.position_size_pct = position_size_pct

    def run(
        self,
        releases: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> BacktestResult:
        """
        Run the backtest over historical releases.

        Args:
            releases: DataFrame with columns:
                - release_date: datetime
                - indicator: str
                - actual: float
                - consensus: float
                - surprise: float (optional)
                - surprise_zscore: float (optional)
            price_data: DataFrame with columns:
                - timestamp: datetime (index)
                - close: float
                Must cover the time range of releases with
                sufficient granularity (e.g., 1-minute bars).

        Returns:
            BacktestResult with all trades and metrics.
        """
        releases = releases.sort_values("release_date").reset_index(drop=True)
        capital = self.initial_capital
        trades: list[Trade] = []
        equity_points: list[dict] = []
        trade_counter = 0

        logger.info(
            "backtest_starting",
            strategy=self.strategy.name,
            n_releases=len(releases),
            initial_capital=self.initial_capital,
        )

        for _, release in releases.iterrows():
            release_time = pd.Timestamp(release["release_date"])

            # Generate signal
            signal = self.strategy.generate_signal(release)

            if signal is None or signal["direction"] == "NONE":
                continue

            # Get entry/exit prices from market data
            entry_time = release_time - timedelta(
                minutes=self.strategy.entry_offset_minutes
            )
            exit_time = release_time + timedelta(
                minutes=self.strategy.exit_offset_minutes
            )

            entry_price = self._get_price(price_data, entry_time)
            exit_price = self._get_price(price_data, exit_time)

            if entry_price is None or exit_price is None:
                logger.debug(
                    "price_not_found",
                    release_date=str(release_time),
                    entry_time=str(entry_time),
                    exit_time=str(exit_time),
                )
                continue

            # Position sizing
            position_capital = capital * self.position_size_pct
            quantity = position_capital / entry_price

            # Compute P&L
            direction = signal["direction"]
            if direction == "LONG":
                gross_pnl = (exit_price - entry_price) * quantity
            else:  # SHORT
                gross_pnl = (entry_price - exit_price) * quantity

            # Transaction costs (entry + exit)
            comm, slippage = self.cost_model.compute_costs(entry_price, quantity)
            total_cost = (comm + slippage) * 2  # Both legs
            net_pnl = gross_pnl - total_cost

            capital += net_pnl
            trade_counter += 1

            trade = Trade(
                trade_id=trade_counter,
                indicator=release.get("indicator", ""),
                release_date=release_time.to_pydatetime(),
                direction=direction,
                asset=signal.get("asset", "SPY"),
                entry_time=entry_time.to_pydatetime(),
                exit_time=exit_time.to_pydatetime(),
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                gross_pnl=round(gross_pnl, 2),
                commission=round(comm * 2, 2),
                slippage_cost=round(slippage * 2, 2),
                net_pnl=round(net_pnl, 2),
                signal_strength=signal.get("confidence", 0),
                metadata=signal.get("metadata", {}),
            )
            trades.append(trade)

            equity_points.append(
                {
                    "date": release_time,
                    "equity": round(capital, 2),
                    "trade_pnl": round(net_pnl, 2),
                }
            )

        # Build equity curve
        equity_curve = pd.DataFrame(equity_points)
        if equity_curve.empty:
            equity_curve = pd.DataFrame(columns=["date", "equity", "trade_pnl"])

        result = BacktestResult(
            strategy_name=self.strategy.name,
            start_date=releases["release_date"].min(),
            end_date=releases["release_date"].max(),
            trades=trades,
            initial_capital=self.initial_capital,
            final_capital=round(capital, 2),
            equity_curve=equity_curve,
        )

        logger.info("backtest_complete", **result.summary())
        return result

    @staticmethod
    def _get_price(
        price_data: pd.DataFrame, target_time: pd.Timestamp
    ) -> float | None:
        """
        Get the closest price at or before target_time.

        Uses forward-fill logic: take the last known price
        at or before the target time.
        """
        if price_data.empty:
            return None

        # Ensure datetime index
        if not isinstance(price_data.index, pd.DatetimeIndex):
            return None

        # Find closest price at or before target_time
        available = price_data[price_data.index <= target_time]
        if available.empty:
            # Try nearest
            idx = price_data.index.get_indexer([target_time], method="nearest")[0]
            if idx >= 0:
                return float(price_data["close"].iloc[idx])
            return None

        return float(available["close"].iloc[-1])


# ---------------------------------------------------------------------------
# Base Strategy Interface
# ---------------------------------------------------------------------------


class BaseStrategy:
    """
    Abstract base for event-driven trading strategies.

    Subclasses must implement generate_signal().
    """

    def __init__(
        self,
        name: str,
        entry_offset_minutes: int = 5,
        exit_offset_minutes: int = 30,
    ):
        self.name = name
        self.entry_offset_minutes = entry_offset_minutes
        self.exit_offset_minutes = exit_offset_minutes

    def generate_signal(
        self, release: pd.Series
    ) -> dict[str, Any] | None:
        """
        Generate a trading signal for a macro release.

        Args:
            release: Series with release data (indicator, actual,
                     consensus, surprise, surprise_zscore).

        Returns:
            Dict with keys: direction ('LONG'/'SHORT'/'NONE'),
            confidence (0-1), asset, metadata.
            None = no trade.
        """
        raise NotImplementedError
