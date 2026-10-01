"""
Macro Event Trading Strategies.

Concrete strategy implementations for event-driven trading
around macroeconomic data releases.

Core logic:
    - If nowcast predicts surprise > threshold → LONG risk assets
    - If nowcast predicts surprise < -threshold → SHORT risk assets
    - Confidence scales with |z-score| magnitude
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog

from src.backtesting import BaseStrategy

logger = structlog.get_logger(__name__)


class SurpriseDirectionStrategy(BaseStrategy):
    """
    Trade based on surprise direction and magnitude.

    Rules:
        - Positive surprise (actual > consensus) → LONG
        - Negative surprise (actual < consensus) → SHORT
        - |Z-score| < threshold → No trade (filter noise)

    This is the simplest macro event strategy and serves as
    the baseline for more complex approaches.
    """

    def __init__(
        self,
        name: str = "surprise_direction",
        zscore_threshold: float = 0.5,
        asset: str = "SPY",
        entry_offset_minutes: int = 1,
        exit_offset_minutes: int = 30,
        indicators: list[str] | None = None,
    ):
        """
        Args:
            name: Strategy identifier.
            zscore_threshold: Minimum |z-score| to trigger a trade.
            asset: Default asset to trade.
            entry_offset_minutes: Minutes before release to enter.
            exit_offset_minutes: Minutes after release to exit.
            indicators: Filter to only trade these indicators.
                        None = trade all.
        """
        super().__init__(name, entry_offset_minutes, exit_offset_minutes)
        self.zscore_threshold = zscore_threshold
        self.asset = asset
        self.indicators = indicators

    def generate_signal(
        self, release: pd.Series
    ) -> dict[str, Any] | None:
        """Generate signal based on surprise z-score."""
        indicator = release.get("indicator", "")

        # Filter by indicator if specified
        if self.indicators and indicator not in self.indicators:
            return None

        zscore = release.get("surprise_zscore")
        surprise = release.get("surprise")

        # Need z-score for signal
        if zscore is None or pd.isna(zscore):
            if surprise is not None and not pd.isna(surprise):
                zscore = surprise  # Fallback: use raw surprise
            else:
                return None

        # Threshold filter
        if abs(zscore) < self.zscore_threshold:
            return None

        direction = "LONG" if zscore > 0 else "SHORT"
        confidence = min(abs(zscore) / 3.0, 1.0)  # Normalize to 0-1

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "asset": self.asset,
            "metadata": {
                "indicator": indicator,
                "zscore": round(float(zscore), 4),
                "surprise": float(surprise) if surprise is not None else None,
                "strategy": self.name,
            },
        }


class NowcastSurpriseStrategy(BaseStrategy):
    """
    Trade based on nowcast vs consensus divergence.

    Instead of waiting for the actual release, this strategy
    enters BEFORE the release based on the nowcast prediction:

        - Nowcast > consensus → expect positive surprise → LONG
        - Nowcast < consensus → expect negative surprise → SHORT

    Key advantage: enters position before the event, capturing
    the full surprise move. Risk: nowcast could be wrong.
    """

    def __init__(
        self,
        name: str = "nowcast_surprise",
        divergence_threshold: float = 0.1,
        asset: str = "SPY",
        entry_offset_minutes: int = 5,
        exit_offset_minutes: int = 30,
        indicators: list[str] | None = None,
        asset_map: dict[str, str] | None = None,
    ):
        """
        Args:
            name: Strategy identifier.
            divergence_threshold: Min |nowcast - consensus| to trade.
            asset: Default asset.
            entry_offset_minutes: Minutes before release.
            exit_offset_minutes: Minutes after release.
            indicators: Filter to specific indicators.
            asset_map: Map indicator → asset (e.g., CPI → TLT).
        """
        super().__init__(name, entry_offset_minutes, exit_offset_minutes)
        self.divergence_threshold = divergence_threshold
        self.asset = asset
        self.indicators = indicators
        self.asset_map = asset_map or {}

    def generate_signal(
        self, release: pd.Series
    ) -> dict[str, Any] | None:
        """Generate signal from nowcast-consensus divergence."""
        indicator = release.get("indicator", "")

        if self.indicators and indicator not in self.indicators:
            return None

        nowcast = release.get("nowcast_estimate")
        consensus = release.get("consensus")

        if nowcast is None or consensus is None:
            return None
        if pd.isna(nowcast) or pd.isna(consensus):
            return None

        divergence = nowcast - consensus

        if abs(divergence) < self.divergence_threshold:
            return None

        direction = "LONG" if divergence > 0 else "SHORT"
        confidence = min(abs(divergence) / (self.divergence_threshold * 5), 1.0)
        target_asset = self.asset_map.get(indicator, self.asset)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "asset": target_asset,
            "metadata": {
                "indicator": indicator,
                "nowcast": round(float(nowcast), 4),
                "consensus": round(float(consensus), 4),
                "divergence": round(float(divergence), 4),
                "strategy": self.name,
            },
        }


class InflationBondStrategy(BaseStrategy):
    """
    Specialized strategy for CPI/PCE releases trading bonds.

    Logic:
        - Higher-than-expected inflation → SHORT TLT (bonds drop)
        - Lower-than-expected inflation → LONG TLT (bonds rally)
        - Also trades TIPS spread via TIP

    This exploits the tight relationship between inflation surprises
    and bond yields.
    """

    def __init__(
        self,
        name: str = "inflation_bonds",
        zscore_threshold: float = 0.3,
        entry_offset_minutes: int = 1,
        exit_offset_minutes: int = 60,
    ):
        super().__init__(name, entry_offset_minutes, exit_offset_minutes)
        self.zscore_threshold = zscore_threshold
        self.inflation_indicators = {"CPI", "CORE_CPI", "PCE", "CORE_PCE", "PPI"}

    def generate_signal(
        self, release: pd.Series
    ) -> dict[str, Any] | None:
        """Generate bond trade signal from inflation surprises."""
        indicator = release.get("indicator", "")

        if indicator not in self.inflation_indicators:
            return None

        zscore = release.get("surprise_zscore")
        if zscore is None or pd.isna(zscore):
            return None

        if abs(zscore) < self.zscore_threshold:
            return None

        # Inverse: hot inflation → short bonds, cool inflation → long bonds
        direction = "SHORT" if zscore > 0 else "LONG"
        confidence = min(abs(zscore) / 2.5, 1.0)

        return {
            "direction": direction,
            "confidence": round(confidence, 4),
            "asset": "TLT",
            "metadata": {
                "indicator": indicator,
                "zscore": round(float(zscore), 4),
                "logic": "hot_inflation→short_bonds" if zscore > 0 else "cool_inflation→long_bonds",
                "strategy": self.name,
            },
        }


class CompositeStrategy(BaseStrategy):
    """
    Combines multiple sub-strategies and takes the highest-confidence signal.

    Useful for running a portfolio of strategies simultaneously.
    """

    def __init__(
        self,
        name: str = "composite",
        strategies: list[BaseStrategy] | None = None,
        min_confidence: float = 0.3,
        entry_offset_minutes: int = 1,
        exit_offset_minutes: int = 30,
    ):
        super().__init__(name, entry_offset_minutes, exit_offset_minutes)
        self.strategies = strategies or []
        self.min_confidence = min_confidence

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a sub-strategy."""
        self.strategies.append(strategy)

    def generate_signal(
        self, release: pd.Series
    ) -> dict[str, Any] | None:
        """Pick the highest-confidence signal from all sub-strategies."""
        best_signal = None
        best_confidence = self.min_confidence

        for strategy in self.strategies:
            signal = strategy.generate_signal(release)
            if signal and signal.get("confidence", 0) > best_confidence:
                best_signal = signal
                best_confidence = signal["confidence"]

        return best_signal
