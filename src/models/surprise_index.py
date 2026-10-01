"""
Surprise Index — Z-Score based surprise measurement.

Measures the "surprise" of an economic release relative to historical
surprise distributions. A Z-score > 2 indicates a significant surprise
that typically moves markets.

    surprise_raw = actual - consensus
    surprise_zscore = (surprise_raw - μ_surprise) / σ_surprise

The rolling window ensures stationarity: what counts as a "big"
surprise adapts over time as market expectations evolve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result Container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurpriseResult:
    """Computed surprise metrics for a single release."""

    indicator: str
    release_date: str
    actual: float
    consensus: float
    surprise_raw: float
    surprise_zscore: float
    surprise_percentile: float
    is_significant: bool  # |z| > threshold
    historical_mean: float
    historical_std: float

    @property
    def direction(self) -> str:
        """Direction of surprise."""
        if self.surprise_raw > 0:
            return "ABOVE"
        elif self.surprise_raw < 0:
            return "BELOW"
        return "INLINE"


# ---------------------------------------------------------------------------
# Surprise Index Calculator
# ---------------------------------------------------------------------------


class SurpriseIndex:
    """
    Computes rolling Z-score surprise index for economic releases.

    Features:
        - Rolling window standardization (adapts to regime changes)
        - Weighted Z-scores (recent surprises matter more)
        - Multi-indicator composite index
        - Significance detection with configurable thresholds
    """

    def __init__(
        self,
        window: int = 20,
        significance_threshold: float = 1.5,
        min_observations: int = 6,
        decay_factor: float = 0.0,
    ):
        """
        Args:
            window: Rolling window for std/mean computation.
            significance_threshold: |Z| above this = significant.
            min_observations: Minimum data points to compute Z-score.
            decay_factor: Exponential decay for weighted Z-scores.
                          0 = equal weights, 0.1 = recent matters more.
        """
        self.window = window
        self.significance_threshold = significance_threshold
        self.min_observations = min_observations
        self.decay_factor = decay_factor

    def compute_surprise(
        self,
        actual: float,
        consensus: float,
        history: pd.DataFrame | None = None,
    ) -> SurpriseResult:
        """
        Compute surprise Z-score for a single release.

        Args:
            actual: Actual released value.
            consensus: Market consensus estimate.
            history: DataFrame with columns ['actual', 'consensus']
                     ordered by date ascending. Used for rolling stats.

        Returns:
            SurpriseResult with raw and standardized surprise.
        """
        surprise_raw = actual - consensus

        if history is not None and len(history) >= self.min_observations:
            hist_surprises = (history["actual"] - history["consensus"]).dropna()

            # Apply rolling window
            if len(hist_surprises) > self.window:
                hist_surprises = hist_surprises.iloc[-self.window:]

            # Apply exponential decay weights if configured
            if self.decay_factor > 0:
                weights = np.exp(
                    -self.decay_factor * np.arange(len(hist_surprises))[::-1]
                )
                weights /= weights.sum()
                hist_mean = float(np.average(hist_surprises.values, weights=weights))
                hist_std = float(
                    np.sqrt(
                        np.average(
                            (hist_surprises.values - hist_mean) ** 2, weights=weights
                        )
                    )
                )
            else:
                hist_mean = float(hist_surprises.mean())
                hist_std = float(hist_surprises.std(ddof=1))

            # Z-score
            if hist_std > 1e-10:
                zscore = (surprise_raw - hist_mean) / hist_std
            else:
                zscore = 0.0

            # Percentile rank
            percentile = float(
                np.mean(hist_surprises.values <= surprise_raw) * 100
            )
        else:
            hist_mean = 0.0
            hist_std = 1.0
            zscore = surprise_raw  # Fallback: raw value as Z-score
            percentile = 50.0

        return SurpriseResult(
            indicator="",
            release_date="",
            actual=actual,
            consensus=consensus,
            surprise_raw=round(surprise_raw, 6),
            surprise_zscore=round(zscore, 4),
            surprise_percentile=round(percentile, 2),
            is_significant=abs(zscore) >= self.significance_threshold,
            historical_mean=round(hist_mean, 6),
            historical_std=round(hist_std, 6),
        )

    def compute_series(
        self,
        releases: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute rolling surprise Z-scores for a series of releases.

        Args:
            releases: DataFrame with columns:
                - 'release_date': datetime
                - 'actual': float
                - 'consensus': float
                Ordered by release_date ascending.

        Returns:
            DataFrame with added columns:
                - 'surprise_raw': actual - consensus
                - 'surprise_zscore': rolling Z-score
                - 'surprise_percentile': percentile rank
                - 'is_significant': bool
        """
        df = releases.copy()
        df = df.sort_values("release_date").reset_index(drop=True)

        # Raw surprise
        df["surprise_raw"] = df["actual"] - df["consensus"]

        # Rolling Z-score
        zscores = []
        percentiles = []
        significants = []

        for i in range(len(df)):
            if i < self.min_observations:
                zscores.append(np.nan)
                percentiles.append(np.nan)
                significants.append(False)
                continue

            # History up to (but not including) current observation
            start_idx = max(0, i - self.window)
            history = df.iloc[start_idx:i]

            current = df.iloc[i]
            result = self.compute_surprise(
                actual=current["actual"],
                consensus=current["consensus"],
                history=history[["actual", "consensus"]],
            )

            zscores.append(result.surprise_zscore)
            percentiles.append(result.surprise_percentile)
            significants.append(result.is_significant)

        df["surprise_zscore"] = zscores
        df["surprise_percentile"] = percentiles
        df["is_significant"] = significants

        logger.info(
            "surprise_series_computed",
            n_releases=len(df),
            n_significant=sum(significants),
            mean_zscore=round(float(np.nanmean(zscores)), 4),
        )

        return df

    def composite_index(
        self,
        indicator_surprises: dict[str, pd.DataFrame],
        weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """
        Compute a weighted composite surprise index across indicators.

        The composite index captures the overall "surprise environment" —
        are releases generally beating or missing expectations?

        Args:
            indicator_surprises: Dict of indicator -> DataFrame with
                                 'release_date' and 'surprise_zscore'.
            weights: Optional indicator weights. Default = equal weight.

        Returns:
            DataFrame with date and composite_zscore columns.
        """
        if weights is None:
            weights = {k: 1.0 / len(indicator_surprises) for k in indicator_surprises}

        # Normalize weights
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}

        # Collect all Z-scores with dates
        all_data = []
        for indicator, df in indicator_surprises.items():
            if "surprise_zscore" not in df.columns:
                continue

            w = weights.get(indicator, 0)
            for _, row in df.iterrows():
                if pd.notna(row.get("surprise_zscore")):
                    all_data.append(
                        {
                            "date": row["release_date"],
                            "indicator": indicator,
                            "zscore": row["surprise_zscore"],
                            "weighted_zscore": row["surprise_zscore"] * w,
                        }
                    )

        if not all_data:
            return pd.DataFrame(columns=["date", "composite_zscore"])

        combined = pd.DataFrame(all_data)
        combined["date"] = pd.to_datetime(combined["date"])

        # Group by date and sum weighted Z-scores
        composite = (
            combined.groupby("date")
            .agg(
                composite_zscore=("weighted_zscore", "sum"),
                n_indicators=("indicator", "nunique"),
                indicators=("indicator", lambda x: list(x)),
            )
            .reset_index()
            .sort_values("date")
        )

        return composite


def compute_surprise_zscore(
    actual: float,
    consensus: float,
    historical_surprises: pd.Series,
    window: int = 20,
) -> float:
    """
    Convenience function for quick Z-score computation.

    Args:
        actual: Released value.
        consensus: Consensus estimate.
        historical_surprises: Series of past (actual - consensus) values.
        window: Rolling window size.

    Returns:
        Z-score of the surprise.
    """
    surprise = actual - consensus
    recent = historical_surprises.iloc[-window:] if len(historical_surprises) > window else historical_surprises
    
    if len(recent) < 3:
        return 0.0

    std = recent.std(ddof=1)
    if std < 1e-10:
        return 0.0

    return float((surprise - recent.mean()) / std)
