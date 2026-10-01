"""
Bridge Equation Nowcaster.

Uses OLS regression with high-frequency proxy indicators to nowcast
a target macro release before it's officially published.

Bridge equations are the workhorse of central bank nowcasting:
    target_t = β₀ + β₁·proxy1_t + β₂·proxy2_t + ... + ε

The "bridge" connects monthly/quarterly targets to higher-frequency
proxies that are available earlier (e.g., weekly gas prices → monthly CPI).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import Ridge

from src.models.nowcast.base_nowcast import BaseNowcaster

logger = structlog.get_logger(__name__)


class BridgeEquationNowcaster(BaseNowcaster):
    """
    OLS/Ridge bridge equation model for macro nowcasting.

    Features:
        - Ridge regression (L2 regularization) to handle correlated proxies
        - Automatic lag selection for each proxy series
        - Expanding window residual-based confidence intervals
        - Month-over-month and level features
    """

    def __init__(
        self,
        name: str,
        indicator: str,
        alpha: float = 1.0,
        max_lag: int = 3,
        use_mom_features: bool = True,
        confidence_level: float = 0.90,
        min_training_samples: int = 24,
    ):
        """
        Args:
            name: Model identifier.
            indicator: Target indicator (e.g., 'CPI').
            alpha: Ridge regularization strength.
            max_lag: Maximum lag months for proxy features.
            use_mom_features: If True, include month-over-month changes.
            confidence_level: CI coverage level.
            min_training_samples: Minimum training observations.
        """
        super().__init__(
            name=name,
            indicator=indicator,
            confidence_level=confidence_level,
            min_training_samples=min_training_samples,
        )
        self.alpha = alpha
        self.max_lag = max_lag
        self.use_mom_features = use_mom_features

        self._model: Ridge | None = None
        self._residual_std: float = 1.0
        self._feature_names: list[str] = []

    def _fit_impl(
        self,
        target: pd.Series,
        features: pd.DataFrame,
    ) -> None:
        """Fit Ridge regression on training data."""
        # Drop any rows with NaN
        combined = features.copy()
        combined["_target"] = target
        combined = combined.dropna()

        if len(combined) < self.min_training_samples:
            raise ValueError(
                f"After dropping NaN, only {len(combined)} samples remain "
                f"(need {self.min_training_samples})"
            )

        X = combined.drop(columns=["_target"]).values
        y = combined["_target"].values

        self._model = Ridge(alpha=self.alpha, fit_intercept=True)
        self._model.fit(X, y)

        # Store residual std for confidence intervals
        y_pred = self._model.predict(X)
        residuals = y - y_pred
        self._residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0
        self._feature_names = list(features.columns)

        logger.debug(
            "bridge_fitted",
            model=self.name,
            n_features=X.shape[1],
            residual_std=round(self._residual_std, 4),
            r_squared=round(float(self._model.score(X, y)), 4),
        )

    def _predict_impl(
        self,
        features: pd.DataFrame,
    ) -> tuple[float, float, float]:
        """Generate prediction with residual-based confidence interval."""
        if self._model is None:
            raise RuntimeError("Model not fitted")

        X = features.values
        point = float(self._model.predict(X)[0])

        # Confidence interval from residual standard deviation
        from scipy import stats

        z = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin = z * self._residual_std

        return point, point - margin, point + margin

    def _get_features(
        self,
        proxy_data: dict[str, pd.DataFrame],
        as_of_date: datetime,
    ) -> pd.DataFrame:
        """
        Build feature matrix from proxy data.

        For each proxy:
            - Level value at as_of_date (or latest available before)
            - Month-over-month change (if use_mom_features=True)
            - Lagged values up to max_lag
        """
        feature_dict: dict[str, float] = {}

        for proxy_name, df in proxy_data.items():
            if df.empty:
                continue

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" in df.columns:
                    df = df.set_index("date")
                df.index = pd.to_datetime(df.index)

            # Filter to data available before as_of_date (no future leak)
            ts = pd.Timestamp(as_of_date)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            available = df[df.index <= ts]
            if available.empty:
                continue

            # Get the value column
            value_col = "value" if "value" in available.columns else available.columns[0]
            series = available[value_col].dropna()

            if series.empty:
                continue

            # Current level
            latest_val = float(series.iloc[-1])
            feature_dict[f"{proxy_name}_level"] = latest_val

            # Month-over-month change
            if self.use_mom_features and len(series) >= 2:
                mom_change = latest_val - float(series.iloc[-2])
                feature_dict[f"{proxy_name}_mom"] = mom_change

                # Percent change
                prev = float(series.iloc[-2])
                if prev != 0:
                    feature_dict[f"{proxy_name}_pct"] = (mom_change / abs(prev)) * 100

            # Lagged values
            for lag in range(1, min(self.max_lag + 1, len(series))):
                feature_dict[f"{proxy_name}_lag{lag}"] = float(series.iloc[-(lag + 1)])

        if not feature_dict:
            raise ValueError(f"No features could be extracted for {as_of_date}")

        return pd.DataFrame([feature_dict], index=[as_of_date])

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (regression coefficients).

        Returns:
            DataFrame with feature names and their coefficients,
            sorted by absolute importance.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted")

        importance = pd.DataFrame(
            {
                "feature": self._feature_names,
                "coefficient": self._model.coef_,
                "abs_importance": np.abs(self._model.coef_),
            }
        )
        importance = importance.sort_values("abs_importance", ascending=False)
        importance["rank"] = range(1, len(importance) + 1)
        return importance.reset_index(drop=True)

    @staticmethod
    def build_feature_matrix(
        target: pd.Series,
        proxy_data: dict[str, pd.DataFrame],
        max_lag: int = 3,
        use_mom: bool = True,
    ) -> pd.DataFrame:
        """
        Build a complete historical feature matrix aligned with target dates.

        This is a convenience method for training — extracts features
        for every date in the target series.

        Args:
            target: Target series with datetime index.
            proxy_data: Dict of proxy name -> DataFrame with 'value' column.
            max_lag: Max lags to include.
            use_mom: Include month-over-month changes.

        Returns:
            Feature DataFrame aligned with target index.
        """
        all_features: dict[str, list[float | None]] = {}
        dates = target.index

        for proxy_name, df in proxy_data.items():
            if df.empty:
                continue

            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" in df.columns:
                    df = df.set_index("date")
                df.index = pd.to_datetime(df.index)

            value_col = "value" if "value" in df.columns else df.columns[0]
            series = df[value_col].sort_index()

            levels = []
            moms = []
            pcts = []
            lags: dict[int, list[float | None]] = {lag: [] for lag in range(1, max_lag + 1)}

            for dt in dates:
                available = series[series.index <= dt]

                if available.empty:
                    levels.append(None)
                    moms.append(None)
                    pcts.append(None)
                    for lag in range(1, max_lag + 1):
                        lags[lag].append(None)
                    continue

                val = float(available.iloc[-1])
                levels.append(val)

                if len(available) >= 2:
                    prev = float(available.iloc[-2])
                    moms.append(val - prev)
                    pcts.append(((val - prev) / abs(prev)) * 100 if prev != 0 else None)
                else:
                    moms.append(None)
                    pcts.append(None)

                for lag in range(1, max_lag + 1):
                    if len(available) > lag:
                        lags[lag].append(float(available.iloc[-(lag + 1)]))
                    else:
                        lags[lag].append(None)

            all_features[f"{proxy_name}_level"] = levels

            if use_mom:
                all_features[f"{proxy_name}_mom"] = moms
                all_features[f"{proxy_name}_pct"] = pcts

            for lag in range(1, max_lag + 1):
                all_features[f"{proxy_name}_lag{lag}"] = lags[lag]

        return pd.DataFrame(all_features, index=dates)
