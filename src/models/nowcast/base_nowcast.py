"""
Abstract base class for all nowcasting models.

Every nowcast model must implement fit(), predict(), and evaluate().
Walk-forward validation is built into the base class to prevent
time-series leakage — no random train/test splits allowed.

Design Principles:
    - Walk-forward only: train on [0..t], predict t+1
    - Expanding window: each step adds data, never removes
    - Feature snapshots: store which inputs drove each prediction
    - Confidence intervals: every estimate must include uncertainty bounds
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result Containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NowcastResult:
    """Single nowcast prediction result."""

    indicator: str
    target_date: datetime
    estimated_at: datetime
    point_estimate: float
    confidence_lower: float
    confidence_upper: float
    model_name: str
    features_used: dict[str, float] = field(default_factory=dict)

    @property
    def confidence_width(self) -> float:
        """Width of the confidence interval."""
        return self.confidence_upper - self.confidence_lower


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for walk-forward validation."""

    mae: float
    rmse: float
    mape: float | None  # None if any actual is zero
    directional_accuracy: float  # % of correct direction predictions
    n_predictions: int
    mean_confidence_width: float
    coverage_rate: float  # % of actuals within CI
    predictions: pd.DataFrame  # Full prediction vs actual table

    def summary(self) -> dict[str, Any]:
        """Return metrics as a flat dict for logging/display."""
        return {
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "mape": round(self.mape, 4) if self.mape is not None else None,
            "directional_accuracy": round(self.directional_accuracy, 4),
            "n_predictions": self.n_predictions,
            "mean_ci_width": round(self.mean_confidence_width, 4),
            "coverage_rate": round(self.coverage_rate, 4),
        }


# ---------------------------------------------------------------------------
# Abstract Base Nowcaster
# ---------------------------------------------------------------------------


class BaseNowcaster(ABC):
    """
    Abstract base class for macro nowcasting models.

    Subclasses must implement:
        - _fit_impl(): Train the model on historical data
        - _predict_impl(): Generate a point estimate + confidence interval
        - _get_features(): Extract features for the prediction date

    The base class handles:
        - Walk-forward validation (expanding window)
        - Metric computation (MAE, RMSE, directional accuracy)
        - Feature snapshot logging
        - Confidence interval calibration tracking
    """

    def __init__(
        self,
        name: str,
        indicator: str,
        confidence_level: float = 0.90,
        min_training_samples: int = 12,
    ):
        self.name = name
        self.indicator = indicator
        self.confidence_level = confidence_level
        self.min_training_samples = min_training_samples

        self._is_fitted = False
        self._training_end: datetime | None = None

        logger.info(
            "nowcaster_initialized",
            model=name,
            indicator=indicator,
            confidence_level=confidence_level,
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    # --- Abstract Methods (must be implemented by subclasses) ---

    @abstractmethod
    def _fit_impl(
        self,
        target: pd.Series,
        features: pd.DataFrame,
    ) -> None:
        """
        Train the model on historical data.

        Args:
            target: Historical actual values indexed by date.
            features: Feature matrix indexed by date, aligned with target.
        """
        ...

    @abstractmethod
    def _predict_impl(
        self,
        features: pd.DataFrame,
    ) -> tuple[float, float, float]:
        """
        Generate a prediction.

        Args:
            features: Feature row(s) for the prediction date.

        Returns:
            (point_estimate, confidence_lower, confidence_upper)
        """
        ...

    @abstractmethod
    def _get_features(
        self,
        proxy_data: dict[str, pd.DataFrame],
        as_of_date: datetime,
    ) -> pd.DataFrame:
        """
        Extract features from proxy data for a specific date.

        Args:
            proxy_data: Dict of proxy series name -> DataFrame.
            as_of_date: Date to compute features for (no future data allowed).

        Returns:
            Feature DataFrame with one row per target date.
        """
        ...

    # --- Public API ---

    def fit(
        self,
        target: pd.Series,
        features: pd.DataFrame,
    ) -> None:
        """
        Fit the model on training data.

        Args:
            target: Historical actual values.
            features: Feature matrix aligned with target.

        Raises:
            ValueError: If insufficient training samples.
        """
        if len(target) < self.min_training_samples:
            raise ValueError(
                f"Need at least {self.min_training_samples} samples, got {len(target)}"
            )

        # Ensure alignment
        common_idx = target.index.intersection(features.index)
        target = target.loc[common_idx]
        features = features.loc[common_idx]

        logger.info(
            "nowcaster_fitting",
            model=self.name,
            n_samples=len(target),
            features=list(features.columns),
        )

        self._fit_impl(target, features)
        self._is_fitted = True
        self._training_end = target.index.max()

        logger.info(
            "nowcaster_fitted",
            model=self.name,
            training_end=str(self._training_end),
        )

    def predict(
        self,
        proxy_data: dict[str, pd.DataFrame],
        target_date: datetime,
    ) -> NowcastResult:
        """
        Generate a nowcast for a target release date.

        Args:
            proxy_data: Dict of proxy series (e.g., gasoline prices).
            target_date: The upcoming release date to nowcast.

        Returns:
            NowcastResult with point estimate and confidence interval.

        Raises:
            RuntimeError: If model is not fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Model {self.name} must be fitted before prediction")

        features = self._get_features(proxy_data, as_of_date=target_date)
        point, lower, upper = self._predict_impl(features)

        # Snapshot features for reproducibility
        feature_snapshot = {}
        if len(features) > 0:
            last_row = features.iloc[-1]
            feature_snapshot = {
                col: round(float(val), 4)
                for col, val in last_row.items()
                if pd.notna(val)
            }

        result = NowcastResult(
            indicator=self.indicator,
            target_date=target_date,
            estimated_at=datetime.now(tz=__import__('datetime').timezone.utc),
            point_estimate=round(point, 4),
            confidence_lower=round(lower, 4),
            confidence_upper=round(upper, 4),
            model_name=self.name,
            features_used=feature_snapshot,
        )

        logger.info(
            "nowcast_generated",
            model=self.name,
            indicator=self.indicator,
            target_date=str(target_date),
            estimate=result.point_estimate,
            ci=f"[{result.confidence_lower}, {result.confidence_upper}]",
        )

        return result

    def walk_forward_evaluate(
        self,
        target: pd.Series,
        features: pd.DataFrame,
        initial_window: int | None = None,
        step: int = 1,
    ) -> EvaluationMetrics:
        """
        Walk-forward (expanding window) cross-validation.

        CRITICAL: This is the ONLY valid evaluation method for time series.
        Never use random train/test split — it causes data leakage.

        Process:
            1. Train on [0 .. initial_window]
            2. Predict sample at initial_window + 1
            3. Expand window by `step`, retrain, predict next
            4. Repeat until end of data

        Args:
            target: Full target series.
            features: Full feature matrix.
            initial_window: Starting training size (default: min_training_samples).
            step: How many samples to advance each iteration.

        Returns:
            EvaluationMetrics with all walk-forward results.
        """
        if initial_window is None:
            initial_window = self.min_training_samples

        # Ensure alignment
        common_idx = target.index.intersection(features.index)
        target = target.loc[common_idx].sort_index()
        features = features.loc[common_idx].sort_index()

        n = len(target)
        if n <= initial_window:
            raise ValueError(
                f"Need more data than initial_window ({initial_window}), got {n}"
            )

        predictions = []
        actuals = []
        ci_lowers = []
        ci_uppers = []
        dates = []

        logger.info(
            "walk_forward_starting",
            model=self.name,
            n_total=n,
            initial_window=initial_window,
            n_predictions=len(range(initial_window, n, step)),
        )

        for t in range(initial_window, n, step):
            # Train on [0..t)
            train_target = target.iloc[:t]
            train_features = features.iloc[:t]

            # Predict at t
            test_features = features.iloc[t : t + 1]
            actual = target.iloc[t]

            try:
                self._fit_impl(train_target, train_features)
                self._is_fitted = True
                point, lower, upper = self._predict_impl(test_features)

                predictions.append(point)
                actuals.append(actual)
                ci_lowers.append(lower)
                ci_uppers.append(upper)
                dates.append(target.index[t])

            except Exception as e:
                logger.warning(
                    "walk_forward_step_failed",
                    model=self.name,
                    step=t,
                    error=str(e),
                )
                continue

        if not predictions:
            raise ValueError("Walk-forward produced no valid predictions")

        return self._compute_metrics(
            dates=dates,
            actuals=actuals,
            predictions=predictions,
            ci_lowers=ci_lowers,
            ci_uppers=ci_uppers,
        )

    def _compute_metrics(
        self,
        dates: list,
        actuals: list[float],
        predictions: list[float],
        ci_lowers: list[float],
        ci_uppers: list[float],
    ) -> EvaluationMetrics:
        """Compute evaluation metrics from walk-forward results."""
        actuals_arr = np.array(actuals)
        preds_arr = np.array(predictions)
        errors = actuals_arr - preds_arr

        # MAE, RMSE
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))

        # MAPE (avoid division by zero)
        if np.all(actuals_arr != 0):
            mape = float(np.mean(np.abs(errors / actuals_arr))) * 100
        else:
            mape = None

        # Directional accuracy (did we predict the direction of change?)
        if len(actuals_arr) > 1:
            actual_changes = np.diff(actuals_arr)
            pred_changes = np.diff(preds_arr)
            correct_dir = np.sum(np.sign(actual_changes) == np.sign(pred_changes))
            dir_accuracy = float(correct_dir / len(actual_changes))
        else:
            dir_accuracy = 0.0

        # Confidence interval metrics
        ci_lowers_arr = np.array(ci_lowers)
        ci_uppers_arr = np.array(ci_uppers)
        ci_widths = ci_uppers_arr - ci_lowers_arr
        mean_ci_width = float(np.mean(ci_widths))

        # Coverage: % of actuals within CI
        within_ci = np.sum(
            (actuals_arr >= ci_lowers_arr) & (actuals_arr <= ci_uppers_arr)
        )
        coverage = float(within_ci / len(actuals_arr))

        # Build results DataFrame
        results_df = pd.DataFrame(
            {
                "date": dates,
                "actual": actuals,
                "predicted": predictions,
                "error": errors.tolist(),
                "ci_lower": ci_lowers,
                "ci_upper": ci_uppers,
            }
        )

        metrics = EvaluationMetrics(
            mae=mae,
            rmse=rmse,
            mape=mape,
            directional_accuracy=dir_accuracy,
            n_predictions=len(predictions),
            mean_confidence_width=mean_ci_width,
            coverage_rate=coverage,
            predictions=results_df,
        )

        logger.info(
            "walk_forward_complete",
            model=self.name,
            **metrics.summary(),
        )

        return metrics
