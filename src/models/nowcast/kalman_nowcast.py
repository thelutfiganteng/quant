"""
Kalman Filter Nowcaster.

Uses a linear state-space model with Kalman filtering for dynamic
nowcasting. Unlike static regression (bridge equation), the Kalman
filter continuously updates its estimates as new data arrives,
making it ideal for real-time nowcasting.

State-space model:
    x_t = F·x_{t-1} + B·u_t + w_t    (state transition)
    z_t = H·x_t + v_t                  (observation)

Where:
    x_t = latent macro state
    z_t = observed proxy indicators
    F   = state transition matrix
    H   = observation matrix
    w_t ~ N(0, Q)  process noise
    v_t ~ N(0, R)  observation noise
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import structlog

from src.models.nowcast.base_nowcast import BaseNowcaster

logger = structlog.get_logger(__name__)


class KalmanNowcaster(BaseNowcaster):
    """
    Kalman filter-based nowcaster for macro indicators.

    Uses a simple local level + trend model that can be extended
    with proxy indicators as exogenous inputs.

    The Kalman filter provides:
        - Optimal state estimates (BLUE for linear-Gaussian systems)
        - Built-in uncertainty quantification via state covariance
        - Online updates: no need to retrain from scratch
    """

    def __init__(
        self,
        name: str,
        indicator: str,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
        use_trend: bool = True,
        confidence_level: float = 0.90,
        min_training_samples: int = 12,
    ):
        """
        Args:
            name: Model identifier.
            indicator: Target indicator.
            process_noise: Process noise variance (Q diagonal).
            observation_noise: Observation noise variance (R).
            use_trend: Include local trend component.
            confidence_level: CI coverage level.
            min_training_samples: Minimum training observations.
        """
        super().__init__(
            name=name,
            indicator=indicator,
            confidence_level=confidence_level,
            min_training_samples=min_training_samples,
        )
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.use_trend = use_trend

        # State dimension: 1 (level) or 2 (level + trend)
        self._state_dim = 2 if use_trend else 1

        # Kalman state
        self._x: np.ndarray | None = None  # State estimate
        self._P: np.ndarray | None = None  # State covariance
        self._F: np.ndarray | None = None  # Transition matrix
        self._H: np.ndarray | None = None  # Observation matrix
        self._Q: np.ndarray | None = None  # Process noise
        self._R: np.ndarray | None = None  # Observation noise

        # For exogenous features (proxy indicators)
        self._beta: np.ndarray | None = None  # Regression coefficients for proxies
        self._n_exog: int = 0
        self._feature_names: list[str] = []

    def _initialize_matrices(self, initial_value: float) -> None:
        """Initialize Kalman filter matrices."""
        dim = self._state_dim

        # State transition: local level + trend
        if self.use_trend:
            self._F = np.array([[1.0, 1.0], [0.0, 1.0]])
            self._H = np.array([[1.0, 0.0]])
        else:
            self._F = np.array([[1.0]])
            self._H = np.array([[1.0]])

        # Initial state
        self._x = np.zeros(dim)
        self._x[0] = initial_value
        if self.use_trend:
            self._x[1] = 0.0  # Initial trend = 0

        # Covariances
        self._P = np.eye(dim) * 1.0
        self._Q = np.eye(dim) * self.process_noise
        self._R = np.array([[self.observation_noise]])

    def _kalman_predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Kalman predict step: propagate state forward."""
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q
        return x_pred, P_pred

    def _kalman_update(
        self, x_pred: np.ndarray, P_pred: np.ndarray, z: float
    ) -> None:
        """Kalman update step: incorporate new observation."""
        # Innovation
        y = z - (self._H @ x_pred)[0]
        S = (self._H @ P_pred @ self._H.T + self._R)[0, 0]

        # Kalman gain
        K = (P_pred @ self._H.T) / S

        # Update state
        self._x = x_pred + K.flatten() * y
        I = np.eye(self._state_dim)
        self._P = (I - K @ self._H) @ P_pred

    def _fit_impl(
        self,
        target: pd.Series,
        features: pd.DataFrame,
    ) -> None:
        """
        Fit by running Kalman filter through all observations.

        If features are provided, first fit a regression to get
        exogenous coefficients, then run KF on residuals.
        """
        target = target.dropna()
        if len(target) < self.min_training_samples:
            raise ValueError(f"Need {self.min_training_samples} samples")

        values = target.values.astype(float)

        # If features provided, use them as exogenous inputs
        if not features.empty and len(features.columns) > 0:
            combined = features.copy()
            combined["_target"] = target
            combined = combined.dropna()

            if len(combined) >= self.min_training_samples:
                X = combined.drop(columns=["_target"]).values
                y = combined["_target"].values
                self._feature_names = list(features.columns)
                self._n_exog = X.shape[1]

                # Simple OLS for exogenous coefficients
                X_with_intercept = np.column_stack([np.ones(len(X)), X])
                try:
                    self._beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                    values = y - X_with_intercept @ self._beta  # Residuals
                except np.linalg.LinAlgError:
                    self._beta = None
                    self._n_exog = 0

        # Initialize Kalman filter
        self._initialize_matrices(values[0])

        # Run filter through training data
        for i in range(1, len(values)):
            x_pred, P_pred = self._kalman_predict()
            self._kalman_update(x_pred, P_pred, values[i])

        logger.debug(
            "kalman_fitted",
            model=self.name,
            n_obs=len(values),
            state=self._x.tolist(),
            n_exog=self._n_exog,
        )

    def _predict_impl(
        self,
        features: pd.DataFrame,
    ) -> tuple[float, float, float]:
        """Predict next value with Kalman state projection."""
        if self._x is None:
            raise RuntimeError("Kalman filter not initialized")

        # Predict step
        x_pred, P_pred = self._kalman_predict()
        point = float((self._H @ x_pred)[0])

        # Add exogenous component
        if self._beta is not None and not features.empty:
            X = features.values
            if X.shape[1] == self._n_exog:
                X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                exog_pred = float((X_with_intercept @ self._beta)[0])
                point += exog_pred

        # Prediction variance
        pred_var = float((self._H @ P_pred @ self._H.T + self._R)[0, 0])
        pred_std = np.sqrt(pred_var)

        from scipy import stats

        z = stats.norm.ppf((1 + self.confidence_level) / 2)
        margin = z * pred_std

        return point, point - margin, point + margin

    def _get_features(
        self,
        proxy_data: dict[str, pd.DataFrame],
        as_of_date: datetime,
    ) -> pd.DataFrame:
        """Extract latest proxy values as features."""
        feature_dict: dict[str, float] = {}

        for proxy_name, df in proxy_data.items():
            if df.empty:
                continue

            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" in df.columns:
                    df = df.set_index("date")
                df.index = pd.to_datetime(df.index)

            available = df[df.index <= pd.Timestamp(as_of_date)]
            if available.empty:
                continue

            value_col = "value" if "value" in available.columns else available.columns[0]
            series = available[value_col].dropna()

            if not series.empty:
                feature_dict[f"{proxy_name}_level"] = float(series.iloc[-1])

                if len(series) >= 2:
                    feature_dict[f"{proxy_name}_mom"] = (
                        float(series.iloc[-1]) - float(series.iloc[-2])
                    )

        if not feature_dict:
            return pd.DataFrame(index=[as_of_date])

        return pd.DataFrame([feature_dict], index=[as_of_date])

    def update(self, observation: float) -> None:
        """
        Online update: incorporate a single new observation.

        This is the key advantage over static regression — the Kalman
        filter can update its state without refitting from scratch.

        Args:
            observation: New observed value.
        """
        if self._x is None:
            raise RuntimeError("Must fit before updating")

        x_pred, P_pred = self._kalman_predict()
        self._kalman_update(x_pred, P_pred, observation)

        logger.debug(
            "kalman_updated",
            model=self.name,
            observation=observation,
            new_state=self._x.tolist(),
        )

    @property
    def current_state(self) -> dict[str, float]:
        """Get current Kalman state as a readable dict."""
        if self._x is None:
            return {}

        state = {"level": float(self._x[0])}
        if self.use_trend and len(self._x) > 1:
            state["trend"] = float(self._x[1])
        return state
