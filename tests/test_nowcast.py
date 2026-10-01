"""
Tests for Phase 3 — Nowcasting Models.

Tests the bridge equation, Kalman filter, surprise index,
and CPI-specific nowcasters using synthetic data.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.models.nowcast.base_nowcast import BaseNowcaster, NowcastResult
from src.models.nowcast.bridge_model import BridgeEquationNowcaster
from src.models.nowcast.cpi_nowcast import CPINowcaster, NFPNowcaster
from src.models.nowcast.kalman_nowcast import KalmanNowcaster
from src.models.surprise_index import SurpriseIndex, compute_surprise_zscore


# ---------------------------------------------------------------------------
# Fixtures: synthetic macro data
# ---------------------------------------------------------------------------


def _make_dates(n: int = 60, freq: str = "MS") -> pd.DatetimeIndex:
    """Generate monthly dates."""
    return pd.date_range("2019-01-01", periods=n, freq=freq)


def _make_target_series(n: int = 60, seed: int = 42) -> pd.Series:
    """
    Synthetic CPI-like MoM% series.
    Random walk with drift + noise, roughly 0.1-0.5% MoM.
    """
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)
    values = 0.2 + 0.1 * np.cumsum(rng.standard_normal(n) * 0.05) + rng.standard_normal(n) * 0.1
    return pd.Series(values, index=dates, name="cpi_mom")


def _make_proxy_data(
    n: int = 60, n_proxies: int = 3, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """
    Synthetic proxy data correlated with the target.
    Each proxy = target + noise (different noise levels).
    """
    rng = np.random.default_rng(seed)
    dates = _make_dates(n)
    target = _make_target_series(n, seed)

    proxies = {}
    for i in range(n_proxies):
        name = f"proxy_{i}"
        # Proxy = target * weight + noise
        noise = rng.standard_normal(n) * (0.1 + i * 0.05)
        values = target.values * (0.8 + i * 0.1) + noise + rng.uniform(-0.5, 0.5)
        df = pd.DataFrame({"value": values}, index=dates)
        proxies[name] = df

    return proxies


# ===========================================================================
# BridgeEquationNowcaster Tests
# ===========================================================================


class TestBridgeEquation:
    """Tests for the bridge equation (Ridge regression) nowcaster."""

    def test_fit_and_predict(self):
        """Model fits and produces a valid prediction."""
        target = _make_target_series(48)
        proxies = _make_proxy_data(48)

        model = BridgeEquationNowcaster(
            name="test_bridge",
            indicator="CPI",
            alpha=1.0,
            max_lag=2,
            min_training_samples=12,
        )

        features = model.build_feature_matrix(target, proxies, max_lag=2)
        model.fit(target, features)

        assert model.is_fitted
        result = model.predict(proxies, target.index[-1] + timedelta(days=30))
        assert isinstance(result, NowcastResult)
        assert result.confidence_lower < result.point_estimate < result.confidence_upper

    def test_insufficient_data_raises(self):
        """Fitting with too few samples raises ValueError."""
        target = _make_target_series(5)
        proxies = _make_proxy_data(5)

        model = BridgeEquationNowcaster(
            name="test", indicator="CPI", min_training_samples=12
        )
        features = model.build_feature_matrix(target, proxies)

        with pytest.raises(ValueError, match="Need at least"):
            model.fit(target, features)

    def test_predict_before_fit_raises(self):
        """Predicting without fitting raises RuntimeError."""
        model = BridgeEquationNowcaster(name="test", indicator="CPI")
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict({}, datetime(2024, 6, 1))

    def test_feature_importance(self):
        """Feature importance returns sorted coefficients."""
        target = _make_target_series(48)
        proxies = _make_proxy_data(48)

        model = BridgeEquationNowcaster(
            name="test", indicator="CPI", min_training_samples=12
        )
        features = model.build_feature_matrix(target, proxies)
        model.fit(target, features)

        importance = model.get_feature_importance()
        assert len(importance) > 0
        assert "coefficient" in importance.columns
        assert importance["abs_importance"].is_monotonic_decreasing

    def test_walk_forward_evaluation(self):
        """Walk-forward evaluation produces valid metrics."""
        target = _make_target_series(60)
        proxies = _make_proxy_data(60)

        model = BridgeEquationNowcaster(
            name="test", indicator="CPI", min_training_samples=12
        )
        features = model.build_feature_matrix(target, proxies)

        metrics = model.walk_forward_evaluate(
            target, features, initial_window=24, step=1
        )

        assert metrics.n_predictions > 0
        assert metrics.mae >= 0
        assert metrics.rmse >= 0
        assert 0 <= metrics.directional_accuracy <= 1
        assert 0 <= metrics.coverage_rate <= 1
        assert len(metrics.predictions) == metrics.n_predictions

    def test_build_feature_matrix_shape(self):
        """Feature matrix has correct shape and columns."""
        target = _make_target_series(36)
        proxies = _make_proxy_data(36, n_proxies=2)

        features = BridgeEquationNowcaster.build_feature_matrix(
            target, proxies, max_lag=2, use_mom=True
        )

        assert len(features) == len(target)
        # Each proxy: level + mom + pct + lag1 + lag2 = 5 features
        # 2 proxies × 5 = 10
        assert features.shape[1] == 10

    def test_no_future_leakage_in_features(self):
        """Features at time t only use data available at or before t."""
        target = _make_target_series(36)
        proxies = _make_proxy_data(36)

        model = BridgeEquationNowcaster(name="test", indicator="CPI")

        # Get features for a specific date
        as_of = target.index[20]
        features = model._get_features(proxies, as_of)

        # All proxy data should be <= as_of
        for proxy_name, df in proxies.items():
            available = df[df.index <= pd.Timestamp(as_of)]
            latest_proxy = float(available["value"].iloc[-1])
            feat_key = f"{proxy_name}_level"
            if feat_key in features.columns:
                assert features[feat_key].iloc[0] == pytest.approx(latest_proxy)


# ===========================================================================
# KalmanNowcaster Tests
# ===========================================================================


class TestKalmanFilter:
    """Tests for the Kalman filter nowcaster."""

    def test_fit_and_predict(self):
        """Kalman filter fits and predicts."""
        target = _make_target_series(36)
        features = pd.DataFrame(index=target.index)  # No exog

        model = KalmanNowcaster(
            name="test_kalman",
            indicator="CPI",
            process_noise=0.01,
            observation_noise=0.1,
            min_training_samples=12,
        )

        model.fit(target, features)
        assert model.is_fitted

        result = model.predict({}, target.index[-1] + timedelta(days=30))
        assert isinstance(result, NowcastResult)
        assert result.confidence_lower < result.point_estimate < result.confidence_upper

    def test_with_trend(self):
        """Kalman filter with trend component captures upward drift."""
        rng = np.random.default_rng(42)
        dates = _make_dates(48)
        # Create trending series
        trend = np.linspace(0, 2, 48) + rng.standard_normal(48) * 0.1
        target = pd.Series(trend, index=dates)

        model = KalmanNowcaster(
            name="trending", indicator="CPI", use_trend=True, min_training_samples=12
        )
        features = pd.DataFrame(index=target.index)
        model.fit(target, features)

        state = model.current_state
        assert "trend" in state
        assert state["trend"] > 0  # Should detect positive trend

    def test_without_trend(self):
        """Kalman filter without trend component works."""
        target = _make_target_series(36)
        features = pd.DataFrame(index=target.index)

        model = KalmanNowcaster(
            name="no_trend", indicator="CPI", use_trend=False, min_training_samples=12
        )
        model.fit(target, features)

        state = model.current_state
        assert "level" in state
        assert "trend" not in state

    def test_online_update(self):
        """Online update changes the state."""
        target = _make_target_series(24)
        features = pd.DataFrame(index=target.index)

        model = KalmanNowcaster(
            name="test", indicator="CPI", min_training_samples=12
        )
        model.fit(target, features)

        state_before = model.current_state["level"]
        model.update(state_before + 5.0)  # Big positive surprise
        state_after = model.current_state["level"]

        # State should move toward the new observation
        assert state_after > state_before

    def test_walk_forward(self):
        """Walk-forward evaluation with Kalman filter."""
        target = _make_target_series(48)
        features = pd.DataFrame(index=target.index)

        model = KalmanNowcaster(
            name="test", indicator="CPI", min_training_samples=12
        )

        metrics = model.walk_forward_evaluate(
            target, features, initial_window=24, step=2
        )

        assert metrics.n_predictions > 0
        assert metrics.mae >= 0

    def test_with_exogenous_features(self):
        """Kalman filter with exogenous proxy inputs."""
        target = _make_target_series(36)
        proxies = _make_proxy_data(36, n_proxies=2)

        # Build simple feature matrix
        features = pd.DataFrame(index=target.index)
        for name, df in proxies.items():
            features[f"{name}_level"] = df["value"]

        model = KalmanNowcaster(
            name="test_exog", indicator="CPI", min_training_samples=12
        )
        model.fit(target, features)

        assert model.is_fitted
        result = model.predict(proxies, target.index[-1] + timedelta(days=30))
        assert isinstance(result, NowcastResult)


# ===========================================================================
# SurpriseIndex Tests
# ===========================================================================


class TestSurpriseIndex:
    """Tests for the surprise index calculator."""

    def _make_release_history(self, n: int = 30, seed: int = 42) -> pd.DataFrame:
        """Create synthetic release history."""
        rng = np.random.default_rng(seed)
        dates = _make_dates(n)
        actuals = 0.2 + rng.standard_normal(n) * 0.15
        # Consensus = actual + noise (sometimes right, sometimes wrong)
        consensus = actuals + rng.standard_normal(n) * 0.1

        return pd.DataFrame(
            {
                "release_date": dates,
                "actual": actuals,
                "consensus": consensus,
            }
        )

    def test_compute_single_surprise(self):
        """Compute surprise for a single release."""
        idx = SurpriseIndex(window=20, significance_threshold=1.5)
        history = self._make_release_history()

        result = idx.compute_surprise(
            actual=0.5,
            consensus=0.3,
            history=history[["actual", "consensus"]],
        )

        assert result.surprise_raw == pytest.approx(0.2, abs=1e-6)
        assert isinstance(result.surprise_zscore, float)
        assert 0 <= result.surprise_percentile <= 100

    def test_positive_surprise(self):
        """Positive surprise when actual > consensus."""
        idx = SurpriseIndex()
        result = idx.compute_surprise(actual=1.0, consensus=0.5)
        assert result.surprise_raw > 0
        assert result.direction == "ABOVE"

    def test_negative_surprise(self):
        """Negative surprise when actual < consensus."""
        idx = SurpriseIndex()
        result = idx.compute_surprise(actual=0.3, consensus=0.5)
        assert result.surprise_raw < 0
        assert result.direction == "BELOW"

    def test_inline_surprise(self):
        """Inline when actual == consensus."""
        idx = SurpriseIndex()
        result = idx.compute_surprise(actual=0.5, consensus=0.5)
        assert result.surprise_raw == 0
        assert result.direction == "INLINE"

    def test_significance_detection(self):
        """Significant surprise detected when |Z| > threshold."""
        idx = SurpriseIndex(significance_threshold=1.0)

        # Create history with small surprises
        rng = np.random.default_rng(42)
        history = pd.DataFrame(
            {
                "actual": np.ones(30) * 0.2 + rng.standard_normal(30) * 0.01,
                "consensus": np.ones(30) * 0.2,
            }
        )

        # Big surprise relative to history
        result = idx.compute_surprise(actual=0.5, consensus=0.2, history=history)
        assert result.is_significant

    def test_compute_series(self):
        """Compute Z-scores for a full series of releases."""
        idx = SurpriseIndex(window=10, min_observations=3)
        releases = self._make_release_history(30)

        result = idx.compute_series(releases)

        assert "surprise_raw" in result.columns
        assert "surprise_zscore" in result.columns
        assert "is_significant" in result.columns
        assert len(result) == 30
        # First min_observations should be NaN
        assert pd.isna(result["surprise_zscore"].iloc[0])
        # Later values should be computed
        assert pd.notna(result["surprise_zscore"].iloc[-1])

    def test_composite_index(self):
        """Composite index across multiple indicators."""
        idx = SurpriseIndex()

        cpi_releases = self._make_release_history(20, seed=1)
        cpi_releases = idx.compute_series(cpi_releases)

        nfp_releases = self._make_release_history(20, seed=2)
        nfp_releases = idx.compute_series(nfp_releases)

        composite = idx.composite_index(
            {"CPI": cpi_releases, "NFP": nfp_releases},
            weights={"CPI": 0.6, "NFP": 0.4},
        )

        assert "composite_zscore" in composite.columns
        assert len(composite) > 0

    def test_decay_weighted_zscore(self):
        """Exponential decay gives more weight to recent surprises."""
        idx_no_decay = SurpriseIndex(window=20, decay_factor=0.0)
        idx_with_decay = SurpriseIndex(window=20, decay_factor=0.2)

        history = self._make_release_history(30)

        r1 = idx_no_decay.compute_surprise(0.5, 0.3, history[["actual", "consensus"]])
        r2 = idx_with_decay.compute_surprise(0.5, 0.3, history[["actual", "consensus"]])

        # Different decay should give different Z-scores
        assert r1.surprise_zscore != pytest.approx(r2.surprise_zscore, abs=1e-6)

    def test_convenience_function(self):
        """Test the standalone compute_surprise_zscore function."""
        rng = np.random.default_rng(42)
        history = pd.Series(rng.standard_normal(30) * 0.1)

        z = compute_surprise_zscore(
            actual=0.5,
            consensus=0.2,
            historical_surprises=history,
            window=20,
        )

        assert isinstance(z, float)
        assert abs(z) > 0  # Should be nonzero with these values


# ===========================================================================
# CPINowcaster Tests
# ===========================================================================


class TestCPINowcaster:
    """Tests for the CPI-specific nowcaster."""

    def test_mom_mode(self):
        """CPI nowcaster in month-over-month mode."""
        rng = np.random.default_rng(42)
        dates = _make_dates(48)

        # Simulate CPI index (level, around 300)
        cpi_index = pd.Series(
            300 + np.cumsum(rng.standard_normal(48) * 0.3), index=dates
        )

        model = CPINowcaster(mode="mom")
        target = model.prepare_target(cpi_index)

        assert len(target) == 47  # One less due to pct_change
        assert all(abs(v) < 5 for v in target.values)  # MoM should be small %

    def test_yoy_mode(self):
        """CPI nowcaster in year-over-year mode."""
        dates = _make_dates(48)
        rng = np.random.default_rng(42)

        cpi_index = pd.Series(
            300 + np.cumsum(rng.standard_normal(48) * 0.3), index=dates
        )

        model = CPINowcaster(mode="yoy")
        target = model.prepare_target(cpi_index)

        assert len(target) == 36  # 48 - 12 = 36

    def test_transform_mom(self):
        """MoM transformation is correct."""
        series = pd.Series([100, 101, 102.5, 103])
        mom = CPINowcaster.transform_to_mom(series)

        assert mom.iloc[1] == pytest.approx(1.0)  # 1% increase
        assert mom.iloc[2] == pytest.approx(1.4851, abs=1e-3)

    def test_transform_yoy(self):
        """YoY transformation is correct."""
        dates = _make_dates(24)
        series = pd.Series(range(100, 124), index=dates, dtype=float)
        yoy = CPINowcaster.transform_to_yoy(series)

        # YoY at month 12: (112 - 100) / 100 * 100 = 12%
        assert yoy.iloc[12] == pytest.approx(12.0, abs=0.1)

    def test_proxy_series_defined(self):
        """CPI nowcaster has proxy series configured."""
        model = CPINowcaster()
        assert "gasoline" in model.proxy_series
        assert "shelter" in model.proxy_series


# ===========================================================================
# NFPNowcaster Tests
# ===========================================================================


class TestNFPNowcaster:
    """Tests for the NFP-specific nowcaster."""

    def test_prepare_target(self):
        """NFP target is month-over-month change."""
        series = pd.Series([150000, 150200, 150350, 150100])
        target = NFPNowcaster.prepare_target(series)

        assert len(target) == 3
        assert target.iloc[0] == pytest.approx(200)
        assert target.iloc[1] == pytest.approx(150)
        assert target.iloc[2] == pytest.approx(-250)

    def test_proxy_series_defined(self):
        """NFP nowcaster has proxy series configured."""
        model = NFPNowcaster()
        assert "initial_claims" in model.proxy_series
        assert "unemployment_rate" in model.proxy_series


# ===========================================================================
# BaseNowcaster Interface Tests
# ===========================================================================


class TestBaseNowcasterInterface:
    """Tests for the abstract base class interface."""

    def test_cannot_instantiate_abstract(self):
        """Cannot instantiate BaseNowcaster directly."""
        with pytest.raises(TypeError):
            BaseNowcaster(name="test", indicator="CPI")  # type: ignore

    def test_nowcast_result_properties(self):
        """NowcastResult computes derived properties."""
        result = NowcastResult(
            indicator="CPI",
            target_date=datetime(2024, 6, 1),
            estimated_at=datetime(2024, 5, 28),
            point_estimate=0.3,
            confidence_lower=0.1,
            confidence_upper=0.5,
            model_name="test",
        )

        assert result.confidence_width == pytest.approx(0.4)
        assert result.point_estimate == 0.3

    def test_evaluation_metrics_summary(self):
        """EvaluationMetrics summary returns clean dict."""
        from src.models.nowcast.base_nowcast import EvaluationMetrics

        metrics = EvaluationMetrics(
            mae=0.1234,
            rmse=0.1567,
            mape=5.678,
            directional_accuracy=0.65,
            n_predictions=36,
            mean_confidence_width=0.45,
            coverage_rate=0.89,
            predictions=pd.DataFrame(),
        )

        summary = metrics.summary()
        assert summary["mae"] == 0.1234
        assert summary["n_predictions"] == 36
        assert isinstance(summary, dict)
