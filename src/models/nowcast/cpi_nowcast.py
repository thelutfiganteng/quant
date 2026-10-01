"""
CPI Nowcaster — specialized bridge equation model for CPI.

Uses high-frequency proxy indicators that are released before CPI:
    - Weekly gasoline prices (FRED: GASREGW) → 35% of CPI energy component
    - Used car prices (Manheim index proxy via FRED) → volatile component
    - Shelter CPI (lagged rent data) → 33% of core CPI
    - Food prices (FRED: CPIUFDNS) → food component

Architecture:
    CPI_MoM_t ≈ β₀ + β₁·gas_pct_t + β₂·used_cars_pct_t
                + β₃·shelter_lag_t + β₄·food_pct_t + ε

This is based on the Cleveland Fed's approach to CPI nowcasting.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import structlog

from src.models.nowcast.bridge_model import BridgeEquationNowcaster

logger = structlog.get_logger(__name__)

# FRED series IDs for CPI proxy data
CPI_PROXY_SERIES = {
    "gasoline": "GASREGW",        # Weekly U.S. Regular Gasoline Prices
    "used_cars": "CUSR0000SETA02",  # Used cars and trucks CPI
    "shelter": "CUSR0000SAH1",      # Shelter CPI
    "food": "CPIUFDNS",             # Food CPI (not seasonally adjusted)
    "energy": "CPIENGSL",           # Energy CPI
    "medical": "CPIMEDSL",          # Medical care CPI
    "apparel": "CPIAPPSL",          # Apparel CPI
}

# Component weights (approximate BLS weights)
CPI_WEIGHTS = {
    "shelter": 0.336,     # Largest component
    "food": 0.136,
    "energy": 0.065,      # Volatile
    "medical": 0.088,
    "apparel": 0.025,
    "used_cars": 0.040,   # Volatile
}


class CPINowcaster(BridgeEquationNowcaster):
    """
    CPI-specialized nowcaster using known proxy relationships.

    Extends BridgeEquationNowcaster with:
        - Pre-configured proxy series for CPI components
        - Component-weighted feature construction
        - MoM vs YoY prediction modes
        - Seasonal adjustment handling
    """

    def __init__(
        self,
        mode: str = "mom",
        alpha: float = 0.5,
        max_lag: int = 2,
        confidence_level: float = 0.90,
    ):
        """
        Args:
            mode: 'mom' (month-over-month) or 'yoy' (year-over-year).
            alpha: Ridge regularization.
            max_lag: Maximum lag for proxy features.
            confidence_level: CI coverage.
        """
        super().__init__(
            name=f"cpi_nowcast_{mode}",
            indicator="CPI",
            alpha=alpha,
            max_lag=max_lag,
            use_mom_features=True,
            confidence_level=confidence_level,
            min_training_samples=24,
        )
        self.mode = mode
        self.proxy_series = CPI_PROXY_SERIES.copy()

    @staticmethod
    def transform_to_mom(series: pd.Series) -> pd.Series:
        """Convert level series to month-over-month percent change."""
        return series.pct_change() * 100

    @staticmethod
    def transform_to_yoy(series: pd.Series) -> pd.Series:
        """Convert level series to year-over-year percent change."""
        return series.pct_change(periods=12) * 100

    def prepare_target(self, cpi_series: pd.Series) -> pd.Series:
        """
        Transform raw CPI index to the prediction target.

        Args:
            cpi_series: Raw CPI index values (e.g., 307.5).

        Returns:
            Transformed series (MoM% or YoY%).
        """
        if self.mode == "mom":
            target = self.transform_to_mom(cpi_series)
        elif self.mode == "yoy":
            target = self.transform_to_yoy(cpi_series)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'mom' or 'yoy'.")

        return target.dropna()

    def prepare_proxy_features(
        self,
        proxy_data: dict[str, pd.DataFrame],
        target_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Build the CPI-specific feature matrix from proxy data.

        Transforms raw proxy series to changes and aligns with target dates.

        Args:
            proxy_data: Dict of proxy name -> DataFrame.
            target_index: Target series datetime index.

        Returns:
            Feature DataFrame aligned with target_index.
        """
        return self.build_feature_matrix(
            target=pd.Series(index=target_index, dtype=float),
            proxy_data=proxy_data,
            max_lag=self.max_lag,
            use_mom=self.use_mom_features,
        )

    def nowcast_cpi(
        self,
        cpi_history: pd.Series,
        proxy_data: dict[str, pd.DataFrame],
        target_date: datetime | None = None,
    ) -> dict:
        """
        High-level API: nowcast the next CPI release.

        Args:
            cpi_history: Historical CPI index values.
            proxy_data: Dict of proxy series.
            target_date: Target release date (default: next month).

        Returns:
            Dict with nowcast results and metadata.
        """
        # Transform target
        target = self.prepare_target(cpi_history)

        if target.empty:
            raise ValueError("CPI history too short to compute changes")

        # Build features
        features = self.build_feature_matrix(
            target=target,
            proxy_data=proxy_data,
            max_lag=self.max_lag,
            use_mom=self.use_mom_features,
        )

        # Fit model
        self.fit(target, features)

        # Predict
        if target_date is None:
            # Default: predict next month after last available data
            from dateutil.relativedelta import relativedelta

            target_date = target.index[-1] + relativedelta(months=1)

        result = self.predict(proxy_data, target_date)

        # Get feature importance
        importance = self.get_feature_importance()

        return {
            "nowcast": result,
            "feature_importance": importance,
            "training_samples": len(target),
            "mode": self.mode,
            "proxy_series_used": list(proxy_data.keys()),
        }


class NFPNowcaster(BridgeEquationNowcaster):
    """
    NFP (Non-Farm Payrolls) nowcaster.

    Proxies:
        - ADP Employment Report (released 2 days before NFP)
        - Initial Jobless Claims (weekly, inversely correlated)
        - ISM Employment Index (from PMI survey)
        - Continuing Claims (weekly)
    """

    NFP_PROXY_SERIES = {
        "initial_claims": "ICSA",           # Initial Jobless Claims
        "continuing_claims": "CCSA",        # Continuing Claims
        "employment_ism": "MANEMP",         # ISM Manufacturing Employment
        "adp_private": "NPPTTL",            # ADP National Employment
        "unemployment_rate": "UNRATE",      # Unemployment Rate
        "hours_worked": "AWHMAN",           # Avg Weekly Hours (Manufacturing)
    }

    def __init__(
        self,
        alpha: float = 1.0,
        max_lag: int = 2,
        confidence_level: float = 0.90,
    ):
        super().__init__(
            name="nfp_nowcast",
            indicator="NFP",
            alpha=alpha,
            max_lag=max_lag,
            use_mom_features=True,
            confidence_level=confidence_level,
            min_training_samples=24,
        )
        self.proxy_series = self.NFP_PROXY_SERIES.copy()

    @staticmethod
    def prepare_target(nfp_series: pd.Series) -> pd.Series:
        """
        Transform raw payroll level to monthly change (in thousands).

        NFP is reported as month-over-month change in total payrolls.
        """
        return nfp_series.diff().dropna()
