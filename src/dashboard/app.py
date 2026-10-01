"""
Dashboard API server using FastAPI.

Provides REST endpoints for:
- Economic data overview (latest releases, calendar)
- Nowcast estimates
- Market data snapshots
- System health / status

This is the main entry point for the dashboard web application.
Run with: uv run python -m src.dashboard.app
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import get_settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Demo data (used when DB is not connected)
# ---------------------------------------------------------------------------

DEMO_RELEASES = [
    {
        "indicator": "CPI",
        "release_date": "2026-09-10T08:30:00",
        "actual": 2.9,
        "consensus": 3.0,
        "previous": 3.0,
        "surprise": -0.1,
        "surprise_zscore": -0.85,
        "source": "BLS",
    },
    {
        "indicator": "NFP",
        "release_date": "2026-09-05T08:30:00",
        "actual": 187,
        "consensus": 175,
        "previous": 206,
        "surprise": 12,
        "surprise_zscore": 0.92,
        "source": "BLS",
    },
    {
        "indicator": "GDP",
        "release_date": "2026-08-29T08:30:00",
        "actual": 2.8,
        "consensus": 2.0,
        "previous": 1.4,
        "surprise": 0.8,
        "surprise_zscore": 2.15,
        "source": "FRED",
    },
    {
        "indicator": "PCE",
        "release_date": "2026-08-30T08:30:00",
        "actual": 2.5,
        "consensus": 2.5,
        "previous": 2.6,
        "surprise": 0.0,
        "surprise_zscore": 0.0,
        "source": "FRED",
    },
    {
        "indicator": "UNEMPLOYMENT_RATE",
        "release_date": "2026-09-05T08:30:00",
        "actual": 4.3,
        "consensus": 4.1,
        "previous": 4.1,
        "surprise": 0.2,
        "surprise_zscore": 1.45,
        "source": "BLS",
    },
    {
        "indicator": "CORE_CPI",
        "release_date": "2026-09-10T08:30:00",
        "actual": 3.2,
        "consensus": 3.2,
        "previous": 3.3,
        "surprise": 0.0,
        "surprise_zscore": 0.0,
        "source": "BLS",
    },
    {
        "indicator": "INITIAL_CLAIMS",
        "release_date": "2026-09-12T08:30:00",
        "actual": 227,
        "consensus": 230,
        "previous": 232,
        "surprise": -3,
        "surprise_zscore": -0.45,
        "source": "BLS",
    },
    {
        "indicator": "RETAIL_SALES",
        "release_date": "2026-09-17T08:30:00",
        "actual": None,
        "consensus": 0.2,
        "previous": 1.0,
        "surprise": None,
        "surprise_zscore": None,
        "source": "FRED",
    },
]

DEMO_CALENDAR = [
    {
        "indicator": "Retail Sales MoM",
        "country": "US",
        "scheduled_date": "2026-09-17T08:30:00",
        "importance": "HIGH",
        "forecast": 0.2,
        "previous": 1.0,
    },
    {
        "indicator": "Industrial Production MoM",
        "country": "US",
        "scheduled_date": "2026-09-17T09:15:00",
        "importance": "MEDIUM",
        "forecast": 0.2,
        "previous": -0.3,
    },
    {
        "indicator": "FOMC Rate Decision",
        "country": "US",
        "scheduled_date": "2026-09-18T14:00:00",
        "importance": "HIGH",
        "forecast": 5.25,
        "previous": 5.50,
    },
    {
        "indicator": "Initial Jobless Claims",
        "country": "US",
        "scheduled_date": "2026-09-19T08:30:00",
        "importance": "MEDIUM",
        "forecast": 228,
        "previous": 227,
    },
    {
        "indicator": "Existing Home Sales",
        "country": "US",
        "scheduled_date": "2026-09-19T10:00:00",
        "importance": "MEDIUM",
        "forecast": 3.90,
        "previous": 3.95,
    },
    {
        "indicator": "Philadelphia Fed Manufacturing",
        "country": "US",
        "scheduled_date": "2026-09-19T08:30:00",
        "importance": "MEDIUM",
        "forecast": -1.0,
        "previous": -7.0,
    },
]

DEMO_NOWCASTS = [
    {
        "indicator": "CPI",
        "target_release_date": "2026-10-10T08:30:00",
        "estimated_at": "2026-09-16T06:00:00",
        "point_estimate": 2.8,
        "confidence_lower": 2.6,
        "confidence_upper": 3.0,
        "model_name": "cpi_bridge_v1",
        "consensus": 2.9,
    },
    {
        "indicator": "NFP",
        "target_release_date": "2026-10-03T08:30:00",
        "estimated_at": "2026-09-16T06:00:00",
        "point_estimate": 165,
        "confidence_lower": 140,
        "confidence_upper": 190,
        "model_name": "nfp_kalman_v1",
        "consensus": 170,
    },
    {
        "indicator": "GDP",
        "target_release_date": "2026-10-30T08:30:00",
        "estimated_at": "2026-09-16T06:00:00",
        "point_estimate": 2.5,
        "confidence_lower": 2.1,
        "confidence_upper": 2.9,
        "model_name": "gdp_bridge_v1",
        "consensus": 2.3,
    },
]

DEMO_SURPRISE_HISTORY = {
    "CPI": [
        {"date": "2026-04-10", "surprise": 0.1, "zscore": 0.8},
        {"date": "2026-05-14", "surprise": -0.2, "zscore": -1.5},
        {"date": "2026-06-12", "surprise": 0.0, "zscore": 0.0},
        {"date": "2026-07-11", "surprise": -0.1, "zscore": -0.7},
        {"date": "2026-08-13", "surprise": 0.1, "zscore": 0.8},
        {"date": "2026-09-10", "surprise": -0.1, "zscore": -0.85},
    ],
    "NFP": [
        {"date": "2026-04-04", "surprise": -15, "zscore": -0.6},
        {"date": "2026-05-02", "surprise": 20, "zscore": 0.8},
        {"date": "2026-06-06", "surprise": -30, "zscore": -1.2},
        {"date": "2026-07-03", "surprise": 45, "zscore": 1.8},
        {"date": "2026-08-01", "surprise": -10, "zscore": -0.4},
        {"date": "2026-09-05", "surprise": 12, "zscore": 0.92},
    ],
}


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    logger.info("dashboard_starting")
    yield
    logger.info("dashboard_shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Macro Quant Dashboard",
        description="Macroeconomic data analysis, nowcasting, and signal generation",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static files
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # --- Routes ---

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Serve the main dashboard page."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return HTMLResponse("<h1>Dashboard not found. Check static/index.html</h1>")

    @app.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": settings.environment.value,
            "version": "0.1.0",
        }

    @app.get("/api/releases")
    async def get_releases():
        """Get recent economic releases with surprise data."""
        return {"releases": DEMO_RELEASES}

    @app.get("/api/calendar")
    async def get_calendar():
        """Get upcoming economic calendar events."""
        return {"events": DEMO_CALENDAR}

    @app.get("/api/nowcasts")
    async def get_nowcasts():
        """Get latest nowcast estimates."""
        return {"nowcasts": DEMO_NOWCASTS}

    @app.get("/api/surprises/{indicator}")
    async def get_surprise_history(indicator: str):
        """Get surprise history for a specific indicator."""
        history = DEMO_SURPRISE_HISTORY.get(indicator.upper(), [])
        return {"indicator": indicator.upper(), "history": history}

    @app.get("/api/signals")
    async def get_signals():
        """Get current and historical trading signals using the Phase 5 engine."""
        from datetime import datetime, timezone, timedelta
        from src.signals import SignalGenerator, SignalScorer, SignalStatus

        scorer = SignalScorer(
            backtest_hit_rates={"CPI": 0.62, "NFP": 0.58, "GDP": 0.55}
        )
        gen = SignalGenerator(scorer=scorer, min_confidence=0.1)

        # Generate signals with specific dates
        # NFP is Oct 2, GDP is Oct 29, CPI was Sep 11
        now = datetime.now(timezone.utc)
        
        demo_data = [
            ("NFP", 165, 170, "nfp_kalman_v1", datetime(2026, 10, 2, 12, 30, tzinfo=timezone.utc)),
            ("GDP", 2.5, 2.3, "gdp_bridge_v1", datetime(2026, 10, 29, 12, 30, tzinfo=timezone.utc)),
            ("CPI", 2.8, 2.9, "cpi_bridge_v1", datetime(2026, 9, 11, 12, 30, tzinfo=timezone.utc)),
            ("NFP", 140, 155, "nfp_kalman_v1", datetime(2026, 9, 4, 12, 30, tzinfo=timezone.utc)), # Historical
        ]

        active_signals = []
        historical_signals = []
        
        for indicator, nowcast, consensus, model, event_time in demo_data:
            sig = gen.generate_from_nowcast(
                indicator=indicator,
                nowcast_estimate=nowcast,
                consensus=consensus,
                event_time=event_time,
                model_name=model,
            )
            
            if sig:
                # If event_time has passed, mark as EXECUTED for history
                if event_time < now:
                    sig.status = SignalStatus.EXECUTED
                    
                d = sig.to_dict()
                
                # Format dates for frontend
                d["event_time_formatted"] = event_time.strftime("%b %d, %Y")
                
                # Add human-readable reason
                div = nowcast - consensus
                if div > 0:
                    d["reason"] = f"Nowcast above consensus (+{abs(div):.1f}), model: {model}"
                else:
                    d["reason"] = f"Nowcast below consensus ({div:.1f}), model: {model}"
                    
                if sig.status in [SignalStatus.ACTIVE, SignalStatus.PENDING]:
                    active_signals.append(d)
                else:
                    historical_signals.append(d)

        # Sort active by event date closest to now
        active_signals.sort(key=lambda x: x["event_time"])
        # Sort historical by event date most recent first
        historical_signals.sort(key=lambda x: x["event_time"], reverse=True)

        return {
            "signals": active_signals,
            "historical": historical_signals
        }

    @app.get("/api/backtest/summary")
    async def get_backtest_summary():
        """Get backtest performance summary."""
        return {
            "strategy": "Macro Event Surprise",
            "period": "2020-01-01 to 2026-09-01",
            "metrics": {
                "total_return": 34.7,
                "annualized_return": 4.8,
                "sharpe_ratio": 1.42,
                "max_drawdown": -8.3,
                "win_rate": 58.2,
                "total_trades": 312,
                "trades_per_year": 47,
                "avg_trade_duration_minutes": 35,
                "profit_factor": 1.67,
            },
            "monthly_returns": [
                {"month": "2026-01", "return": 2.1},
                {"month": "2026-02", "return": -0.8},
                {"month": "2026-03", "return": 1.5},
                {"month": "2026-04", "return": 3.2},
                {"month": "2026-05", "return": -1.1},
                {"month": "2026-06", "return": 0.7},
                {"month": "2026-07", "return": 2.8},
                {"month": "2026-08", "return": -0.3},
                {"month": "2026-09", "return": 1.4},
            ],
        }

    @app.get("/api/system/status")
    async def get_system_status():
        """Get system component status."""
        return {
            "components": {
                "database": {"status": "demo_mode", "message": "Using demo data"},
                "fred_api": {
                    "status": "configured" if settings.fred_api_key else "not_configured",
                    "rate_limit": f"{settings.fred_rate_limit} req/s",
                },
                "bls_api": {
                    "status": "configured" if settings.bls_api_key else "not_configured",
                    "rate_limit": f"{settings.bls_rate_limit} req/s",
                },
                "polygon_api": {
                    "status": "configured" if settings.polygon_api_key else "not_configured",
                    "rate_limit": f"{settings.polygon_rate_limit} req/s",
                },
                "trading_economics_api": {
                    "status": "configured" if settings.trading_economics_api_key else "not_configured",
                },
                "scheduler": {"status": "ready", "pipeline_steps": 7},
                "nowcast_model": {"status": "ready", "last_run": "2026-09-16T06:00:00"},
            },
            "uptime": "Demo Mode",
        }

    @app.post("/api/pipeline/run")
    async def trigger_pipeline():
        """
        Trigger a full pipeline run (ingest → nowcast → signals).
        Uses the Phase 6 orchestrator.
        """
        from src.scheduler import create_default_pipeline

        pipeline = create_default_pipeline()
        run = await pipeline.run(trigger="api")

        return {
            "run_id": run.run_id,
            "status": run.status,
            "duration_seconds": round(run.duration_seconds, 2),
            "total_records": run.total_records,
            "steps": [
                {
                    "name": s.step_name,
                    "status": s.status.value,
                    "duration": round(s.duration_seconds, 2),
                    "records": s.records_processed,
                    "error": s.error,
                }
                for s in run.steps
            ],
        }

    @app.get("/api/pipeline/status")
    async def pipeline_status():
        """Get pipeline configuration and last run status."""
        from src.scheduler import create_default_pipeline

        pipeline = create_default_pipeline()
        return {
            "steps": pipeline.step_names,
            "n_steps": len(pipeline.step_names),
            "last_run": None,
            "status": "ready",
        }

    @app.get("/api/nowcasts/compute")
    async def compute_nowcast(indicator: str = "CPI", model: str = "bridge"):
        """
        Run a live nowcast using the Phase 3 models with synthetic proxy data.

        When FRED API key is configured, this will use real proxy data.
        For now, uses synthetic data to demonstrate the model pipeline.
        """
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)
        n = 48
        dates = pd.date_range("2022-09-01", periods=n, freq="MS")

        # Synthetic target (CPI MoM %)
        target = pd.Series(
            0.25 + np.cumsum(rng.standard_normal(n) * 0.02) + rng.standard_normal(n) * 0.08,
            index=dates,
            name="target",
        )

        # Synthetic proxies
        proxy_data = {}
        for pname in ["gasoline", "shelter", "food"]:
            noise = rng.standard_normal(n) * 0.05
            proxy_data[pname] = pd.DataFrame(
                {"value": target.values * (0.7 + rng.uniform(0, 0.3)) + noise},
                index=dates,
            )

        try:
            if model == "kalman":
                from src.models.nowcast.kalman_nowcast import KalmanNowcaster

                nowcaster = KalmanNowcaster(
                    name=f"{indicator.lower()}_kalman",
                    indicator=indicator,
                    min_training_samples=12,
                )
                features = pd.DataFrame(index=target.index)
                for name, df in proxy_data.items():
                    features[f"{name}_level"] = df["value"]
            else:
                from src.models.nowcast.bridge_model import BridgeEquationNowcaster

                nowcaster = BridgeEquationNowcaster(
                    name=f"{indicator.lower()}_bridge",
                    indicator=indicator,
                    min_training_samples=12,
                )
                features = nowcaster.build_feature_matrix(
                    target, proxy_data, max_lag=2, use_mom=True
                )

            nowcaster.fit(target, features)
            from datetime import timezone
            next_date = datetime(2026, 10, 1, tzinfo=timezone.utc)
            result = nowcaster.predict(proxy_data, next_date)

            # Walk-forward metrics
            metrics = nowcaster.walk_forward_evaluate(
                target, features, initial_window=24, step=1
            )

            return {
                "indicator": indicator,
                "model": model,
                "nowcast": {
                    "point_estimate": result.point_estimate,
                    "confidence_lower": result.confidence_lower,
                    "confidence_upper": result.confidence_upper,
                    "model_name": result.model_name,
                    "features_used": result.features_used,
                },
                "evaluation": metrics.summary(),
                "status": "computed",
            }
        except Exception as e:
            logger.error("nowcast_compute_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/surprises/compute/{indicator}")
    async def compute_surprise_series(indicator: str):
        """
        Compute rolling Z-score surprise series from demo release data.
        Demonstrates the SurpriseIndex module.
        """
        import numpy as np
        import pandas as pd

        from src.models.surprise_index import SurpriseIndex

        rng = np.random.default_rng(hash(indicator) % 2**32)
        n = 24
        dates = pd.date_range("2024-09-01", periods=n, freq="MS")

        actuals = 0.2 + rng.standard_normal(n) * 0.15
        consensus = actuals + rng.standard_normal(n) * 0.08

        releases = pd.DataFrame({
            "release_date": dates,
            "actual": actuals,
            "consensus": consensus,
        })

        idx = SurpriseIndex(window=12, significance_threshold=1.5, min_observations=4)
        result = idx.compute_series(releases)

        history = []
        for _, row in result.iterrows():
            if pd.notna(row.get("surprise_zscore")):
                history.append({
                    "date": row["release_date"].strftime("%Y-%m-%d"),
                    "surprise": round(float(row["surprise_raw"]), 4),
                    "zscore": round(float(row["surprise_zscore"]), 4),
                    "significant": bool(row["is_significant"]),
                })

        return {
            "indicator": indicator.upper(),
            "computed": True,
            "history": history,
            "n_significant": sum(1 for h in history if h["significant"]),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.dashboard.app:app",
        host="0.0.0.0",
        port=8050,
        reload=True,
        log_level="info",
    )
