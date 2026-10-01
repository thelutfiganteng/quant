"""
Orchestration & Scheduler.

Ties all phases together into an automated pipeline:

    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
    │  Ingestion   │───▶│  Nowcasting   │───▶│   Signals   │
    │  (FRED/BLS)  │    │  (Bridge/KF)  │    │  (Scoring)  │
    └─────────────┘    └──────────────┘    └─────────────┘
          │                                       │
          ▼                                       ▼
    ┌─────────────┐                        ┌─────────────┐
    │   Storage    │                        │  Dashboard   │
    │  (SQLite/PG) │                        │  (FastAPI)   │
    └─────────────┘                        └─────────────┘

Features:
    - Idempotent: safe to re-run (upserts, dedup)
    - Observable: structured logging with run IDs
    - Configurable: schedule via cron or manual trigger
    - Resilient: individual step failures don't crash pipeline
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Step Status
# ---------------------------------------------------------------------------


class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class StepResult:
    """Result of a single pipeline step."""

    step_name: str
    status: StepStatus
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float = 0.0
    records_processed: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == StepStatus.SUCCESS


@dataclass
class PipelineRun:
    """Complete pipeline execution record."""

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    steps: list[StepResult] = field(default_factory=list)
    trigger: str = "manual"  # "manual", "scheduled", "event"

    @property
    def status(self) -> str:
        if any(s.status == StepStatus.RUNNING for s in self.steps):
            return "RUNNING"
        if any(s.status == StepStatus.FAILED for s in self.steps):
            return "PARTIAL_FAILURE"
        if all(s.status == StepStatus.SUCCESS for s in self.steps):
            return "SUCCESS"
        return "UNKNOWN"

    @property
    def duration_seconds(self) -> float:
        if self.finished_at and self.started_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0

    @property
    def total_records(self) -> int:
        return sum(s.records_processed for s in self.steps)

    def summary(self) -> dict[str, Any]:
        """Flat summary for logging/dashboard."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "trigger": self.trigger,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_records": self.total_records,
            "steps": [
                {
                    "name": s.step_name,
                    "status": s.status.value,
                    "duration": round(s.duration_seconds, 2),
                    "records": s.records_processed,
                    "error": s.error,
                }
                for s in self.steps
            ],
        }


# ---------------------------------------------------------------------------
# Pipeline Step Definition
# ---------------------------------------------------------------------------


@dataclass
class PipelineStep:
    """
    Definition of a pipeline step.

    Each step wraps an async function that does the actual work.
    Steps can depend on previous steps and will be skipped
    if dependencies fail.
    """

    name: str
    func: Callable[..., Coroutine[Any, Any, dict[str, Any]]]
    depends_on: list[str] = field(default_factory=list)
    timeout_seconds: float = 300.0
    retry_count: int = 0
    enabled: bool = True


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """
    Main orchestrator that executes the data pipeline.

    Runs steps sequentially with:
        - Dependency checking (skip if upstream failed)
        - Timeout protection
        - Retry logic
        - Structured logging with run IDs
        - Full execution history
    """

    def __init__(self):
        self._steps: list[PipelineStep] = []
        self._run_history: list[PipelineRun] = []
        self._max_history: int = 100

    def add_step(self, step: PipelineStep) -> None:
        """Register a pipeline step."""
        self._steps.append(step)
        logger.debug("step_registered", step=step.name, depends_on=step.depends_on)

    def add_steps(self, steps: list[PipelineStep]) -> None:
        """Register multiple pipeline steps."""
        for step in steps:
            self.add_step(step)

    async def run(
        self,
        trigger: str = "manual",
        step_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> PipelineRun:
        """
        Execute the full pipeline.

        Args:
            trigger: What triggered this run ("manual", "scheduled", "event").
            step_kwargs: Optional per-step kwargs dict.

        Returns:
            PipelineRun with all step results.
        """
        run = PipelineRun(
            run_id=self._generate_run_id(),
            started_at=datetime.now(timezone.utc),
            trigger=trigger,
        )

        step_kwargs = step_kwargs or {}
        completed_steps: dict[str, StepResult] = {}

        logger.info(
            "pipeline_started",
            run_id=run.run_id,
            trigger=trigger,
            n_steps=len(self._steps),
        )

        for step in self._steps:
            # Check if step is enabled
            if not step.enabled:
                result = StepResult(
                    step_name=step.name,
                    status=StepStatus.SKIPPED,
                    started_at=datetime.now(timezone.utc),
                    finished_at=datetime.now(timezone.utc),
                )
                run.steps.append(result)
                completed_steps[step.name] = result
                continue

            # Check dependencies
            deps_met = True
            for dep in step.depends_on:
                dep_result = completed_steps.get(dep)
                if dep_result is None or not dep_result.is_success:
                    deps_met = False
                    break

            if not deps_met:
                result = StepResult(
                    step_name=step.name,
                    status=StepStatus.SKIPPED,
                    started_at=datetime.now(timezone.utc),
                    finished_at=datetime.now(timezone.utc),
                    error=f"Dependency not met: {step.depends_on}",
                )
                run.steps.append(result)
                completed_steps[step.name] = result
                logger.warning(
                    "step_skipped_dependency",
                    run_id=run.run_id,
                    step=step.name,
                    deps=step.depends_on,
                )
                continue

            # Execute step with retry
            result = await self._execute_step(
                step, run.run_id, step_kwargs.get(step.name, {})
            )
            run.steps.append(result)
            completed_steps[step.name] = result

        run.finished_at = datetime.now(timezone.utc)

        # Store in history
        self._run_history.append(run)
        if len(self._run_history) > self._max_history:
            self._run_history = self._run_history[-self._max_history:]

        logger.info("pipeline_completed", **run.summary())

        return run

    async def _execute_step(
        self,
        step: PipelineStep,
        run_id: str,
        kwargs: dict[str, Any],
    ) -> StepResult:
        """Execute a single step with timeout and retry."""
        started = datetime.now(timezone.utc)
        last_error: str | None = None
        attempts = step.retry_count + 1

        for attempt in range(1, attempts + 1):
            logger.info(
                "step_starting",
                run_id=run_id,
                step=step.name,
                attempt=attempt,
                max_attempts=attempts,
            )

            try:
                step_result = await asyncio.wait_for(
                    step.func(**kwargs),
                    timeout=step.timeout_seconds,
                )

                finished = datetime.now(timezone.utc)
                records = step_result.get("records_processed", 0)

                result = StepResult(
                    step_name=step.name,
                    status=StepStatus.SUCCESS,
                    started_at=started,
                    finished_at=finished,
                    duration_seconds=(finished - started).total_seconds(),
                    records_processed=records,
                    metadata=step_result,
                )

                logger.info(
                    "step_completed",
                    run_id=run_id,
                    step=step.name,
                    records=records,
                    duration=round(result.duration_seconds, 2),
                )

                return result

            except asyncio.TimeoutError:
                last_error = f"Timeout after {step.timeout_seconds}s"
                logger.error(
                    "step_timeout",
                    run_id=run_id,
                    step=step.name,
                    timeout=step.timeout_seconds,
                )

            except Exception as e:
                last_error = str(e)
                logger.error(
                    "step_failed",
                    run_id=run_id,
                    step=step.name,
                    attempt=attempt,
                    error=last_error,
                )

            # Wait before retry
            if attempt < attempts:
                await asyncio.sleep(min(2 ** attempt, 30))

        # All attempts failed
        finished = datetime.now(timezone.utc)
        return StepResult(
            step_name=step.name,
            status=StepStatus.FAILED,
            started_at=started,
            finished_at=finished,
            duration_seconds=(finished - started).total_seconds(),
            error=last_error,
        )

    async def run_single_step(
        self, step_name: str, **kwargs: Any
    ) -> StepResult:
        """Run a single step by name (for manual testing/debugging)."""
        step = next((s for s in self._steps if s.name == step_name), None)
        if step is None:
            raise ValueError(f"Step '{step_name}' not found")

        return await self._execute_step(step, f"manual-{step_name}", kwargs)

    @property
    def run_history(self) -> list[PipelineRun]:
        """Get pipeline run history (most recent first)."""
        return list(reversed(self._run_history))

    @property
    def last_run(self) -> PipelineRun | None:
        """Get the most recent pipeline run."""
        return self._run_history[-1] if self._run_history else None

    @property
    def step_names(self) -> list[str]:
        """List registered step names."""
        return [s.name for s in self._steps]

    @staticmethod
    def _generate_run_id() -> str:
        """Generate a unique run ID."""
        return f"run-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Pre-built Pipeline Steps
# ---------------------------------------------------------------------------


async def step_ingest_fred(**kwargs: Any) -> dict[str, Any]:
    """
    Ingest data from FRED API.

    In production, this calls the FREDClient to fetch latest series.
    In demo mode, returns synthetic data.
    """
    from src.config import get_settings

    settings = get_settings()
    records = 0

    if settings.fred_api_key:
        logger.info("fred_ingestion_live", api_key_set=True)
        # Production: use actual FRED client
        # from src.data_ingestion.fred_client import FREDClient
        # async with FREDClient(...) as client:
        #     for series_id in FRED_SERIES:
        #         data = await client.get_series(series_id)
        #         records += len(data)
        records = 0  # Placeholder for live mode
    else:
        logger.info("fred_ingestion_demo", api_key_set=False)
        records = 15  # Demo: simulated

    return {"records_processed": records, "source": "FRED"}


async def step_ingest_bls(**kwargs: Any) -> dict[str, Any]:
    """Ingest data from BLS API."""
    from src.config import get_settings

    settings = get_settings()

    if settings.bls_api_key:
        records = 0  # Placeholder for live mode
    else:
        records = 8  # Demo

    return {"records_processed": records, "source": "BLS"}


async def step_run_nowcasts(**kwargs: Any) -> dict[str, Any]:
    """
    Run nowcast models on latest data.

    Generates estimates for upcoming releases using bridge equation
    and Kalman filter models.
    """
    import numpy as np
    import pandas as pd

    from src.models.nowcast.bridge_model import BridgeEquationNowcaster

    rng = np.random.default_rng(42)
    n = 48
    dates = pd.date_range("2022-09-01", periods=n, freq="MS")

    # Synthetic target and proxies for demo
    target = pd.Series(
        0.25 + np.cumsum(rng.standard_normal(n) * 0.02)
        + rng.standard_normal(n) * 0.08,
        index=dates,
    )

    proxy_data = {}
    for pname in ["gasoline", "shelter", "food"]:
        noise = rng.standard_normal(n) * 0.05
        proxy_data[pname] = pd.DataFrame(
            {"value": target.values * (0.7 + rng.uniform(0, 0.3)) + noise},
            index=dates,
        )

    # Fit and predict
    model = BridgeEquationNowcaster(
        name="cpi_bridge_scheduled",
        indicator="CPI",
        min_training_samples=12,
        max_lag=2,
    )
    features = model.build_feature_matrix(target, proxy_data, max_lag=2)
    model.fit(target, features)

    from datetime import timedelta
    next_date = dates[-1] + timedelta(days=30)
    result = model.predict(proxy_data, next_date)

    return {
        "records_processed": 1,
        "estimates": {
            "CPI": {
                "point": result.point_estimate,
                "lower": result.confidence_lower,
                "upper": result.confidence_upper,
            }
        },
    }


async def step_compute_surprises(**kwargs: Any) -> dict[str, Any]:
    """Compute surprise Z-scores for recent releases."""
    import numpy as np
    import pandas as pd

    from src.models.surprise_index import SurpriseIndex

    rng = np.random.default_rng(42)
    n = 24
    dates = pd.date_range("2024-09-01", periods=n, freq="MS")

    releases = pd.DataFrame({
        "release_date": dates,
        "actual": 0.2 + rng.standard_normal(n) * 0.15,
        "consensus": 0.2 + rng.standard_normal(n) * 0.08,
    })

    idx = SurpriseIndex(window=12, min_observations=4)
    result = idx.compute_series(releases)

    n_significant = int(result["is_significant"].sum())

    return {
        "records_processed": n,
        "n_significant": n_significant,
    }


async def step_generate_signals(**kwargs: Any) -> dict[str, Any]:
    """Generate trading signals from nowcast vs consensus."""
    from src.signals import SignalGenerator, SignalScorer

    scorer = SignalScorer(
        backtest_hit_rates={"CPI": 0.62, "NFP": 0.58, "GDP": 0.55}
    )
    gen = SignalGenerator(scorer=scorer, min_confidence=0.1)

    from datetime import timedelta

    demo_data = [
        ("CPI", 2.8, 2.9, 0.65),
        ("NFP", 165, 170, 0.55),
        ("GDP", 2.5, 2.3, 0.50),
    ]

    count = 0
    for indicator, nowcast, consensus, conf in demo_data:
        event_time = datetime.now(timezone.utc) + timedelta(days=14)
        signal = gen.generate_from_nowcast(
            indicator=indicator,
            nowcast_estimate=nowcast,
            consensus=consensus,
            event_time=event_time,
            nowcast_confidence=conf,
        )
        if signal:
            count += 1

    return {
        "records_processed": count,
        "signals_generated": count,
    }


async def step_expire_signals(**kwargs: Any) -> dict[str, Any]:
    """Expire stale signals."""
    # In production, this would query signal storage
    return {"records_processed": 0, "expired": 0}


async def step_health_check(**kwargs: Any) -> dict[str, Any]:
    """Check system health — API connectivity, DB, etc."""
    from src.config import get_settings

    settings = get_settings()

    health = {
        "fred_configured": bool(settings.fred_api_key),
        "bls_configured": bool(settings.bls_api_key),
        "polygon_configured": bool(settings.polygon_api_key),
        "environment": settings.environment.value,
    }

    return {"records_processed": 0, "health": health}


# ---------------------------------------------------------------------------
# Pre-built Pipeline Factory
# ---------------------------------------------------------------------------


def create_default_pipeline() -> PipelineOrchestrator:
    """
    Create the default macro quant pipeline.

    Pipeline order:
        1. Health check
        2. FRED ingestion
        3. BLS ingestion
        4. Run nowcast models (depends on ingestion)
        5. Compute surprises (depends on ingestion)
        6. Generate signals (depends on nowcasts + surprises)
        7. Expire stale signals
    """
    orchestrator = PipelineOrchestrator()

    orchestrator.add_steps([
        PipelineStep(
            name="health_check",
            func=step_health_check,
            timeout_seconds=30,
        ),
        PipelineStep(
            name="ingest_fred",
            func=step_ingest_fred,
            depends_on=["health_check"],
            timeout_seconds=120,
            retry_count=2,
        ),
        PipelineStep(
            name="ingest_bls",
            func=step_ingest_bls,
            depends_on=["health_check"],
            timeout_seconds=120,
            retry_count=2,
        ),
        PipelineStep(
            name="run_nowcasts",
            func=step_run_nowcasts,
            depends_on=["ingest_fred"],
            timeout_seconds=60,
        ),
        PipelineStep(
            name="compute_surprises",
            func=step_compute_surprises,
            depends_on=["ingest_fred"],
            timeout_seconds=60,
        ),
        PipelineStep(
            name="generate_signals",
            func=step_generate_signals,
            depends_on=["run_nowcasts", "compute_surprises"],
            timeout_seconds=30,
        ),
        PipelineStep(
            name="expire_signals",
            func=step_expire_signals,
            timeout_seconds=10,
        ),
    ])

    return orchestrator
