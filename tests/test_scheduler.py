"""
Tests for Phase 6 — Orchestration & Scheduler.

Tests the pipeline orchestrator, step execution, dependency
resolution, retry logic, and the pre-built macro quant pipeline.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from src.scheduler import (
    PipelineOrchestrator,
    PipelineRun,
    PipelineStep,
    StepResult,
    StepStatus,
    create_default_pipeline,
    step_compute_surprises,
    step_generate_signals,
    step_health_check,
    step_run_nowcasts,
)


# ---------------------------------------------------------------------------
# Helper step functions for testing
# ---------------------------------------------------------------------------


async def _success_step(**kwargs: Any) -> dict[str, Any]:
    """Always succeeds."""
    return {"records_processed": 10, "test": True}


async def _failing_step(**kwargs: Any) -> dict[str, Any]:
    """Always raises."""
    raise RuntimeError("Intentional test failure")


async def _slow_step(**kwargs: Any) -> dict[str, Any]:
    """Takes too long (simulates timeout)."""
    await asyncio.sleep(10)
    return {"records_processed": 0}


async def _parameterized_step(**kwargs: Any) -> dict[str, Any]:
    """Accepts and returns kwargs."""
    return {"records_processed": kwargs.get("count", 0), "params": kwargs}


# ===========================================================================
# StepResult Tests
# ===========================================================================


class TestStepResult:
    """Tests for StepResult data class."""

    def test_success_result(self):
        """Success result has correct properties."""
        result = StepResult(
            step_name="test",
            status=StepStatus.SUCCESS,
            started_at=datetime.now(timezone.utc),
            records_processed=42,
        )
        assert result.is_success
        assert result.records_processed == 42

    def test_failed_result(self):
        """Failed result is not success."""
        result = StepResult(
            step_name="test",
            status=StepStatus.FAILED,
            started_at=datetime.now(timezone.utc),
            error="boom",
        )
        assert not result.is_success
        assert result.error == "boom"


# ===========================================================================
# PipelineRun Tests
# ===========================================================================


class TestPipelineRun:
    """Tests for PipelineRun aggregate."""

    def test_all_success_status(self):
        """All steps success → SUCCESS."""
        run = PipelineRun(
            run_id="test-1",
            started_at=datetime.now(timezone.utc),
            steps=[
                StepResult("a", StepStatus.SUCCESS, datetime.now(timezone.utc)),
                StepResult("b", StepStatus.SUCCESS, datetime.now(timezone.utc)),
            ],
        )
        assert run.status == "SUCCESS"

    def test_partial_failure_status(self):
        """Any failed step → PARTIAL_FAILURE."""
        run = PipelineRun(
            run_id="test-1",
            started_at=datetime.now(timezone.utc),
            steps=[
                StepResult("a", StepStatus.SUCCESS, datetime.now(timezone.utc)),
                StepResult("b", StepStatus.FAILED, datetime.now(timezone.utc)),
            ],
        )
        assert run.status == "PARTIAL_FAILURE"

    def test_total_records(self):
        """Total records sums across steps."""
        run = PipelineRun(
            run_id="test-1",
            started_at=datetime.now(timezone.utc),
            steps=[
                StepResult("a", StepStatus.SUCCESS, datetime.now(timezone.utc), records_processed=10),
                StepResult("b", StepStatus.SUCCESS, datetime.now(timezone.utc), records_processed=25),
            ],
        )
        assert run.total_records == 35

    def test_summary_dict(self):
        """Summary returns valid dict."""
        run = PipelineRun(
            run_id="test-1",
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            steps=[
                StepResult("a", StepStatus.SUCCESS, datetime.now(timezone.utc)),
            ],
        )
        summary = run.summary()
        assert summary["run_id"] == "test-1"
        assert "steps" in summary
        assert len(summary["steps"]) == 1


# ===========================================================================
# PipelineOrchestrator Tests
# ===========================================================================


class TestPipelineOrchestrator:
    """Tests for the pipeline orchestrator."""

    @pytest.mark.asyncio
    async def test_simple_pipeline(self):
        """Pipeline with one successful step."""
        orch = PipelineOrchestrator()
        orch.add_step(PipelineStep(name="step1", func=_success_step))

        run = await orch.run(trigger="test")

        assert run.status == "SUCCESS"
        assert len(run.steps) == 1
        assert run.steps[0].records_processed == 10

    @pytest.mark.asyncio
    async def test_multi_step_pipeline(self):
        """Pipeline with multiple sequential steps."""
        orch = PipelineOrchestrator()
        orch.add_steps([
            PipelineStep(name="step1", func=_success_step),
            PipelineStep(name="step2", func=_success_step),
            PipelineStep(name="step3", func=_success_step),
        ])

        run = await orch.run()

        assert run.status == "SUCCESS"
        assert len(run.steps) == 3
        assert run.total_records == 30

    @pytest.mark.asyncio
    async def test_step_failure_recorded(self):
        """Failed step is recorded with error."""
        orch = PipelineOrchestrator()
        orch.add_step(PipelineStep(name="fail", func=_failing_step))

        run = await orch.run()

        assert run.status == "PARTIAL_FAILURE"
        assert run.steps[0].status == StepStatus.FAILED
        assert "Intentional test failure" in run.steps[0].error

    @pytest.mark.asyncio
    async def test_dependency_skip(self):
        """Downstream step skipped if dependency fails."""
        orch = PipelineOrchestrator()
        orch.add_steps([
            PipelineStep(name="upstream", func=_failing_step),
            PipelineStep(
                name="downstream",
                func=_success_step,
                depends_on=["upstream"],
            ),
        ])

        run = await orch.run()

        assert run.steps[0].status == StepStatus.FAILED
        assert run.steps[1].status == StepStatus.SKIPPED
        assert "Dependency" in run.steps[1].error

    @pytest.mark.asyncio
    async def test_independent_steps_after_failure(self):
        """Steps without failed dependency still run."""
        orch = PipelineOrchestrator()
        orch.add_steps([
            PipelineStep(name="a", func=_failing_step),
            PipelineStep(name="b", func=_success_step),  # No dependency on "a"
        ])

        run = await orch.run()

        assert run.steps[0].status == StepStatus.FAILED
        assert run.steps[1].status == StepStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Step that exceeds timeout is killed."""
        orch = PipelineOrchestrator()
        orch.add_step(PipelineStep(
            name="slow", func=_slow_step, timeout_seconds=0.1
        ))

        run = await orch.run()

        assert run.steps[0].status == StepStatus.FAILED
        assert "Timeout" in run.steps[0].error

    @pytest.mark.asyncio
    async def test_disabled_step_skipped(self):
        """Disabled step is skipped."""
        orch = PipelineOrchestrator()
        orch.add_step(PipelineStep(
            name="disabled", func=_success_step, enabled=False
        ))

        run = await orch.run()

        assert run.steps[0].status == StepStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_run_history(self):
        """Run history is tracked."""
        orch = PipelineOrchestrator()
        orch.add_step(PipelineStep(name="s", func=_success_step))

        await orch.run(trigger="test1")
        await orch.run(trigger="test2")

        assert len(orch.run_history) == 2
        assert orch.last_run is not None
        assert orch.last_run.trigger == "test2"

    @pytest.mark.asyncio
    async def test_step_kwargs(self):
        """Step receives custom kwargs."""
        orch = PipelineOrchestrator()
        orch.add_step(PipelineStep(name="param", func=_parameterized_step))

        run = await orch.run(
            step_kwargs={"param": {"count": 42}}
        )

        assert run.steps[0].records_processed == 42

    @pytest.mark.asyncio
    async def test_run_single_step(self):
        """Can run a single step by name."""
        orch = PipelineOrchestrator()
        orch.add_steps([
            PipelineStep(name="a", func=_success_step),
            PipelineStep(name="b", func=_parameterized_step),
        ])

        result = await orch.run_single_step("b", count=99)

        assert result.is_success
        assert result.records_processed == 99

    @pytest.mark.asyncio
    async def test_run_nonexistent_step_raises(self):
        """Running a non-existent step raises ValueError."""
        orch = PipelineOrchestrator()
        with pytest.raises(ValueError, match="not found"):
            await orch.run_single_step("nonexistent")

    @pytest.mark.asyncio
    async def test_step_names(self):
        """step_names lists all registered steps."""
        orch = PipelineOrchestrator()
        orch.add_steps([
            PipelineStep(name="alpha", func=_success_step),
            PipelineStep(name="beta", func=_success_step),
        ])
        assert orch.step_names == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_pipeline_run_id_unique(self):
        """Each run gets a unique ID."""
        orch = PipelineOrchestrator()
        orch.add_step(PipelineStep(name="s", func=_success_step))

        run1 = await orch.run()
        run2 = await orch.run()

        assert run1.run_id != run2.run_id


# ===========================================================================
# Pre-built Pipeline Steps Tests
# ===========================================================================


class TestPrebuiltSteps:
    """Tests for the pre-built pipeline step functions."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Health check step runs."""
        result = await step_health_check()
        assert "health" in result
        assert "environment" in result["health"]

    @pytest.mark.asyncio
    async def test_run_nowcasts(self):
        """Nowcast step runs and returns estimates."""
        result = await step_run_nowcasts()
        assert result["records_processed"] == 1
        assert "estimates" in result
        assert "CPI" in result["estimates"]

    @pytest.mark.asyncio
    async def test_compute_surprises(self):
        """Surprise computation step runs."""
        result = await step_compute_surprises()
        assert result["records_processed"] == 24

    @pytest.mark.asyncio
    async def test_generate_signals(self):
        """Signal generation step runs."""
        result = await step_generate_signals()
        assert result["signals_generated"] >= 1


# ===========================================================================
# Default Pipeline Integration Test
# ===========================================================================


class TestDefaultPipeline:
    """Integration tests for the full default pipeline."""

    @pytest.mark.asyncio
    async def test_default_pipeline_runs(self):
        """Full default pipeline runs end-to-end."""
        pipeline = create_default_pipeline()

        assert len(pipeline.step_names) == 7
        assert "health_check" in pipeline.step_names
        assert "generate_signals" in pipeline.step_names

        run = await pipeline.run(trigger="integration_test")

        assert run.status == "SUCCESS"
        assert run.total_records > 0
        assert all(s.is_success for s in run.steps)

    @pytest.mark.asyncio
    async def test_default_pipeline_summary(self):
        """Pipeline summary contains all expected fields."""
        pipeline = create_default_pipeline()
        run = await pipeline.run()

        summary = run.summary()
        assert "run_id" in summary
        assert "status" in summary
        assert "duration_seconds" in summary
        assert len(summary["steps"]) == 7
