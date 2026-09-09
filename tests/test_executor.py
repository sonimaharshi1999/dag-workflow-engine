# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""Tests for the pipeline executor."""

from __future__ import annotations

from io import StringIO
from typing import Any

import pytest
from rich.console import Console

from dag_workflow_engine.executor import PipelineExecutor
from dag_workflow_engine.models import PipelineConfig, TaskConfig, TaskStatus
from dag_workflow_engine.state import StateStore
from dag_workflow_engine.tasks.registry import TaskRegistry
from dag_workflow_engine.ui import PipelineUI


def _quiet_ui() -> PipelineUI:
    """Create a PipelineUI that writes to a string buffer (no terminal output)."""
    return PipelineUI(console=Console(file=StringIO(), force_terminal=True))


def _make_registry() -> TaskRegistry:
    """Create a registry with test-friendly task implementations."""
    reg = TaskRegistry()

    def success_task(params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        ctx["executed"] = ctx.get("executed", [])
        ctx["executed"].append(params.get("name", "unnamed"))
        return {"status": "ok"}

    def fail_task(params: dict[str, Any], ctx: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Intentional failure")

    reg.register("succeed", success_task)
    reg.register("fail", fail_task)
    return reg


class TestPipelineExecutor:
    """Tests for end-to-end pipeline execution."""

    def test_simple_pipeline_succeeds(self) -> None:
        """A linear chain of succeeding tasks should all pass."""
        config = PipelineConfig(
            name="test-simple",
            tasks=[
                TaskConfig(id="a", task_type="succeed", params={"name": "a"}),
                TaskConfig(id="b", task_type="succeed", depends_on=["a"], params={"name": "b"}),
                TaskConfig(id="c", task_type="succeed", depends_on=["b"], params={"name": "c"}),
            ],
        )
        executor = PipelineExecutor(
            config=config,
            registry=_make_registry(),
            state_store=StateStore(":memory:"),
            ui=_quiet_ui(),
        )
        result = executor.run()
        assert result.success is True
        assert all(
            r.status == TaskStatus.SUCCESS
            for r in result.task_results.values()
        )

    def test_failure_skips_downstream(self) -> None:
        """When a task fails, its dependents should be skipped."""
        config = PipelineConfig(
            name="test-fail",
            tasks=[
                TaskConfig(
                    id="bad",
                    task_type="fail",
                    retry={"max_retries": 0, "base_delay": 0.01},
                ),
                TaskConfig(id="downstream", task_type="succeed", depends_on=["bad"]),
            ],
        )
        executor = PipelineExecutor(
            config=config,
            registry=_make_registry(),
            state_store=StateStore(":memory:"),
            ui=_quiet_ui(),
        )
        result = executor.run()
        assert result.success is False
        assert result.task_results["bad"].status == TaskStatus.FAILED
        assert result.task_results["downstream"].status == TaskStatus.SKIPPED

    def test_parallel_execution(self) -> None:
        """Independent tasks in the same generation should all execute."""
        config = PipelineConfig(
            name="test-parallel",
            max_parallel=4,
            tasks=[
                TaskConfig(id="root", task_type="succeed", params={"name": "root"}),
                TaskConfig(id="b1", task_type="succeed", depends_on=["root"], params={"name": "b1"}),
                TaskConfig(id="b2", task_type="succeed", depends_on=["root"], params={"name": "b2"}),
                TaskConfig(id="b3", task_type="succeed", depends_on=["root"], params={"name": "b3"}),
                TaskConfig(id="merge", task_type="succeed", depends_on=["b1", "b2", "b3"], params={"name": "merge"}),
            ],
        )
        executor = PipelineExecutor(
            config=config,
            registry=_make_registry(),
            state_store=StateStore(":memory:"),
            ui=_quiet_ui(),
        )
        result = executor.run()
        assert result.success is True
        assert len(result.task_results) == 5

    def test_retry_exhaustion(self) -> None:
        """A task that always fails should exhaust its retries."""
        config = PipelineConfig(
            name="test-retry",
            tasks=[
                TaskConfig(
                    id="flaky",
                    task_type="fail",
                    retry={"max_retries": 2, "base_delay": 0.01, "max_delay": 0.02},
                ),
            ],
        )
        executor = PipelineExecutor(
            config=config,
            registry=_make_registry(),
            state_store=StateStore(":memory:"),
            ui=_quiet_ui(),
        )
        result = executor.run()
        assert result.success is False
        assert result.task_results["flaky"].status == TaskStatus.FAILED
        assert result.task_results["flaky"].attempts == 3  # 1 initial + 2 retries

    def test_state_persistence(self) -> None:
        """Task results should be persisted to the state store."""
        store = StateStore(":memory:")
        config = PipelineConfig(
            name="test-persist",
            tasks=[
                TaskConfig(id="a", task_type="succeed", params={"name": "a"}),
            ],
        )
        executor = PipelineExecutor(
            config=config,
            registry=_make_registry(),
            state_store=store,
            ui=_quiet_ui(),
        )
        result = executor.run()
        assert result.success is True

        # Verify the state store has the result
        last_run = store.get_last_run_id("test-persist")
        assert last_run is not None
        tasks = store.load_task_results(last_run)
        assert "a" in tasks
        assert tasks["a"].status == TaskStatus.SUCCESS
