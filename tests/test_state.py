# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""Tests for SQLite state persistence."""

from __future__ import annotations

from datetime import datetime, timezone

from dag_workflow_engine.models import TaskResult, TaskStatus
from dag_workflow_engine.state import StateStore


class TestStateStore:
    """Tests for SQLite-backed state persistence."""

    def test_create_and_finish_run(self) -> None:
        """Creating and finishing a run should persist correctly."""
        with StateStore(":memory:") as store:
            store.create_run("run-1", "test-pipeline")
            store.finish_run("run-1", True)
            last = store.get_last_run_id("test-pipeline")
            assert last == "run-1"

    def test_save_and_load_task_result(self) -> None:
        """Task results should round-trip through SQLite."""
        with StateStore(":memory:") as store:
            store.create_run("run-2", "test")

            result = TaskResult(
                task_id="task_a",
                status=TaskStatus.SUCCESS,
                started_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                finished_at=datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
                attempts=2,
                output={"mse": 0.05},
            )
            store.save_task_result("run-2", result)

            loaded = store.load_task_results("run-2")
            assert "task_a" in loaded
            assert loaded["task_a"].status == TaskStatus.SUCCESS
            assert loaded["task_a"].attempts == 2
            assert loaded["task_a"].output == {"mse": 0.05}

    def test_get_successful_tasks(self) -> None:
        """Should return only task IDs with SUCCESS status."""
        with StateStore(":memory:") as store:
            store.create_run("run-3", "test")

            store.save_task_result("run-3", TaskResult(
                task_id="ok_task", status=TaskStatus.SUCCESS,
            ))
            store.save_task_result("run-3", TaskResult(
                task_id="bad_task", status=TaskStatus.FAILED,
            ))

            successful = store.get_successful_tasks("run-3")
            assert successful == {"ok_task"}

    def test_upsert_updates_existing(self) -> None:
        """Saving a task result twice should update the existing record."""
        with StateStore(":memory:") as store:
            store.create_run("run-4", "test")

            store.save_task_result("run-4", TaskResult(
                task_id="t1", status=TaskStatus.RUNNING, attempts=1,
            ))
            store.save_task_result("run-4", TaskResult(
                task_id="t1", status=TaskStatus.SUCCESS, attempts=2,
            ))

            loaded = store.load_task_results("run-4")
            assert loaded["t1"].status == TaskStatus.SUCCESS
            assert loaded["t1"].attempts == 2

    def test_no_last_run(self) -> None:
        """get_last_run_id should return None for unknown pipelines."""
        with StateStore(":memory:") as store:
            assert store.get_last_run_id("ghost-pipeline") is None
