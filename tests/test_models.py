# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""Tests for Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dag_workflow_engine.models import (
    PipelineConfig,
    RetryConfig,
    TaskConfig,
    TaskResult,
    TaskStatus,
)


class TestRetryConfig:
    """Tests for RetryConfig backoff calculation."""

    def test_delay_for_attempt_exponential(self) -> None:
        """Backoff delay should grow exponentially."""
        cfg = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=60.0)
        assert cfg.delay_for_attempt(0) == 1.0
        assert cfg.delay_for_attempt(1) == 2.0
        assert cfg.delay_for_attempt(2) == 4.0
        assert cfg.delay_for_attempt(3) == 8.0

    def test_delay_capped_at_max(self) -> None:
        """Backoff delay should not exceed max_delay."""
        cfg = RetryConfig(base_delay=10.0, exponential_base=3.0, max_delay=50.0)
        # 10 * 3^3 = 270, should be capped at 50
        assert cfg.delay_for_attempt(3) == 50.0


class TestTaskConfig:
    """Tests for TaskConfig validation."""

    def test_valid_task(self) -> None:
        """A well-formed task config should validate without error."""
        task = TaskConfig(id="load_data", task_type="load_data")
        assert task.id == "load_data"
        assert task.depends_on == []

    def test_invalid_id_rejected(self) -> None:
        """Task IDs must match the allowed pattern."""
        with pytest.raises(ValidationError):
            TaskConfig(id="123-bad-start", task_type="custom")

    def test_duplicate_deps_deduplicated(self) -> None:
        """Duplicate entries in depends_on should be removed."""
        task = TaskConfig(
            id="t1",
            task_type="custom",
            depends_on=["a", "b", "a", "c", "b"],
        )
        assert task.depends_on == ["a", "b", "c"]


class TestPipelineConfig:
    """Tests for PipelineConfig validation."""

    def test_duplicate_task_ids_rejected(self) -> None:
        """Pipeline with duplicate task IDs should fail validation."""
        with pytest.raises(ValidationError, match="Duplicate task id"):
            PipelineConfig(
                name="test",
                tasks=[
                    TaskConfig(id="t1", task_type="custom"),
                    TaskConfig(id="t1", task_type="custom"),
                ],
            )

    def test_empty_tasks_rejected(self) -> None:
        """Pipeline must have at least one task."""
        with pytest.raises(ValidationError):
            PipelineConfig(name="test", tasks=[])


class TestTaskResult:
    """Tests for TaskResult properties."""

    def test_duration_seconds(self) -> None:
        """Duration should be computed from start/finish times."""
        from datetime import datetime, timezone, timedelta

        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        finish = start + timedelta(seconds=3.5)
        result = TaskResult(
            task_id="t1",
            status=TaskStatus.SUCCESS,
            started_at=start,
            finished_at=finish,
        )
        assert result.duration_seconds == pytest.approx(3.5)

    def test_duration_none_when_not_finished(self) -> None:
        """Duration should be None if the task has not finished."""
        result = TaskResult(task_id="t1", status=TaskStatus.RUNNING)
        assert result.duration_seconds is None
