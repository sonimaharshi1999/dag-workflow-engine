# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""
Data models for pipeline configuration, task definitions, and execution results.

Uses Pydantic v2 for validation and serialization of all pipeline structures.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TaskStatus(str, enum.Enum):
    """Lifecycle states for a task within a pipeline execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class RetryConfig(BaseModel):
    """Configuration for retry behaviour with exponential backoff."""

    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay: float = Field(default=1.0, gt=0, description="Base delay in seconds")
    max_delay: float = Field(default=60.0, gt=0, description="Maximum delay cap in seconds")
    exponential_base: float = Field(default=2.0, gt=1.0)

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate the backoff delay for a given retry attempt number.

        Args:
            attempt: Zero-based attempt index (0 = first retry).

        Returns:
            Delay in seconds, capped at max_delay.
        """
        delay: float = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


class TaskConfig(BaseModel):
    """Definition of a single task within the pipeline DAG."""

    id: str = Field(..., min_length=1, pattern=r"^[a-zA-Z_][a-zA-Z0-9_-]*$")
    task_type: str = Field(..., min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    timeout: float | None = Field(default=None, gt=0, description="Timeout in seconds")
    description: str = Field(default="")

    @field_validator("depends_on")
    @classmethod
    def deduplicate_deps(cls, v: list[str]) -> list[str]:
        """Remove duplicate dependency entries while preserving order."""
        seen: set[str] = set()
        result: list[str] = []
        for dep in v:
            if dep not in seen:
                seen.add(dep)
                result.append(dep)
        return result


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration parsed from YAML."""

    name: str = Field(..., min_length=1)
    version: str = Field(default="1.0.0")
    description: str = Field(default="")
    tasks: list[TaskConfig] = Field(..., min_length=1)
    max_parallel: int = Field(default=4, ge=1, le=32)

    @field_validator("tasks")
    @classmethod
    def unique_task_ids(cls, v: list[TaskConfig]) -> list[TaskConfig]:
        """Ensure all task IDs are unique within the pipeline."""
        ids: set[str] = set()
        for task in v:
            if task.id in ids:
                raise ValueError(f"Duplicate task id: {task.id!r}")
            ids.add(task.id)
        return v


class TaskResult(BaseModel):
    """Outcome of a single task execution."""

    task_id: str
    status: TaskStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempts: int = 0
    error: str | None = None
    output: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock seconds the task ran, or None if not finished."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None


class PipelineResult(BaseModel):
    """Aggregate result of an entire pipeline run."""

    pipeline_name: str
    started_at: datetime
    finished_at: datetime | None = None
    task_results: dict[str, TaskResult] = Field(default_factory=dict)
    success: bool = False

    @property
    def duration_seconds(self) -> float | None:
        """Total pipeline wall-clock time."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return None

    @property
    def failed_tasks(self) -> list[str]:
        """Return IDs of tasks that ended in FAILED status."""
        return [
            tid for tid, r in self.task_results.items()
            if r.status == TaskStatus.FAILED
        ]
