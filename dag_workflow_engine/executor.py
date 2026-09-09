# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""
Pipeline executor with topological ordering, parallel execution within
generations, retry with exponential backoff, and state persistence.
"""

from __future__ import annotations

import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from typing import Any

from dag_workflow_engine.dag import DAGBuilder, DAGValidationError
from dag_workflow_engine.models import (
    PipelineConfig,
    PipelineResult,
    TaskConfig,
    TaskResult,
    TaskStatus,
)
from dag_workflow_engine.state import StateStore
from dag_workflow_engine.tasks.registry import TaskRegistry, get_registry
from dag_workflow_engine.ui import PipelineUI


def _utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


class ExecutionError(Exception):
    """Raised when the pipeline executor encounters an unrecoverable error."""


class PipelineExecutor:
    """Orchestrate the execution of a pipeline DAG.

    Walks the topological generations of the DAG, executing tasks in
    parallel within each generation and sequentially across generations.
    Handles retries, state persistence, and Rich UI updates.

    Args:
        config: The pipeline configuration to execute.
        registry: Task type registry (uses global default if None).
        state_store: SQLite state store (uses in-memory if None).
        ui: Rich UI instance (creates a default if None).
        resume_run_id: If set, skip tasks that already succeeded in this run.
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: TaskRegistry | None = None,
        state_store: StateStore | None = None,
        ui: PipelineUI | None = None,
        resume_run_id: str | None = None,
    ) -> None:
        self.config: PipelineConfig = config
        self.registry: TaskRegistry = registry or get_registry()
        self.state: StateStore = state_store or StateStore(":memory:")
        self.ui: PipelineUI = ui or PipelineUI()
        self.resume_run_id: str | None = resume_run_id

        # Build and validate the DAG
        self.dag: DAGBuilder = DAGBuilder(config)
        issues = self.dag.validate()
        if any("Cycle detected" in i for i in issues):
            raise DAGValidationError(
                "Pipeline contains a cycle: " + "; ".join(issues)
            )

        # Shared mutable context for passing data between tasks
        self._context: dict[str, Any] = {}

    def run(self) -> PipelineResult:
        """Execute the full pipeline and return aggregated results.

        Returns:
            PipelineResult with per-task outcomes and overall success flag.
        """
        run_id = self.resume_run_id or str(uuid.uuid4())
        started_at = _utcnow()

        # Determine which tasks to skip (resume support)
        skip_tasks: set[str] = set()
        if self.resume_run_id:
            skip_tasks = self.state.get_successful_tasks(run_id)
        else:
            self.state.create_run(run_id, self.config.name)

        generations = self.dag.topological_generations()
        task_results: dict[str, TaskResult] = {}

        # Print header
        self.ui.print_header(self.config.name, len(self.config.tasks))

        task_descs = {
            t.id: t.description or t.task_type
            for t in self.config.tasks
        }
        self.ui.print_dag_structure(generations, task_descs)

        all_success = True

        for gen_idx, generation in enumerate(generations):
            self.ui.print_generation_header(gen_idx, generation)

            # Check if any upstream task failed -- skip dependents
            runnable: list[str] = []
            for task_id in generation:
                if task_id in skip_tasks:
                    tr = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.SKIPPED,
                        attempts=0,
                    )
                    task_results[task_id] = tr
                    self.ui.print_task_status(task_id, TaskStatus.SKIPPED, "already succeeded (resumed)")
                    continue

                deps = self.dag.predecessors(task_id)
                deps_ok = all(
                    task_results.get(d, TaskResult(task_id=d, status=TaskStatus.SUCCESS)).status
                    in (TaskStatus.SUCCESS, TaskStatus.SKIPPED)
                    for d in deps
                )
                if not deps_ok:
                    tr = TaskResult(
                        task_id=task_id,
                        status=TaskStatus.SKIPPED,
                        attempts=0,
                        error="Skipped due to upstream failure",
                    )
                    task_results[task_id] = tr
                    self.state.save_task_result(run_id, tr)
                    self.ui.print_task_status(
                        task_id, TaskStatus.SKIPPED, "skipped (upstream failed)"
                    )
                    all_success = False
                    continue

                runnable.append(task_id)

            # Execute runnable tasks in parallel
            if runnable:
                gen_results = self._execute_generation(runnable, run_id)
                task_results.update(gen_results)

                if any(r.status == TaskStatus.FAILED for r in gen_results.values()):
                    all_success = False

        finished_at = _utcnow()
        self.state.finish_run(run_id, all_success)

        result = PipelineResult(
            pipeline_name=self.config.name,
            started_at=started_at,
            finished_at=finished_at,
            task_results=task_results,
            success=all_success,
        )

        self.ui.print_summary(result)
        return result

    def _execute_generation(
        self, task_ids: list[str], run_id: str
    ) -> dict[str, TaskResult]:
        """Execute a set of tasks concurrently (one generation).

        Args:
            task_ids: Task IDs that are ready to run.
            run_id: The current pipeline run identifier.

        Returns:
            Mapping of task_id to TaskResult for this generation.
        """
        results: dict[str, TaskResult] = {}
        max_workers = min(len(task_ids), self.config.max_parallel)

        if max_workers == 1:
            # Single task -- run inline to avoid thread overhead
            tid = task_ids[0]
            result = self._execute_task_with_retry(tid, run_id)
            results[tid] = result
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures: dict[str, Future[TaskResult]] = {}
                for tid in task_ids:
                    futures[tid] = pool.submit(
                        self._execute_task_with_retry, tid, run_id
                    )

                for tid, future in futures.items():
                    results[tid] = future.result()

        return results

    def _execute_task_with_retry(self, task_id: str, run_id: str) -> TaskResult:
        """Execute a single task with retry logic and state persistence.

        Args:
            task_id: The task to execute.
            run_id: The current pipeline run identifier.

        Returns:
            TaskResult with final outcome after all attempts.
        """
        task_config: TaskConfig = self.dag.get_task(task_id)
        task_func = self.registry.get(task_config.task_type)
        max_attempts = task_config.retry.max_retries + 1

        self.ui.print_task_start(task_id)
        started_at = _utcnow()

        last_error: str | None = None

        for attempt in range(max_attempts):
            try:
                output = task_func(task_config.params, self._context)
                finished_at = _utcnow()
                result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.SUCCESS,
                    started_at=started_at,
                    finished_at=finished_at,
                    attempts=attempt + 1,
                    output=output or {},
                )
                self.state.save_task_result(run_id, result)
                self.ui.print_task_success(task_id, result.duration_seconds)
                return result

            except Exception as exc:
                last_error = str(exc)
                remaining = max_attempts - attempt - 1

                if remaining > 0:
                    delay = task_config.retry.delay_for_attempt(attempt)
                    self.ui.print_task_retry(task_id, attempt + 1, delay)
                    time.sleep(delay)
                else:
                    self.ui.print_task_failure(task_id, last_error, attempt + 1)

        # All retries exhausted
        finished_at = _utcnow()
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            started_at=started_at,
            finished_at=finished_at,
            attempts=max_attempts,
            error=last_error,
        )
        self.state.save_task_result(run_id, result)
        return result
