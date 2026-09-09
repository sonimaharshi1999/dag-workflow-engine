# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""
Execution state persistence via SQLite.

Stores per-run, per-task execution records so that a pipeline can resume
from the last successful checkpoint after a crash or manual abort.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dag_workflow_engine.models import TaskResult, TaskStatus


def _utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def _dt_to_str(dt: datetime | None) -> str | None:
    """Serialise a datetime to ISO-8601 string."""
    return dt.isoformat() if dt else None


def _str_to_dt(s: str | None) -> datetime | None:
    """Deserialise an ISO-8601 string back to datetime."""
    if s is None:
        return None
    return datetime.fromisoformat(s)


class StateStore:
    """SQLite-backed persistence for pipeline execution state.

    Each pipeline run gets a unique ``run_id``.  Individual task results
    are upserted as they complete, enabling resume-on-failure.
    """

    def __init__(self, db_path: str | Path = "pipeline_state.db") -> None:
        self.db_path: Path = Path(db_path)
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables if they do not exist."""
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id       TEXT PRIMARY KEY,
                    pipeline     TEXT NOT NULL,
                    started_at   TEXT NOT NULL,
                    finished_at  TEXT,
                    success      INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS task_results (
                    run_id      TEXT NOT NULL,
                    task_id     TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    started_at  TEXT,
                    finished_at TEXT,
                    attempts    INTEGER DEFAULT 0,
                    error       TEXT,
                    output      TEXT DEFAULT '{}',
                    PRIMARY KEY (run_id, task_id),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                """
            )

    # ------------------------------------------------------------------
    # Run-level operations
    # ------------------------------------------------------------------

    def create_run(self, run_id: str, pipeline_name: str) -> None:
        """Insert a new run record."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs (run_id, pipeline, started_at) VALUES (?, ?, ?)",
                (run_id, pipeline_name, _dt_to_str(_utcnow())),
            )

    def finish_run(self, run_id: str, success: bool) -> None:
        """Mark a run as finished."""
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at = ?, success = ? WHERE run_id = ?",
                (_dt_to_str(_utcnow()), int(success), run_id),
            )

    # ------------------------------------------------------------------
    # Task-level operations
    # ------------------------------------------------------------------

    def save_task_result(self, run_id: str, result: TaskResult) -> None:
        """Upsert a task result for the given run."""
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO task_results
                    (run_id, task_id, status, started_at, finished_at,
                     attempts, error, output)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, task_id) DO UPDATE SET
                    status      = excluded.status,
                    started_at  = excluded.started_at,
                    finished_at = excluded.finished_at,
                    attempts    = excluded.attempts,
                    error       = excluded.error,
                    output      = excluded.output
                """,
                (
                    run_id,
                    result.task_id,
                    result.status.value,
                    _dt_to_str(result.started_at),
                    _dt_to_str(result.finished_at),
                    result.attempts,
                    result.error,
                    json.dumps(result.output),
                ),
            )

    def load_task_results(self, run_id: str) -> dict[str, TaskResult]:
        """Load all task results for a run, keyed by task_id."""
        cursor = self._conn.execute(
            "SELECT * FROM task_results WHERE run_id = ?", (run_id,)
        )
        results: dict[str, TaskResult] = {}
        for row in cursor:
            results[row["task_id"]] = TaskResult(
                task_id=row["task_id"],
                status=TaskStatus(row["status"]),
                started_at=_str_to_dt(row["started_at"]),
                finished_at=_str_to_dt(row["finished_at"]),
                attempts=row["attempts"],
                error=row["error"],
                output=json.loads(row["output"] or "{}"),
            )
        return results

    def get_successful_tasks(self, run_id: str) -> set[str]:
        """Return task IDs that already succeeded in the given run."""
        cursor = self._conn.execute(
            "SELECT task_id FROM task_results WHERE run_id = ? AND status = ?",
            (run_id, TaskStatus.SUCCESS.value),
        )
        return {row["task_id"] for row in cursor}

    def get_last_run_id(self, pipeline_name: str) -> str | None:
        """Return the most recent run_id for a pipeline, or None."""
        cursor = self._conn.execute(
            "SELECT run_id FROM runs WHERE pipeline = ? ORDER BY started_at DESC LIMIT 1",
            (pipeline_name,),
        )
        row = cursor.fetchone()
        return row["run_id"] if row else None

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> StateStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
