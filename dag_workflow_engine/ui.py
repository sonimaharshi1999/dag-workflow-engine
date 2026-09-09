# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""
Rich terminal UI for displaying DAG execution progress.

Provides live-updating tables and panels that show which tasks are running,
completed, or failed, with timing information and retry counts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from dag_workflow_engine.models import PipelineResult, TaskResult, TaskStatus


STATUS_STYLES: dict[TaskStatus, tuple[str, str]] = {
    TaskStatus.PENDING:  ("dim",           "[.]"),
    TaskStatus.RUNNING:  ("bold yellow",   "[>]"),
    TaskStatus.SUCCESS:  ("bold green",    "[+]"),
    TaskStatus.FAILED:   ("bold red",      "[X]"),
    TaskStatus.SKIPPED:  ("dim magenta",   "[-]"),
    TaskStatus.RETRYING: ("bold yellow",   "[~]"),
}


class PipelineUI:
    """Rich-based terminal UI for pipeline execution progress."""

    def __init__(self, console: Console | None = None) -> None:
        self.console: Console = console or Console()

    def print_header(self, pipeline_name: str, total_tasks: int) -> None:
        """Print a styled pipeline header."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]{pipeline_name}[/bold cyan]\n"
                f"[dim]{total_tasks} tasks in pipeline[/dim]",
                title="DAG Workflow Engine",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        self.console.print()

    def print_dag_structure(
        self,
        generations: list[list[str]],
        task_descriptions: dict[str, str] | None = None,
    ) -> None:
        """Print the DAG as a tree grouped by execution generation."""
        tree = Tree("[bold]Pipeline DAG[/bold]", guide_style="dim")

        for i, gen in enumerate(generations):
            gen_node = tree.add(f"[bold cyan]Generation {i}[/bold cyan] (parallel)")
            for task_id in gen:
                desc = ""
                if task_descriptions and task_id in task_descriptions:
                    desc = f" - [dim]{task_descriptions[task_id]}[/dim]"
                gen_node.add(f"[white]{task_id}[/white]{desc}")

        self.console.print(tree)
        self.console.print()

    def print_task_status(self, task_id: str, status: TaskStatus, detail: str = "") -> None:
        """Print a single task status update line."""
        style, icon = STATUS_STYLES.get(status, ("", "[?]"))
        msg = Text()
        msg.append(f"  {icon} ", style=style)
        msg.append(task_id, style="bold")
        if detail:
            msg.append(f" {detail}", style="dim")
        self.console.print(msg)

    def print_task_start(self, task_id: str) -> None:
        """Print a task-started message."""
        self.print_task_status(task_id, TaskStatus.RUNNING, "started")

    def print_task_success(self, task_id: str, duration: float | None = None) -> None:
        """Print a task-completed message."""
        detail = f"completed in {duration:.2f}s" if duration else "completed"
        self.print_task_status(task_id, TaskStatus.SUCCESS, detail)

    def print_task_failure(self, task_id: str, error: str, attempt: int = 0) -> None:
        """Print a task-failure message."""
        detail = f"FAILED (attempt {attempt}): {error}"
        self.print_task_status(task_id, TaskStatus.FAILED, detail)

    def print_task_retry(self, task_id: str, attempt: int, delay: float) -> None:
        """Print a retry notification."""
        detail = f"retrying (attempt {attempt}, backoff {delay:.1f}s)"
        self.print_task_status(task_id, TaskStatus.RETRYING, detail)

    def print_generation_header(self, gen_index: int, tasks: list[str]) -> None:
        """Print a generation separator."""
        task_list = ", ".join(tasks)
        self.console.print(
            f"\n[bold cyan]--- Generation {gen_index} ---[/bold cyan] "
            f"[dim]({len(tasks)} tasks: {task_list})[/dim]"
        )

    def print_summary(self, result: PipelineResult) -> None:
        """Print a final execution summary table."""
        self.console.print()

        table = Table(
            title="Execution Summary",
            show_header=True,
            header_style="bold",
            border_style="cyan",
        )
        table.add_column("Task", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Attempts", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Error", max_width=40)

        for task_id, tr in result.task_results.items():
            style, icon = STATUS_STYLES.get(tr.status, ("", "[?]"))
            status_text = Text(f"{icon} {tr.status.value}", style=style)

            duration_str = ""
            if tr.duration_seconds is not None:
                duration_str = f"{tr.duration_seconds:.3f}s"

            table.add_row(
                task_id,
                status_text,
                str(tr.attempts),
                duration_str,
                tr.error or "",
            )

        self.console.print(table)

        # Overall result
        total_duration = result.duration_seconds
        duration_text = f" in {total_duration:.2f}s" if total_duration else ""

        if result.success:
            self.console.print(
                f"\n[bold green]Pipeline completed successfully{duration_text}[/bold green]"
            )
        else:
            failed = ", ".join(result.failed_tasks)
            self.console.print(
                f"\n[bold red]Pipeline FAILED{duration_text}[/bold red]"
            )
            self.console.print(f"[red]  Failed tasks: {failed}[/red]")

        self.console.print()

    def print_validation_issues(self, issues: list[str]) -> None:
        """Print DAG validation issues."""
        if not issues:
            self.console.print("[green]DAG validation passed[/green]")
            return

        self.console.print("[bold red]DAG Validation Issues:[/bold red]")
        for issue in issues:
            self.console.print(f"  [red]* {issue}[/red]")
