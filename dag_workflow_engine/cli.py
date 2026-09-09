# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""
Click-based command-line interface for the DAG Workflow Engine.

Provides commands to run, validate, and inspect pipelines defined in YAML.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

from dag_workflow_engine.dag import DAGBuilder, DAGValidationError
from dag_workflow_engine.executor import PipelineExecutor
from dag_workflow_engine.loader import load_pipeline, PipelineLoadError
from dag_workflow_engine.state import StateStore
from dag_workflow_engine.tasks.registry import get_registry
from dag_workflow_engine.ui import PipelineUI


console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="dag-workflow-engine")
def cli() -> None:
    """DAG Workflow Engine -- ML Pipeline Orchestrator.

    Define task dependencies as a directed acyclic graph in YAML, execute
    with topological ordering and automatic parallelism, handle retries
    with exponential backoff, and track execution state.
    """


@cli.command()
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.option(
    "--db",
    default="pipeline_state.db",
    help="SQLite database path for state persistence.",
)
@click.option(
    "--resume/--no-resume",
    default=False,
    help="Resume from the last failed run of this pipeline.",
)
@click.option(
    "--max-parallel",
    type=int,
    default=None,
    help="Override max parallel tasks (default: from YAML).",
)
def run(
    pipeline_file: str,
    db: str,
    resume: bool,
    max_parallel: int | None,
) -> None:
    """Execute a pipeline from a YAML definition file."""
    try:
        config = load_pipeline(pipeline_file)
    except PipelineLoadError as exc:
        console.print(f"[bold red]Error loading pipeline:[/bold red] {exc}")
        sys.exit(1)

    if max_parallel is not None:
        config.max_parallel = max_parallel

    state_store = StateStore(db)
    ui = PipelineUI(console)

    resume_run_id: str | None = None
    if resume:
        resume_run_id = state_store.get_last_run_id(config.name)
        if resume_run_id:
            console.print(
                f"[yellow]Resuming from run: {resume_run_id}[/yellow]"
            )
        else:
            console.print("[yellow]No previous run found, starting fresh.[/yellow]")

    try:
        executor = PipelineExecutor(
            config=config,
            state_store=state_store,
            ui=ui,
            resume_run_id=resume_run_id,
        )
        result = executor.run()
    except DAGValidationError as exc:
        console.print(f"[bold red]DAG Error:[/bold red] {exc}")
        sys.exit(1)
    finally:
        state_store.close()

    if not result.success:
        sys.exit(1)


@cli.command()
@click.argument("pipeline_file", type=click.Path(exists=True))
def validate(pipeline_file: str) -> None:
    """Validate a pipeline YAML file without executing it."""
    try:
        config = load_pipeline(pipeline_file)
    except PipelineLoadError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    ui = PipelineUI(console)

    try:
        dag = DAGBuilder(config)
    except DAGValidationError as exc:
        console.print(f"[bold red]DAG Error:[/bold red] {exc}")
        sys.exit(1)

    issues = dag.validate()
    ui.print_validation_issues(issues)

    if not issues:
        summary = dag.summary()
        console.print(f"\n[bold]Pipeline:[/bold] {config.name}")
        console.print(f"[bold]Tasks:[/bold] {summary['total_tasks']}")
        console.print(f"[bold]Edges:[/bold] {summary['total_edges']}")
        console.print(f"[bold]Generations:[/bold] {summary['parallel_generations']}")
        console.print(f"[bold]Critical Path:[/bold] {' -> '.join(summary['critical_path'])}")

        generations = dag.topological_generations()
        task_descs = {t.id: t.description or t.task_type for t in config.tasks}
        ui.print_dag_structure(generations, task_descs)

    registry = get_registry()
    unknown_types: list[str] = []
    for task in config.tasks:
        if not registry.contains(task.task_type):
            unknown_types.append(f"{task.id} ({task.task_type})")

    if unknown_types:
        console.print(
            f"\n[yellow]Warning: Unknown task types: {', '.join(unknown_types)}[/yellow]"
        )


@cli.command()
def list_tasks() -> None:
    """List all registered built-in task types."""
    registry = get_registry()
    console.print("[bold]Registered Task Types:[/bold]\n")
    for name in registry.list_types():
        console.print(f"  - {name}")


@cli.command()
@click.argument("pipeline_file", type=click.Path(exists=True))
def info(pipeline_file: str) -> None:
    """Display detailed information about a pipeline."""
    try:
        config = load_pipeline(pipeline_file)
    except PipelineLoadError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    try:
        dag = DAGBuilder(config)
    except DAGValidationError as exc:
        console.print(f"[bold red]DAG Error:[/bold red] {exc}")
        sys.exit(1)

    console.print(f"\n[bold cyan]Pipeline:[/bold cyan] {config.name}")
    console.print(f"[bold]Version:[/bold] {config.version}")
    console.print(f"[bold]Description:[/bold] {config.description}")
    console.print(f"[bold]Max Parallel:[/bold] {config.max_parallel}")

    console.print(f"\n[bold]Tasks ({len(config.tasks)}):[/bold]")
    for task in config.tasks:
        deps = ", ".join(task.depends_on) if task.depends_on else "(none)"
        console.print(f"  [cyan]{task.id}[/cyan] [{task.task_type}]")
        console.print(f"    depends_on: {deps}")
        console.print(f"    retries: {task.retry.max_retries}")
        if task.params:
            console.print(f"    params: {task.params}")

    summary = dag.summary()
    console.print(f"\n[bold]Roots:[/bold] {', '.join(summary['roots'])}")
    console.print(f"[bold]Leaves:[/bold] {', '.join(summary['leaves'])}")
    console.print(
        f"[bold]Critical Path:[/bold] {' -> '.join(summary['critical_path'])}"
    )
    console.print()


if __name__ == "__main__":
    cli()
