# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""
DAG construction, validation, and analysis using NetworkX.

Builds a directed acyclic graph from a PipelineConfig, validates that no
cycles exist, resolves dependencies, and exposes parallelism information
via topological generations.
"""

from __future__ import annotations

from typing import Iterator

import networkx as nx

from dag_workflow_engine.models import PipelineConfig, TaskConfig


class DAGValidationError(Exception):
    """Raised when the pipeline graph violates DAG constraints."""


class DAGBuilder:
    """Construct and query a directed acyclic graph from pipeline configuration.

    Attributes:
        graph: The underlying NetworkX DiGraph.
        config: The pipeline configuration this DAG was built from.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config: PipelineConfig = config
        self.graph: nx.DiGraph = nx.DiGraph()
        self._task_map: dict[str, TaskConfig] = {}
        self._build()

    def _build(self) -> None:
        """Populate the graph from the pipeline configuration."""
        for task in self.config.tasks:
            self._task_map[task.id] = task
            self.graph.add_node(task.id, task_config=task)

        for task in self.config.tasks:
            for dep in task.depends_on:
                if dep not in self._task_map:
                    raise DAGValidationError(
                        f"Task {task.id!r} depends on unknown task {dep!r}"
                    )
                # Edge direction: dependency -> dependent (dep must finish first)
                self.graph.add_edge(dep, task.id)

    def validate(self) -> list[str]:
        """Validate the DAG and return a list of issues (empty means valid).

        Checks:
        - No cycles
        - All dependencies exist (already enforced in _build)
        - Graph is weakly connected (warning, not fatal)

        Returns:
            List of validation issue descriptions.
        """
        issues: list[str] = []

        if not nx.is_directed_acyclic_graph(self.graph):
            try:
                cycle = nx.find_cycle(self.graph, orientation="original")
                cycle_path = " -> ".join(f"{u}" for u, v, _ in cycle)
                issues.append(f"Cycle detected: {cycle_path}")
            except nx.NetworkXNoCycle:
                pass

        if not nx.is_weakly_connected(self.graph) and self.graph.number_of_nodes() > 1:
            components = list(nx.weakly_connected_components(self.graph))
            issues.append(
                f"Graph has {len(components)} disconnected components "
                f"(not fatal, but may indicate missing dependencies)"
            )

        return issues

    def topological_order(self) -> list[str]:
        """Return task IDs in a valid topological execution order.

        Raises:
            DAGValidationError: If the graph contains cycles.
        """
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible as exc:
            raise DAGValidationError("Graph contains a cycle") from exc

    def topological_generations(self) -> list[list[str]]:
        """Return task IDs grouped into parallel execution generations.

        Each generation is a list of tasks that can execute concurrently
        because all their dependencies are in earlier generations.

        Returns:
            A list of generations, each a list of task IDs.
        """
        try:
            return [list(gen) for gen in nx.topological_generations(self.graph)]
        except nx.NetworkXUnfeasible as exc:
            raise DAGValidationError("Graph contains a cycle") from exc

    def get_task(self, task_id: str) -> TaskConfig:
        """Look up a TaskConfig by its ID.

        Raises:
            KeyError: If the task ID is not in the DAG.
        """
        if task_id not in self._task_map:
            raise KeyError(f"Unknown task: {task_id!r}")
        return self._task_map[task_id]

    def predecessors(self, task_id: str) -> list[str]:
        """Return direct dependency task IDs for the given task."""
        return list(self.graph.predecessors(task_id))

    def successors(self, task_id: str) -> list[str]:
        """Return task IDs that directly depend on the given task."""
        return list(self.graph.successors(task_id))

    def roots(self) -> list[str]:
        """Return task IDs with no dependencies (entry points)."""
        return [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]

    def leaves(self) -> list[str]:
        """Return task IDs with no dependents (exit points)."""
        return [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]

    def critical_path(self) -> list[str]:
        """Compute the longest path through the DAG (critical path).

        Returns:
            List of task IDs on the critical path.
        """
        return list(nx.dag_longest_path(self.graph))

    def task_depth(self, task_id: str) -> int:
        """Return the depth (longest path from any root) of a task."""
        max_depth: int = 0
        for root in self.roots():
            try:
                paths = list(nx.all_simple_paths(self.graph, root, task_id))
                for path in paths:
                    max_depth = max(max_depth, len(path) - 1)
            except nx.NetworkXNoPath:
                continue
        return max_depth

    def summary(self) -> dict[str, int | list[str]]:
        """Return a summary of DAG properties."""
        return {
            "total_tasks": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "roots": self.roots(),
            "leaves": self.leaves(),
            "parallel_generations": len(self.topological_generations()),
            "critical_path": self.critical_path(),
        }
