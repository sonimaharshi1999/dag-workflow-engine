# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""Tests for DAG construction and analysis."""

from __future__ import annotations

import pytest

from dag_workflow_engine.dag import DAGBuilder, DAGValidationError
from dag_workflow_engine.models import PipelineConfig, TaskConfig


def _make_config(tasks: list[TaskConfig], name: str = "test") -> PipelineConfig:
    """Helper to create a PipelineConfig from a list of TaskConfig."""
    return PipelineConfig(name=name, tasks=tasks)


class TestDAGBuilder:
    """Tests for DAG construction and topological analysis."""

    def test_linear_chain(self) -> None:
        """A -> B -> C should produce 3 generations of 1 task each."""
        config = _make_config([
            TaskConfig(id="a", task_type="custom"),
            TaskConfig(id="b", task_type="custom", depends_on=["a"]),
            TaskConfig(id="c", task_type="custom", depends_on=["b"]),
        ])
        dag = DAGBuilder(config)
        gens = dag.topological_generations()
        assert len(gens) == 3
        assert gens[0] == ["a"]
        assert gens[1] == ["b"]
        assert gens[2] == ["c"]

    def test_parallel_branches(self) -> None:
        """Two independent tasks after a root should be in the same generation."""
        config = _make_config([
            TaskConfig(id="root", task_type="custom"),
            TaskConfig(id="branch_a", task_type="custom", depends_on=["root"]),
            TaskConfig(id="branch_b", task_type="custom", depends_on=["root"]),
            TaskConfig(id="merge", task_type="custom", depends_on=["branch_a", "branch_b"]),
        ])
        dag = DAGBuilder(config)
        gens = dag.topological_generations()
        assert len(gens) == 3
        assert set(gens[1]) == {"branch_a", "branch_b"}

    def test_cycle_detection(self) -> None:
        """A cycle in the graph should be caught during validation."""
        config = _make_config([
            TaskConfig(id="a", task_type="custom", depends_on=["c"]),
            TaskConfig(id="b", task_type="custom", depends_on=["a"]),
            TaskConfig(id="c", task_type="custom", depends_on=["b"]),
        ])
        dag = DAGBuilder(config)
        issues = dag.validate()
        assert any("Cycle" in i for i in issues)

    def test_unknown_dependency_raises(self) -> None:
        """Referencing a non-existent task should raise DAGValidationError."""
        config = _make_config([
            TaskConfig(id="a", task_type="custom", depends_on=["ghost"]),
        ])
        with pytest.raises(DAGValidationError, match="unknown task"):
            DAGBuilder(config)

    def test_roots_and_leaves(self) -> None:
        """Roots have no deps; leaves have no dependents."""
        config = _make_config([
            TaskConfig(id="a", task_type="custom"),
            TaskConfig(id="b", task_type="custom", depends_on=["a"]),
            TaskConfig(id="c", task_type="custom", depends_on=["a"]),
        ])
        dag = DAGBuilder(config)
        assert dag.roots() == ["a"]
        assert set(dag.leaves()) == {"b", "c"}

    def test_critical_path(self) -> None:
        """Critical path should be the longest path through the DAG."""
        config = _make_config([
            TaskConfig(id="a", task_type="custom"),
            TaskConfig(id="b", task_type="custom", depends_on=["a"]),
            TaskConfig(id="c", task_type="custom", depends_on=["b"]),
            TaskConfig(id="d", task_type="custom", depends_on=["a"]),
        ])
        dag = DAGBuilder(config)
        cp = dag.critical_path()
        assert len(cp) == 3  # a -> b -> c is the longest

    def test_summary(self) -> None:
        """Summary should report correct counts."""
        config = _make_config([
            TaskConfig(id="a", task_type="custom"),
            TaskConfig(id="b", task_type="custom", depends_on=["a"]),
        ])
        dag = DAGBuilder(config)
        s = dag.summary()
        assert s["total_tasks"] == 2
        assert s["total_edges"] == 1
