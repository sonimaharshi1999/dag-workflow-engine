# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""Integration test: run a full ML pipeline end-to-end from YAML."""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

from rich.console import Console

from dag_workflow_engine.executor import PipelineExecutor
from dag_workflow_engine.loader import load_pipeline_from_string
from dag_workflow_engine.models import TaskStatus
from dag_workflow_engine.state import StateStore
from dag_workflow_engine.ui import PipelineUI


ML_PIPELINE_YAML = """
name: integration-test-pipeline
version: "1.0.0"
description: "Full ML pipeline for integration testing"
max_parallel: 2

tasks:
  - id: load
    task_type: load_data
    params:
      n_samples: 100
      n_features: 3
      seed: 42

  - id: validate
    task_type: validate_data
    depends_on: [load]

  - id: transform
    task_type: transform
    depends_on: [validate]
    params:
      method: standardize

  - id: train
    task_type: train
    depends_on: [transform]
    params:
      learning_rate: 0.01
      epochs: 50
      seed: 42

  - id: evaluate
    task_type: evaluate
    depends_on: [train]
    params:
      test_samples: 30
      seed: 99
"""


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_ml_pipeline(self) -> None:
        """Run a complete ML pipeline from YAML and verify all tasks succeed."""
        config = load_pipeline_from_string(ML_PIPELINE_YAML)
        ui = PipelineUI(console=Console(file=StringIO(), force_terminal=True))
        store = StateStore(":memory:")

        executor = PipelineExecutor(
            config=config,
            state_store=store,
            ui=ui,
        )
        result = executor.run()

        assert result.success is True
        assert result.pipeline_name == "integration-test-pipeline"
        assert result.duration_seconds is not None
        assert result.duration_seconds > 0

        # All 5 tasks should succeed
        assert len(result.task_results) == 5
        for task_id, tr in result.task_results.items():
            assert tr.status == TaskStatus.SUCCESS, f"Task {task_id} did not succeed"

        # Verify evaluation produced metrics
        eval_output = result.task_results["evaluate"].output
        assert "test_mse" in eval_output
        assert "test_r2" in eval_output
        assert eval_output["test_mse"] >= 0

    def test_pipeline_from_example_file(self) -> None:
        """Load and execute the shipped simple_pipeline.yml example."""
        example_dir = Path(__file__).parent.parent / "examples"
        simple_yaml = example_dir / "simple_pipeline.yml"

        if not simple_yaml.exists():
            # Skip if examples dir is missing (e.g. in isolated test env)
            return

        from dag_workflow_engine.loader import load_pipeline

        config = load_pipeline(simple_yaml)
        ui = PipelineUI(console=Console(file=StringIO(), force_terminal=True))
        store = StateStore(":memory:")

        executor = PipelineExecutor(
            config=config,
            state_store=store,
            ui=ui,
        )
        result = executor.run()
        assert result.success is True
