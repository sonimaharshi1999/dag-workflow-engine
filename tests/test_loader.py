# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""Tests for YAML pipeline loader."""

from __future__ import annotations

import os
import tempfile

import pytest

from dag_workflow_engine.loader import (
    PipelineLoadError,
    load_pipeline,
    load_pipeline_from_string,
)


class TestLoadPipelineFromString:
    """Tests for loading pipelines from YAML strings."""

    def test_valid_yaml(self) -> None:
        """A valid YAML string should produce a PipelineConfig."""
        yaml_str = """
name: test-pipeline
tasks:
  - id: task_a
    task_type: custom
  - id: task_b
    task_type: custom
    depends_on: [task_a]
"""
        config = load_pipeline_from_string(yaml_str)
        assert config.name == "test-pipeline"
        assert len(config.tasks) == 2

    def test_invalid_yaml_raises(self) -> None:
        """Malformed YAML should raise PipelineLoadError."""
        with pytest.raises(PipelineLoadError, match="Invalid YAML"):
            load_pipeline_from_string("{{invalid yaml")

    def test_missing_name_raises(self) -> None:
        """Missing required field 'name' should raise PipelineLoadError."""
        with pytest.raises(PipelineLoadError, match="validation failed"):
            load_pipeline_from_string("tasks:\n  - id: a\n    task_type: custom\n")


class TestLoadPipelineFromFile:
    """Tests for loading pipelines from YAML files."""

    def test_nonexistent_file_raises(self) -> None:
        """Loading a file that does not exist should raise PipelineLoadError."""
        with pytest.raises(PipelineLoadError, match="not found"):
            load_pipeline("/nonexistent/path/pipeline.yml")

    def test_load_from_file(self) -> None:
        """A valid YAML file should load successfully."""
        yaml_content = """
name: file-test
tasks:
  - id: step1
    task_type: custom
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            f.flush()
            path = f.name

        try:
            config = load_pipeline(path)
            assert config.name == "file-test"
            assert len(config.tasks) == 1
        finally:
            os.unlink(path)

    def test_wrong_extension_raises(self) -> None:
        """A file with a non-YAML extension should raise PipelineLoadError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("name: test\ntasks:\n  - id: a\n    task_type: custom\n")
            f.flush()
            path = f.name

        try:
            with pytest.raises(PipelineLoadError, match="Expected .yml or .yaml"):
                load_pipeline(path)
        finally:
            os.unlink(path)
