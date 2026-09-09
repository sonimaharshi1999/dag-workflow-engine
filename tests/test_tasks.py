# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""Tests for built-in task implementations."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import pytest

from dag_workflow_engine.tasks.builtin import (
    custom_task,
    evaluate,
    export,
    load_data,
    train,
    transform,
    validate_data,
)
from dag_workflow_engine.tasks.registry import TaskRegistry, get_registry


class TestTaskRegistry:
    """Tests for the task type registry."""

    def test_register_and_retrieve(self) -> None:
        """A registered task should be retrievable by name."""
        reg = TaskRegistry()
        reg.register("test_task", lambda p, c: {"ok": True})
        func = reg.get("test_task")
        assert func({}, {}) == {"ok": True}

    def test_duplicate_registration_raises(self) -> None:
        """Registering the same name twice should raise ValueError."""
        reg = TaskRegistry()
        reg.register("dup", lambda p, c: {})
        with pytest.raises(ValueError, match="already registered"):
            reg.register("dup", lambda p, c: {})

    def test_unknown_task_raises(self) -> None:
        """Getting an unregistered task type should raise KeyError."""
        reg = TaskRegistry()
        with pytest.raises(KeyError, match="Unknown task type"):
            reg.get("nonexistent")

    def test_global_registry_has_builtins(self) -> None:
        """The global registry should contain all built-in task types."""
        reg = get_registry()
        expected = {"load_data", "transform", "train", "evaluate", "export", "custom", "validate_data"}
        assert expected.issubset(set(reg.list_types()))


class TestLoadData:
    """Tests for the load_data task."""

    def test_generates_correct_shape(self) -> None:
        """load_data should generate the requested number of samples and features."""
        ctx: dict[str, Any] = {}
        result = load_data({"n_samples": 50, "n_features": 3, "seed": 42}, ctx)
        assert result["n_samples"] == 50
        assert result["n_features"] == 3
        assert len(ctx["X"]) == 50
        assert len(ctx["X"][0]) == 3
        assert len(ctx["y"]) == 50

    def test_reproducibility(self) -> None:
        """Same seed should produce identical data."""
        ctx1: dict[str, Any] = {}
        ctx2: dict[str, Any] = {}
        load_data({"n_samples": 10, "n_features": 2, "seed": 99}, ctx1)
        load_data({"n_samples": 10, "n_features": 2, "seed": 99}, ctx2)
        assert ctx1["X"] == ctx2["X"]
        assert ctx1["y"] == ctx2["y"]


class TestTransform:
    """Tests for the transform task."""

    def test_standardize(self) -> None:
        """Standardization should center features near zero."""
        import statistics

        ctx: dict[str, Any] = {}
        load_data({"n_samples": 200, "n_features": 3, "seed": 42}, ctx)
        result = transform({"method": "standardize"}, ctx)
        assert result["method"] == "standardize"
        # Check first feature is approximately centered
        col0 = [row[0] for row in ctx["X"]]
        assert abs(statistics.mean(col0)) < 0.01

    def test_unknown_method_raises(self) -> None:
        """An unrecognized transform method should raise ValueError."""
        ctx: dict[str, Any] = {}
        load_data({"n_samples": 10, "n_features": 2, "seed": 1}, ctx)
        with pytest.raises(ValueError, match="Unknown transform method"):
            transform({"method": "banana"}, ctx)


class TestTrainAndEvaluate:
    """Tests for train and evaluate tasks together."""

    def test_train_produces_weights(self) -> None:
        """Training should produce model weights in the context."""
        ctx: dict[str, Any] = {}
        load_data({"n_samples": 100, "n_features": 3, "seed": 42}, ctx)
        result = train({"learning_rate": 0.01, "epochs": 50, "seed": 42}, ctx)
        assert "model_weights" in ctx
        assert len(ctx["model_weights"]) == 4  # 3 features + 1 bias
        assert result["training_mse"] >= 0

    def test_evaluate_produces_metrics(self) -> None:
        """Evaluation should compute MSE, RMSE, and R-squared."""
        ctx: dict[str, Any] = {}
        load_data({"n_samples": 200, "n_features": 3, "seed": 42}, ctx)
        train({"learning_rate": 0.01, "epochs": 100, "seed": 42}, ctx)
        result = evaluate({"test_samples": 50, "seed": 77}, ctx)
        assert "test_mse" in result
        assert "test_rmse" in result
        assert "test_r2" in result
        assert result["test_mse"] >= 0
        assert result["test_rmse"] >= 0


class TestExport:
    """Tests for the export task."""

    def test_export_creates_file(self) -> None:
        """Export should write a valid JSON file."""
        ctx: dict[str, Any] = {
            "model_type": "linear_regression",
            "model_weights": [0.1, 0.2, 0.3],
            "feature_names": ["f0", "f1"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.json")
            result = export({"output_path": path}, ctx)
            assert result["output_path"] == path
            assert result["file_size_bytes"] > 0
            with open(path) as f:
                data = json.load(f)
            assert data["model_type"] == "linear_regression"
            assert data["weights"] == [0.1, 0.2, 0.3]


class TestValidateData:
    """Tests for the validate_data task."""

    def test_valid_dataset_passes(self) -> None:
        """A well-formed dataset should pass all checks."""
        ctx: dict[str, Any] = {}
        load_data({"n_samples": 50, "n_features": 3, "seed": 42}, ctx)
        result = validate_data({}, ctx)
        assert result["is_valid"] is True
        assert result["n_samples"] == 50

    def test_empty_dataset_fails(self) -> None:
        """An empty dataset should fail validation."""
        result = validate_data({}, {})
        assert result["is_valid"] is False
