# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""
Built-in task implementations for common ML pipeline stages.

All tasks use only the Python standard library plus lightweight synthetic
data generation -- no paid APIs or external downloads required.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

from dag_workflow_engine.tasks.registry import TaskRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_rng(params: dict[str, Any]) -> random.Random:
    """Create a seeded Random instance for reproducibility."""
    seed = params.get("seed", 42)
    return random.Random(seed)


def _generate_synthetic_dataset(
    n_samples: int,
    n_features: int,
    rng: random.Random,
    noise: float = 0.1,
) -> tuple[list[list[float]], list[float]]:
    """Generate a synthetic regression-style dataset.

    Returns:
        (X, y) where X is n_samples x n_features and y is n_samples.
    """
    # Random coefficients for a linear model
    coefficients = [rng.gauss(0, 1) for _ in range(n_features)]
    intercept = rng.gauss(0, 0.5)

    X: list[list[float]] = []
    y: list[float] = []
    for _ in range(n_samples):
        row = [rng.gauss(0, 1) for _ in range(n_features)]
        target = intercept + sum(c * x for c, x in zip(coefficients, row))
        target += rng.gauss(0, noise)
        X.append(row)
        y.append(target)
    return X, y


# ---------------------------------------------------------------------------
# Task: load_data
# ---------------------------------------------------------------------------

def load_data(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Generate or load a synthetic dataset.

    Params:
        n_samples (int): Number of samples (default 1000).
        n_features (int): Number of features (default 5).
        seed (int): Random seed (default 42).
        noise (float): Gaussian noise std (default 0.1).

    Output:
        n_samples, n_features, data_hash
    """
    n_samples: int = params.get("n_samples", 1000)
    n_features: int = params.get("n_features", 5)
    noise: float = params.get("noise", 0.1)
    rng = _seed_rng(params)

    X, y = _generate_synthetic_dataset(n_samples, n_features, rng, noise)

    # Store in context for downstream tasks
    context["X"] = X
    context["y"] = y
    context["feature_names"] = [f"feature_{i}" for i in range(n_features)]

    # Compute a hash for data lineage tracking
    data_str = json.dumps({"X_shape": [n_samples, n_features], "y_len": len(y)})
    data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:12]

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "data_hash": data_hash,
    }


# ---------------------------------------------------------------------------
# Task: transform
# ---------------------------------------------------------------------------

def transform(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Apply feature transformations (standardization, polynomial features).

    Params:
        method (str): "standardize" | "normalize" | "polynomial" (default "standardize").
        degree (int): Polynomial degree if method is "polynomial" (default 2).

    Output:
        method, n_features_out
    """
    method: str = params.get("method", "standardize")
    X: list[list[float]] = context.get("X", [])

    if not X:
        raise ValueError("No data in context. Run load_data first.")

    n_samples = len(X)
    n_features = len(X[0])

    if method == "standardize":
        # Z-score standardization per feature
        for j in range(n_features):
            col = [X[i][j] for i in range(n_samples)]
            mean = statistics.mean(col)
            std = statistics.stdev(col) if len(col) > 1 else 1.0
            std = std if std > 1e-10 else 1.0
            for i in range(n_samples):
                X[i][j] = (X[i][j] - mean) / std

    elif method == "normalize":
        # Min-max normalization per feature
        for j in range(n_features):
            col = [X[i][j] for i in range(n_samples)]
            min_val = min(col)
            max_val = max(col)
            rng_val = max_val - min_val if (max_val - min_val) > 1e-10 else 1.0
            for i in range(n_samples):
                X[i][j] = (X[i][j] - min_val) / rng_val

    elif method == "polynomial":
        # Add polynomial interaction features (degree 2)
        degree: int = params.get("degree", 2)
        new_X: list[list[float]] = []
        for row in X:
            new_row = list(row)
            if degree >= 2:
                for a in range(n_features):
                    for b in range(a, n_features):
                        new_row.append(row[a] * row[b])
            new_X.append(new_row)
        X = new_X
    else:
        raise ValueError(f"Unknown transform method: {method!r}")

    context["X"] = X
    n_features_out = len(X[0]) if X else 0
    return {"method": method, "n_features_out": n_features_out}


# ---------------------------------------------------------------------------
# Task: train
# ---------------------------------------------------------------------------

def train(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Train a simple linear regression model (OLS via normal equations).

    Uses only standard-library math -- no sklearn required.

    Params:
        learning_rate (float): Not used for OLS but logged (default 0.01).
        model_type (str): "linear_regression" (default).

    Output:
        model_type, n_coefficients, training_mse
    """
    X: list[list[float]] = context.get("X", [])
    y: list[float] = context.get("y", [])

    if not X or not y:
        raise ValueError("No training data in context.")

    n = len(X)
    p = len(X[0])

    # Add bias column
    X_bias = [[1.0] + row for row in X]
    p_bias = p + 1

    # OLS closed-form: coeffs = (X^T X)^{-1} X^T y
    # For simplicity, use gradient descent instead (avoids matrix inversion)
    lr: float = params.get("learning_rate", 0.01)
    epochs: int = params.get("epochs", 100)
    rng = _seed_rng(params)

    weights = [rng.gauss(0, 0.01) for _ in range(p_bias)]

    for epoch in range(epochs):
        # Compute predictions
        gradients = [0.0] * p_bias
        total_loss = 0.0
        for i in range(n):
            pred = sum(w * x for w, x in zip(weights, X_bias[i]))
            error = pred - y[i]
            total_loss += error ** 2
            for j in range(p_bias):
                gradients[j] += (2.0 / n) * error * X_bias[i][j]

        # Update weights
        for j in range(p_bias):
            weights[j] -= lr * gradients[j]

    # Final MSE
    mse = 0.0
    for i in range(n):
        pred = sum(w * x for w, x in zip(weights, X_bias[i]))
        mse += (pred - y[i]) ** 2
    mse /= n

    context["model_weights"] = weights
    context["model_type"] = params.get("model_type", "linear_regression")

    return {
        "model_type": context["model_type"],
        "n_coefficients": len(weights),
        "training_mse": round(mse, 6),
    }


# ---------------------------------------------------------------------------
# Task: evaluate
# ---------------------------------------------------------------------------

def evaluate(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the trained model on a held-out synthetic test set.

    Params:
        test_samples (int): Number of test samples to generate (default 200).

    Output:
        test_mse, test_rmse, test_r2
    """
    weights: list[float] = context.get("model_weights", [])
    if not weights:
        raise ValueError("No model weights in context. Run train first.")

    n_features = len(weights) - 1  # subtract bias
    test_n: int = params.get("test_samples", 200)
    rng = _seed_rng(params)

    # Generate fresh test data using a different seed offset
    rng2 = random.Random(rng.randint(0, 999999))
    X_test, y_test = _generate_synthetic_dataset(test_n, n_features, rng2, 0.1)

    # Predict
    predictions: list[float] = []
    for row in X_test:
        pred = weights[0] + sum(w * x for w, x in zip(weights[1:], row))
        predictions.append(pred)

    # Metrics
    mse = sum((p - a) ** 2 for p, a in zip(predictions, y_test)) / test_n
    rmse = math.sqrt(mse)

    y_mean = statistics.mean(y_test)
    ss_res = sum((a - p) ** 2 for p, a in zip(predictions, y_test))
    ss_tot = sum((a - y_mean) ** 2 for a in y_test)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0

    return {
        "test_mse": round(mse, 6),
        "test_rmse": round(rmse, 6),
        "test_r2": round(r2, 6),
    }


# ---------------------------------------------------------------------------
# Task: export
# ---------------------------------------------------------------------------

def export(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Export model artifacts and metadata to JSON.

    Params:
        output_path (str): File path for export (default "model_output.json").

    Output:
        output_path, file_size_bytes
    """
    output_path: str = params.get("output_path", "model_output.json")

    artifact: dict[str, Any] = {
        "model_type": context.get("model_type", "unknown"),
        "weights": context.get("model_weights", []),
        "feature_names": context.get("feature_names", []),
        "metadata": {
            "engine": "dag-workflow-engine",
            "version": "1.0.0",
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    file_size = Path(output_path).stat().st_size

    return {
        "output_path": str(output_path),
        "file_size_bytes": file_size,
    }


# ---------------------------------------------------------------------------
# Task: custom (pass-through / noop)
# ---------------------------------------------------------------------------

def custom_task(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """A pass-through task for user-defined logic or testing.

    Params:
        message (str): A message to include in the output.
        sleep (float): Optional artificial delay in seconds.

    Output:
        message
    """
    message: str = params.get("message", "custom task executed")
    sleep_time: float = params.get("sleep", 0.0)
    if sleep_time > 0:
        time.sleep(sleep_time)
    return {"message": message}


# ---------------------------------------------------------------------------
# Task: validate_data
# ---------------------------------------------------------------------------

def validate_data(params: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Validate dataset quality: check for nulls, shape, and basic stats.

    Output:
        is_valid, n_samples, n_features, checks_passed
    """
    X: list[list[float]] = context.get("X", [])
    y: list[float] = context.get("y", [])

    checks: list[str] = []
    is_valid = True

    if not X:
        is_valid = False
        checks.append("FAIL: X is empty")
    else:
        checks.append(f"PASS: X has {len(X)} samples")

    if not y:
        is_valid = False
        checks.append("FAIL: y is empty")
    else:
        checks.append(f"PASS: y has {len(y)} values")

    if X and y and len(X) != len(y):
        is_valid = False
        checks.append(f"FAIL: X rows ({len(X)}) != y length ({len(y)})")

    if X:
        widths = {len(row) for row in X}
        if len(widths) > 1:
            is_valid = False
            checks.append(f"FAIL: inconsistent feature counts: {widths}")
        else:
            checks.append(f"PASS: consistent {widths.pop()} features")

        # Check for NaN/Inf
        bad_count = sum(
            1 for row in X for v in row
            if not math.isfinite(v)
        )
        if bad_count > 0:
            is_valid = False
            checks.append(f"FAIL: {bad_count} non-finite values in X")
        else:
            checks.append("PASS: no NaN/Inf values")

    return {
        "is_valid": is_valid,
        "n_samples": len(X),
        "n_features": len(X[0]) if X else 0,
        "checks_passed": checks,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_builtin_tasks(registry: TaskRegistry) -> None:
    """Register all built-in task types with the given registry."""
    registry.register("load_data", load_data)
    registry.register("transform", transform)
    registry.register("train", train)
    registry.register("evaluate", evaluate)
    registry.register("export", export)
    registry.register("custom", custom_task)
    registry.register("validate_data", validate_data)
