# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""
YAML pipeline loader and validator.

Reads a YAML file, validates its structure against the Pydantic models,
and returns a fully-typed PipelineConfig ready for DAG construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dag_workflow_engine.models import PipelineConfig


class PipelineLoadError(Exception):
    """Raised when a pipeline YAML file cannot be parsed or validated."""


def load_pipeline(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline definition from a YAML file.

    Args:
        path: Path to the YAML pipeline definition.

    Returns:
        A validated PipelineConfig instance.

    Raises:
        PipelineLoadError: If the file cannot be read or fails validation.
    """
    path = Path(path)

    if not path.exists():
        raise PipelineLoadError(f"Pipeline file not found: {path}")

    if not path.suffix.lower() in (".yml", ".yaml"):
        raise PipelineLoadError(
            f"Expected .yml or .yaml file, got: {path.suffix!r}"
        )

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PipelineLoadError(f"Cannot read {path}: {exc}") from exc

    try:
        raw: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise PipelineLoadError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise PipelineLoadError(
            f"Expected top-level mapping in {path}, got {type(raw).__name__}"
        )

    try:
        config = PipelineConfig(**raw)
    except Exception as exc:
        raise PipelineLoadError(
            f"Pipeline validation failed for {path}: {exc}"
        ) from exc

    return config


def load_pipeline_from_string(yaml_string: str) -> PipelineConfig:
    """Load and validate a pipeline definition from a YAML string.

    Args:
        yaml_string: YAML content as a string.

    Returns:
        A validated PipelineConfig instance.

    Raises:
        PipelineLoadError: If the YAML fails validation.
    """
    try:
        raw: Any = yaml.safe_load(yaml_string)
    except yaml.YAMLError as exc:
        raise PipelineLoadError(f"Invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise PipelineLoadError(
            f"Expected top-level mapping, got {type(raw).__name__}"
        )

    try:
        config = PipelineConfig(**raw)
    except Exception as exc:
        raise PipelineLoadError(f"Pipeline validation failed: {exc}") from exc

    return config
