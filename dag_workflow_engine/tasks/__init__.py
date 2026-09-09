# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""
Built-in task library for common ML pipeline operations.

Provides a registry of task types that map YAML ``task_type`` strings to
callable implementations.  Each task function receives ``(params, context)``
and returns a dict of outputs that downstream tasks can reference.
"""

from dag_workflow_engine.tasks.registry import TaskRegistry, get_registry
from dag_workflow_engine.tasks.builtin import register_builtin_tasks

__all__ = ["TaskRegistry", "get_registry", "register_builtin_tasks"]
