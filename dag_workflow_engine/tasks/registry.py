# DAG Workflow Engine - ML Pipeline Orchestrator
# Author: Maharshi Soni | License: MIT
"""
Task type registry.

Maps string task-type identifiers to callable implementations.  The executor
resolves each task's ``task_type`` through this registry at runtime.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

# A task function receives (params_dict, context_dict) and returns output dict.
TaskFunc = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class TaskRegistry:
    """Central registry mapping task-type names to their implementations."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskFunc] = {}

    def register(self, name: str, func: TaskFunc) -> None:
        """Register a task function under the given name.

        Args:
            name: Unique task-type identifier (e.g. ``"load_data"``).
            func: Callable ``(params, context) -> output_dict``.

        Raises:
            ValueError: If a task with this name is already registered.
        """
        if name in self._tasks:
            raise ValueError(f"Task type {name!r} is already registered")
        self._tasks[name] = func

    def get(self, name: str) -> TaskFunc:
        """Retrieve a registered task function.

        Raises:
            KeyError: If the task type is not registered.
        """
        if name not in self._tasks:
            available = ", ".join(sorted(self._tasks)) or "(none)"
            raise KeyError(
                f"Unknown task type {name!r}. Available: {available}"
            )
        return self._tasks[name]

    def contains(self, name: str) -> bool:
        """Check whether a task type is registered."""
        return name in self._tasks

    def list_types(self) -> list[str]:
        """Return sorted list of registered task-type names."""
        return sorted(self._tasks)


# Module-level singleton
_registry: TaskRegistry | None = None


def get_registry() -> TaskRegistry:
    """Return (and lazily create) the global task registry."""
    global _registry
    if _registry is None:
        _registry = TaskRegistry()
        from dag_workflow_engine.tasks.builtin import register_builtin_tasks
        register_builtin_tasks(_registry)
    return _registry
