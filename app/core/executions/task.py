from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from app.core.executions.state import (
    NodeExecutionResult,
)
from app.core.executions.state import NodeExecutionContext


class TaskQueue(ABC):
    """Abstract adapter for task queue."""

    @abstractmethod
    async def enqueue(self, ctx: NodeExecutionContext) -> str:
        """Enqueue a node execution context and return task ID."""
        pass

    @abstractmethod
    async def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a queued task."""
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued task."""
        pass


class WorkerService(ABC):
    """Abstract base class for worker services."""

    @abstractmethod
    async def execute_node_task(self, ctx_data: Dict[str, Any]) -> NodeExecutionResult:
        """Execute a node task."""
        pass

    @abstractmethod
    def start_worker(self) -> None:
        """Start the worker process."""
        pass

    @abstractmethod
    def stop_worker(self) -> None:
        """Stop the worker process."""
        pass
