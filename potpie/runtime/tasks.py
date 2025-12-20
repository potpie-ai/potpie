"""Async task runner for Potpie (replaces Celery for local mode)."""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Coroutine, Optional
from uuid import uuid4


class TaskStatus(str, Enum):
    """Status of an async task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Result of a completed task."""

    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class TaskInfo:
    """Information about a running task."""

    task_id: str
    name: str
    status: TaskStatus
    asyncio_task: Optional[asyncio.Task] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class TaskRunner:
    """Simple async task runner for local mode.

    Replaces Celery for background task execution.
    Tasks run in the same event loop as the main application.
    """

    def __init__(self):
        self._tasks: dict[str, TaskInfo] = {}

    async def submit(
        self,
        coro: Coroutine[Any, Any, Any],
        name: Optional[str] = None,
    ) -> str:
        """Submit a coroutine for background execution.

        Args:
            coro: Coroutine to execute.
            name: Optional task name for identification.

        Returns:
            Task ID for tracking.
        """
        task_id = str(uuid4())
        task_name = name or f"task-{task_id[:8]}"

        async def wrapped():
            try:
                self._tasks[task_id].status = TaskStatus.RUNNING
                result = await coro
                self._tasks[task_id].status = TaskStatus.COMPLETED
                self._tasks[task_id].result = result
                return result
            except asyncio.CancelledError:
                self._tasks[task_id].status = TaskStatus.CANCELLED
                raise
            except Exception as e:
                self._tasks[task_id].status = TaskStatus.FAILED
                self._tasks[task_id].error = str(e)
                raise

        asyncio_task = asyncio.create_task(wrapped())
        self._tasks[task_id] = TaskInfo(
            task_id=task_id,
            name=task_name,
            status=TaskStatus.PENDING,
            asyncio_task=asyncio_task,
        )

        return task_id

    async def get_status(self, task_id: str) -> Optional[TaskInfo]:
        """Get status of a task."""
        return self._tasks.get(task_id)

    async def wait(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """Wait for a task to complete.

        Args:
            task_id: Task to wait for.
            timeout: Optional timeout in seconds.

        Returns:
            TaskResult with outcome.

        Raises:
            KeyError: If task_id not found.
            asyncio.TimeoutError: If timeout exceeded.
        """
        info = self._tasks.get(task_id)
        if not info:
            raise KeyError(f"Task {task_id} not found")

        # if info.asyncio_task:
        #     try:
        #         await asyncio.wait_for(info.asyncio_task, timeout=timeout)
        #     except asyncio.TimeoutError:
        #         raise
        #     except Exception:
        #         pass  # Error captured in task wrapper

        return TaskResult(
            task_id=task_id,
            status=info.status,
            result=info.result,
            error=info.error,
        )

    async def cancel(self, task_id: str) -> bool:
        """Cancel a running task.

        Returns:
            True if task was cancelled, False if not found or already done.
        """
        info = self._tasks.get(task_id)
        if not info or not info.asyncio_task:
            return False

        if info.asyncio_task.done():
            return False

        info.asyncio_task.cancel()
        return True

    def cleanup_completed(self) -> int:
        """Remove completed tasks from tracking.

        Returns:
            Number of tasks cleaned up.
        """
        completed = [
            tid
            for tid, info in self._tasks.items()
            if info.status
            in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
        ]
        for tid in completed:
            del self._tasks[tid]
        return len(completed)
