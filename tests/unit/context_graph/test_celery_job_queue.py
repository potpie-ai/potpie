"""CGT-4: Celery job queue adapter enqueues the process-batch task by batch id."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.modules.context_graph.celery_job_queue import CeleryContextGraphJobQueue

pytestmark = pytest.mark.unit


def test_enqueue_batch_dispatches_context_graph_process_batch() -> None:
    task = MagicMock()
    with patch(
        "app.modules.context_graph.tasks.context_graph_process_batch",
        task,
    ):
        CeleryContextGraphJobQueue().enqueue_batch("batch-abc")
    task.delay.assert_called_once_with("batch-abc")
