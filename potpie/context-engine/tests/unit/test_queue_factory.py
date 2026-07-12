"""Injected context graph job queue construction."""

import pytest

from potpie_context_engine.bootstrap.queue_factory import get_context_graph_job_queue
from potpie_context_engine.domain.ports.context_graph_job_queue import (
    NoOpContextGraphJobQueue,
)


class _InjectedQueue:
    def enqueue_batch(self, batch_id: str) -> None:
        del batch_id


def test_queue_factory_defaults_to_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND", raising=False)
    assert isinstance(get_context_graph_job_queue(), NoOpContextGraphJobQueue)


def test_queue_factory_returns_injected_port(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND", "celery")
    injected = _InjectedQueue()
    assert get_context_graph_job_queue(injected) is injected


def test_celery_requires_dependency_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND", "celery")
    monkeypatch.setenv("CONTEXT_GRAPH_" + "CELERY_QUEUE_MODULE", "legacy.module")
    with pytest.raises(ValueError, match="EngineDependencies.job_queue"):
        get_context_graph_job_queue()


def test_unknown_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND", "kafka")
    with pytest.raises(ValueError, match="Unknown"):
        get_context_graph_job_queue()


def test_explicit_hatchet_without_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND", "hatchet")
    monkeypatch.delenv("HATCHET_CLIENT_TOKEN", raising=False)
    with pytest.raises(Exception):
        get_context_graph_job_queue()
