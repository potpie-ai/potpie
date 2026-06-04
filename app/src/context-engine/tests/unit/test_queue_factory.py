"""Context graph job queue factory (``bootstrap.queue_factory``)."""

import sys
from types import ModuleType

import pytest

from bootstrap.queue_factory import get_context_graph_job_queue
from domain.ports.context_graph_job_queue import NoOpContextGraphJobQueue


def _register_stub_celery_queue() -> type:
    """Minimal stand-in for the host Celery adapter (no ``app`` import)."""

    class StubCeleryQueue:
        def enqueue_backfill(
            self, pot_id: str, *, target_repo_name: str | None = None
        ) -> None:
            return None

        def enqueue_ingest_pr(
            self,
            pot_id: str,
            pr_number: int,
            *,
            is_live_bridge: bool = True,
            repo_name: str | None = None,
        ) -> None:
            return None

        def enqueue_ingestion_event(self, event_id: str, *, pot_id: str, kind: str) -> None:
            return None

        def enqueue_episode_apply(self, pot_id: str, event_id: str, sequence: int) -> None:
            return None

    mod = ModuleType("stub_celery_queue")
    mod.CeleryContextGraphJobQueue = StubCeleryQueue
    sys.modules["stub_celery_queue"] = mod
    return StubCeleryQueue


@pytest.fixture(autouse=True)
def _clear_stub_module(monkeypatch: pytest.MonkeyPatch) -> None:
    yield
    sys.modules.pop("stub_celery_queue", None)


def test_queue_factory_defaults_to_celery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset backend defaults to Celery (host adapter)."""
    Stub = _register_stub_celery_queue()
    monkeypatch.delenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND", raising=False)
    monkeypatch.setenv("CONTEXT_GRAPH_CELERY_QUEUE_MODULE", "stub_celery_queue")
    q = get_context_graph_job_queue()
    assert isinstance(q, Stub)


def test_queue_factory_explicit_celery(monkeypatch: pytest.MonkeyPatch) -> None:
    Stub = _register_stub_celery_queue()
    monkeypatch.setenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND", "celery")
    monkeypatch.setenv("CONTEXT_GRAPH_CELERY_QUEUE_MODULE", "stub_celery_queue")
    q = get_context_graph_job_queue()
    assert isinstance(q, Stub)


def test_queue_factory_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND", "noop")
    q = get_context_graph_job_queue()
    assert isinstance(q, NoOpContextGraphJobQueue)


def test_unknown_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND", "kafka")
    with pytest.raises(ValueError, match="Unknown"):
        get_context_graph_job_queue()


def test_explicit_hatchet_without_token_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND", "hatchet")
    monkeypatch.delenv("HATCHET_CLIENT_TOKEN", raising=False)
    with pytest.raises(Exception):
        get_context_graph_job_queue()


def test_default_celery_fallback_to_noop_when_host_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Implicit celery (default): if host ``app`` adapter is not importable, use noop (standalone CLI)."""
    monkeypatch.delenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND", raising=False)
    monkeypatch.setenv("CONTEXT_GRAPH_CELERY_QUEUE_MODULE", "module_that_does_not_exist_12345")
    q = get_context_graph_job_queue()
    assert isinstance(q, NoOpContextGraphJobQueue)


def test_explicit_celery_import_error_has_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_GRAPH_JOB_QUEUE_BACKEND", "celery")
    monkeypatch.setenv("CONTEXT_GRAPH_CELERY_QUEUE_MODULE", "missing_celery_module_xyz")
    with pytest.raises(ImportError, match="CONTEXT_GRAPH_CELERY_QUEUE_MODULE"):
        get_context_graph_job_queue()
