"""A record retried with its idempotency key returns the first receipt.

The engine persists the lowered plan for a keyed record, so the product record
operation can be repeated by its caller without writing the learning twice. The
typed client itself never replays a mutation (ADR-0009): the retry here is an
explicit second call, as an agent would make after an unknown outcome.
"""

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from __future__ import annotations

from dataclasses import replace

import pytest

from potpie.cli.commands import _common
from potpie.runtime.composition import build_local_runtime
from potpie_context_engine.adapters.outbound.graph.backends.in_memory_backend import (
    InMemoryGraphBackend,
)
from potpie_context_engine.requests import RecordRequest

pytestmark = pytest.mark.unit


@pytest.fixture()
def runtime(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CONTEXT_ENGINE_HOST_MODE", "in_process")
    composition = build_local_runtime(backend=InMemoryGraphBackend())
    _common.set_runtime(composition)
    composition.root.pots.create_pot(name="retry", use=True)
    yield composition
    composition.close()


def _record(request: RecordRequest):
    return _common.run_engine_operation(
        _common.get_engine_client("retry").record(request)
    )


def _request() -> RecordRequest:
    return RecordRequest(
        record_type="fix",
        summary="Retry pool exhaustion",
        details={
            "root_cause": "Leaked sockets",
            "fix_steps": ["Close sockets in finally"],
        },
        scope={"service": "checkout"},
        source_refs=("fixture:incident:7",),
        idempotency_key="capture:incident:7",
    )


def test_a_keyed_record_repeated_through_the_typed_operation_is_a_duplicate(
    runtime,
) -> None:
    first = _record(_request())
    replay = _record(_request())

    assert first.accepted and first.status == "recorded"
    assert first.mutations_applied > 0
    assert replay.accepted and replay.status == "duplicate"
    assert replay.mutations_applied == 0
    assert replay.metadata["plan_id"] == first.metadata["plan_id"]
    assert replay.metadata["mutation_id"] == first.metadata["mutation_id"]


def test_reusing_a_key_for_different_content_is_refused(runtime) -> None:
    first = _record(_request())
    changed = _record(replace(_request(), details={"root_cause": "Different cause"}))

    assert first.accepted
    assert not changed.accepted
    assert "different content" in (changed.detail or "")
    # The refused attempt does not poison the original retry identity.
    assert _record(_request()).status == "duplicate"
