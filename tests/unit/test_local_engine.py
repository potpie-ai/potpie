from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from potpie.pots.contracts import PotInfo, SourceInfo
from potpie.runtime import ContextSelector, LocalEngineClient, OperationCoordinator
from potpie.runtime.local_engine import (
    LocalContextSelectorResolver,
    LocalEngineOperations,
    build_local_resource_manager,
)
from potpie_context_engine import ContextIdentity, Failure, Success
from potpie_context_engine.domain.ingestion_event_models import (
    EventReceipt,
    IngestionEvent,
)
from potpie_context_engine.requests import (
    ProcessingStatusRequest,
    SearchRequest,
    SubmitArtifactRequest,
    SubmitEventRequest,
)


@dataclass
class _Pots:
    pots: list[PotInfo]
    active: PotInfo | None = None
    default: str | None = None
    sources: dict[str, list[SourceInfo]] | None = None

    def list_pots(self) -> list[PotInfo]:
        return self.pots

    def active_pot(self) -> PotInfo | None:
        return self.active

    def repo_default(self, *, repo: str) -> str | None:
        del repo
        return self.default

    def list_sources(self, *, pot_id: str) -> list[SourceInfo]:
        return list((self.sources or {}).get(pot_id, ()))


def _services(pots: _Pots):
    return SimpleNamespace(
        pots=pots,
        backend=SimpleNamespace(profile="embedded"),
        agent_context=SimpleNamespace(search=MagicMock(return_value={"matches": 1})),
        graph=SimpleNamespace(),
    )


@pytest.mark.anyio
async def test_selector_resolution_uses_exact_context_identity() -> None:
    first = PotInfo(pot_id="pot-1", name="first")
    second = PotInfo(pot_id="pot-2", name="second", active=True)
    resolver = LocalContextSelectorResolver(_services(_Pots([first, second], second)))

    explicit = await resolver.resolve(ContextSelector(kind="explicit", value="first"))
    active = await resolver.resolve(ContextSelector(kind="active"))

    assert isinstance(explicit, Success)
    assert explicit.value.value == "pot-1"
    assert isinstance(active, Success)
    assert active.value.value == "pot-2"


@pytest.mark.anyio
async def test_repository_selector_prefers_registered_default() -> None:
    selected = PotInfo(pot_id="pot-1", name="selected")
    resolver = LocalContextSelectorResolver(
        _services(_Pots([selected], selected, default="pot-1"))
    )

    outcome = await resolver.resolve(
        ContextSelector(kind="repository", value="github.com/acme/repo")
    )

    assert isinstance(outcome, Success)
    assert outcome.value.value == "pot-1"


@pytest.mark.anyio
async def test_local_client_executes_typed_search_against_bound_context() -> None:
    selected = PotInfo(pot_id="pot-1", name="selected", active=True)
    services = _services(_Pots([selected], selected))
    manager = build_local_resource_manager(services)
    client = LocalEngineClient(
        selector=ContextSelector(kind="explicit", value="selected"),
        authentication={"kind": "local_cli"},
        resource_manager=manager,
        coordinator=OperationCoordinator(),
    )

    outcome = await client.search(
        SearchRequest(
            query="typed boundary",
            include=("raw_graph",),
            max_items=3,
        )
    )

    assert outcome == Success({"matches": 1})
    request = services.agent_context.search.call_args.args[0]
    assert request.pot_id == "pot-1"
    assert request.query == "typed boundary"
    assert request.include == ("raw_graph",)
    assert request.max_items == 3
    assert (await manager.shutdown()) == Success(None)


class _Ingestion:
    def __init__(self) -> None:
        self.requests = []

    def submit(self, request, **options):
        self.requests.append((request, options))
        return EventReceipt(event_id="event-1", status="queued")


def _event(*, pot_id: str = "pot-1") -> IngestionEvent:
    return IngestionEvent(
        event_id="event-1",
        pot_id=pot_id,
        ingestion_kind="artifact_evidence",
        source_channel="engine",
        source_system="github",
        event_type="artifact",
        action="pull_request",
        source_id="pr-42",
        dedup_key=None,
        status="queued",
        stage="accepted",
        submitted_at=datetime.now(timezone.utc),
        started_at=None,
        completed_at=None,
        error=None,
        payload={"title": "Typed boundary"},
    )


@pytest.mark.anyio
async def test_local_evidence_operations_bind_context_and_normalize_artifacts() -> None:
    ingestion = _Ingestion()
    operations = LocalEngineOperations(
        SimpleNamespace(ingestion=ingestion, ingestion_events=None)
    )

    event = await operations.submit_event(
        ContextIdentity("pot-1"),
        SubmitEventRequest(
            source_system="github",
            event_type="change",
            action="merged",
            source_id="pr-42",
            payload={"title": "Typed boundary"},
        ),
    )
    artifact = await operations.submit_artifact(
        ContextIdentity("pot-1"),
        SubmitArtifactRequest(
            source_system="github",
            artifact_type="pull_request",
            artifact_id="42",
            artifact={"title": "Typed boundary"},
            source_ref="github:acme/repo#pull/42",
        ),
    )

    assert event == Success(EventReceipt(event_id="event-1", status="queued"))
    assert artifact == Success(EventReceipt(event_id="event-1", status="queued"))
    event_request = ingestion.requests[0][0]
    artifact_request = ingestion.requests[1][0]
    assert event_request.pot_id == "pot-1"
    assert event_request.source_id == "pr-42"
    assert artifact_request.pot_id == "pot-1"
    assert artifact_request.ingestion_kind == "artifact_evidence"
    assert artifact_request.artifact_refs == ("github:acme/repo#pull/42",)
    assert artifact_request.payload["artifact"] == {"title": "Typed boundary"}


@pytest.mark.anyio
async def test_processing_status_is_context_bound_and_typed() -> None:
    stored_event = _event()
    store = SimpleNamespace(get_event=lambda event_id: stored_event)
    operations = LocalEngineOperations(
        SimpleNamespace(ingestion=None, ingestion_events=store)
    )

    outcome = await operations.processing_status(
        ContextIdentity("pot-1"), ProcessingStatusRequest(event_id="event-1")
    )
    other_context = await operations.processing_status(
        ContextIdentity("pot-2"), ProcessingStatusRequest(event_id="event-1")
    )

    assert outcome == Success(stored_event)
    assert isinstance(other_context, Failure)
    assert other_context.error.code == "processing_status_not_found"


@pytest.mark.anyio
async def test_uncomposed_evidence_dependency_returns_typed_failure() -> None:
    operations = LocalEngineOperations(
        SimpleNamespace(ingestion=None, ingestion_events=None)
    )

    outcome = await operations.submit_event(
        ContextIdentity("pot-1"),
        SubmitEventRequest(
            source_system="github",
            event_type="change",
            action="merged",
            source_id="pr-42",
        ),
    )

    assert isinstance(outcome, Failure)
    assert outcome.error.category == "dependency"
    assert outcome.error.code == "unavailable"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("method", "input_request"),
    (
        ("submit_event", SubmitEventRequest()),
        ("submit_artifact", SubmitArtifactRequest()),
    ),
)
async def test_invalid_ingestion_requests_return_domain_failures(
    method: str,
    input_request: SubmitEventRequest | SubmitArtifactRequest,
) -> None:
    operations = LocalEngineOperations(
        SimpleNamespace(ingestion=_Ingestion(), ingestion_events=None)
    )

    outcome = await getattr(operations, method)(ContextIdentity("pot-1"), input_request)

    assert isinstance(outcome, Failure)
    assert outcome.error.category == "domain"
    assert outcome.error.code == "validation_error"
