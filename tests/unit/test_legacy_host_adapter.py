from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from potpie.runtime import (
    ContextSelector,
    DestructiveConfirmation,
    LocalEngineClient,
)
from potpie.runtime.legacy_host_adapter import (
    HostContextSelectorResolver,
    build_legacy_engine_client,
    build_local_resource_manager,
)
from potpie_context_engine import Failure, Success
from potpie_context_engine.domain.ports.services.pot_management import (
    PotInfo,
    SourceInfo,
)
from potpie_context_engine.requests import (
    ImportSnapshotRequest,
    ProposeRequest,
    RecordRequest,
    SearchRequest,
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


def _host(pots: _Pots):
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
    resolver = HostContextSelectorResolver(_host(_Pots([first, second], second)))

    explicit = await resolver.resolve(
        ContextSelector(kind="explicit", value="first")
    )
    active = await resolver.resolve(ContextSelector(kind="active"))

    assert isinstance(explicit, Success)
    assert explicit.value.value == "pot-1"
    assert isinstance(active, Success)
    assert active.value.value == "pot-2"


@pytest.mark.anyio
async def test_repository_selector_prefers_registered_default() -> None:
    selected = PotInfo(pot_id="pot-1", name="selected")
    resolver = HostContextSelectorResolver(
        _host(_Pots([selected], selected, default="pot-1"))
    )

    outcome = await resolver.resolve(
        ContextSelector(kind="repository", value="github.com/acme/repo")
    )

    assert isinstance(outcome, Success)
    assert outcome.value.value == "pot-1"


@pytest.mark.anyio
async def test_local_client_executes_typed_search_against_bound_context() -> None:
    selected = PotInfo(pot_id="pot-1", name="selected", active=True)
    host = _host(_Pots([selected], selected))
    manager = build_local_resource_manager(host)
    client = LocalEngineClient(
        selector=ContextSelector(kind="explicit", value="selected"),
        authentication={"kind": "local_cli"},
        resource_manager=manager,
    )

    outcome = await client.search(
        SearchRequest(
            payload={
                "query": "typed boundary",
                "include": ("raw_graph",),
                "max_items": 3,
            }
        )
    )

    assert outcome == Success({"matches": 1})
    request = host.agent_context.search.call_args.args[0]
    assert request.pot_id == "pot-1"
    assert request.query == "typed boundary"
    assert request.include == ("raw_graph",)
    assert request.max_items == 3
    assert (await manager.shutdown()) == Success(None)


@pytest.mark.anyio
async def test_temporary_daemon_adapter_uses_finite_typed_dispatch() -> None:
    selected = PotInfo(pot_id="pot-1", name="selected", active=True)
    host = _host(_Pots([selected], selected))
    client = build_legacy_engine_client(
        host=host,
        selector=ContextSelector(kind="active"),
    )

    outcome = await client.search(SearchRequest(payload={"query": "daemon"}))

    assert outcome == Success({"matches": 1})
    request = host.agent_context.search.call_args.args[0]
    assert request.pot_id == "pot-1"
    assert request.query == "daemon"


@pytest.mark.anyio
async def test_typed_writes_receive_only_the_bound_context() -> None:
    selected = PotInfo(pot_id="pot-1", name="selected", active=True)
    host = _host(_Pots([selected], selected))
    host.agent_context.record = MagicMock(return_value={"mutations_applied": 1})
    host.graph_workbench = SimpleNamespace(
        propose=MagicMock(return_value={"status": "validated"})
    )
    client = build_legacy_engine_client(
        host=host,
        selector=ContextSelector(kind="active"),
    )

    recorded = await client.record(
        RecordRequest(payload={"record_type": "feature_note", "summary": "typed"})
    )
    proposed = await client.propose(
        ProposeRequest(payload={"mutation": {"operations": []}})
    )

    assert recorded == Success({"mutations_applied": 1})
    record_request = host.agent_context.record.call_args.args[0]
    assert record_request.pot_id == "pot-1"
    assert record_request.record_type == "feature_note"
    assert proposed == Success({"status": "validated"})
    host.graph_workbench.propose.assert_called_once_with(
        {"operations": []},
        pot_id="pot-1",
        ttl_seconds=None,
    )


@pytest.mark.anyio
async def test_legacy_destructive_write_requires_exact_confirmation() -> None:
    selected = PotInfo(pot_id="pot-1", name="selected", active=True)
    host = _host(_Pots([selected], selected))
    host.backend.snapshot = SimpleNamespace(
        import_=MagicMock(return_value={"claims": 2})
    )
    client = build_legacy_engine_client(
        host=host,
        selector=ContextSelector(kind="active"),
    )
    request = ImportSnapshotRequest(payload={"source": "snapshot.json"})

    missing = await client.import_snapshot(request)
    confirmed = await client.import_snapshot(
        request,
        confirmation=DestructiveConfirmation(confirmed=True),
    )

    assert isinstance(missing, Failure)
    assert missing.error.category == "authorization"
    assert missing.error.code == "destructive_intent_invalid"
    assert confirmed == Success({"claims": 2})
    host.backend.snapshot.import_.assert_called_once_with(
        pot_id="pot-1", source="snapshot.json"
    )


@pytest.mark.anyio
async def test_missing_explicit_context_returns_typed_selection_failure() -> None:
    host = _host(_Pots([]))
    client = build_legacy_engine_client(
        host=host,
        selector=ContextSelector(kind="explicit", value="missing"),
    )

    outcome = await client.search(SearchRequest(payload={"query": "anything"}))

    assert isinstance(outcome, Failure)
    assert outcome.error.category == "selection"
    assert outcome.error.code == "pot_not_found"
    host.agent_context.search.assert_not_called()
