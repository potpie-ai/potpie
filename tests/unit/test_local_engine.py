from __future__ import annotations

# ruff: noqa: S101 - pytest unit tests use assertions intentionally.

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from potpie.product.ports.pot_management import PotInfo, SourceInfo
from potpie.runtime import ContextSelector, LocalEngineClient
from potpie.runtime.local_engine import (
    LocalContextSelectorResolver,
    build_local_resource_manager,
)
from potpie_context_engine import Success
from potpie_context_engine.requests import SearchRequest


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
    request = services.agent_context.search.call_args.args[0]
    assert request.pot_id == "pot-1"
    assert request.query == "typed boundary"
    assert request.include == ("raw_graph",)
    assert request.max_items == 3
    assert (await manager.shutdown()) == Success(None)
