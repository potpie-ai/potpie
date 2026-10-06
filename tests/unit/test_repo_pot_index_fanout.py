"""Repo→pot resolution costs one pot-service call, not one call per pot.

``potpie status`` and every command that resolves its pot from the working tree
(``resolve_pot_scope``, and the typed engine's repository selector) used to ask
each pot for its sources. That is a read per pot, so a caller with N pots paid N
of them before the command's real work began. Resolution now reads the repo
source index in one call and matches the working tree itself, and walks pot by
pot only for a pot service that does not serve the index.
"""

from __future__ import annotations

import pytest

from potpie.cli.commands import _common
from potpie.runtime import ContextSelector
from potpie.runtime.local_engine import LocalContextSelectorResolver
from potpie_context_engine import Failure, Success

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_runtime():
    yield
    _common.set_runtime(None)


class _Pot:
    def __init__(
        self, pot_id: str, name: str, active: bool = False, archived: bool = False
    ) -> None:
        self.pot_id = pot_id
        self.name = name
        self.active = active
        self.archived = archived


class _Source:
    def __init__(self, name: str, location: str | None = None, kind: str = "repo"):
        self.source_id = f"src_{name}"
        self.kind = kind
        self.name = name
        self.location = location or name


class _RepoSourceRow:
    def __init__(self, pot_id: str, pot_name: str, name: str, location: str) -> None:
        self.pot_id = pot_id
        self.pot_name = pot_name
        self.name = name
        self.location = location


class _LegacyPots:
    """A pot service that predates the index, counting its traffic."""

    def __init__(self, pots, sources_by_pot, active=None) -> None:
        self._pots = pots
        self._sources = sources_by_pot
        self._active = active
        self.index_calls = 0
        self.source_calls: list[str] = []
        self.pot_list_calls = 0

    def list_pots(self):
        self.pot_list_calls += 1
        return self._pots

    def active_pot(self):
        return self._active

    def repo_default(self, *, repo):
        del repo
        return None

    def list_sources(self, *, pot_id):
        self.source_calls.append(pot_id)
        return self._sources.get(pot_id, [])


class _Pots(_LegacyPots):
    """A pot service that serves the repo source index."""

    def list_repo_sources(self):
        self.index_calls += 1
        return [
            _RepoSourceRow(pot.pot_id, pot.name, source.name, source.location)
            for pot in self._pots
            if not pot.archived
            for source in self._sources.get(pot.pot_id, [])
            if source.kind == "repo"
        ]


class _Host:
    def __init__(self, pots_service) -> None:
        self.pots = pots_service


def _many_pots(count: int, *, linked_index: int) -> tuple[list[_Pot], dict]:
    pots = [_Pot(f"p{i}", f"pot-{i}") for i in range(count)]
    sources = {
        pot.pot_id: [_Source(f"github.com/acme/other-{i}")]
        for i, pot in enumerate(pots)
    }
    sources[pots[linked_index].pot_id] = [_Source("github.com/acme/shop")]
    return pots, sources


@pytest.fixture()
def in_shop_repo(monkeypatch) -> None:
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )


def test_resolution_makes_one_index_call_whatever_the_pot_count(in_shop_repo) -> None:
    pots, sources = _many_pots(50, linked_index=37)
    service = _Pots(pots, sources, active=None)

    pot_id, resolved_via = _common.resolve_pot_scope(_Host(service))

    assert (pot_id, resolved_via) == ("p37", "linked_repo")
    assert service.index_calls == 1
    assert service.source_calls == []
    assert service.pot_list_calls == 0


def test_resolution_falls_back_to_the_per_pot_walk_without_the_index(
    in_shop_repo,
) -> None:
    pots, sources = _many_pots(5, linked_index=2)
    service = _LegacyPots(pots, sources, active=None)

    pot_id, resolved_via = _common.resolve_pot_scope(_Host(service))

    assert (pot_id, resolved_via) == ("p2", "linked_repo")
    assert service.source_calls == [pot.pot_id for pot in pots]


def test_the_fallback_walk_skips_archived_pots(in_shop_repo) -> None:
    """Without the index the walk applies the index's own rule: an archived pot
    is not a routing candidate."""
    pots = [_Pot("p1", "old", archived=True), _Pot("p2", "new")]
    match = _Source("github.com/acme/shop")
    service = _LegacyPots(pots, {"p1": [match], "p2": [match]}, active=None)

    assert _common.resolve_pot_scope(_Host(service)) == ("p2", "linked_repo")
    assert service.source_calls == ["p2"]


def test_index_keeps_path_matching_client_side(monkeypatch, tmp_path) -> None:
    """A repo registered by parent path still matches a nested working tree.

    Whether a path contains the cwd is a client-side fact; pushing the match
    into the pot service would have lost it.
    """
    workdir = tmp_path / "repo" / "packages" / "api"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(_common, "_current_git_remote", lambda cwd: None)
    pots = [_Pot("p1", "monorepo")]
    service = _Pots(pots, {"p1": [_Source(str(tmp_path / "repo"))]}, active=None)

    assert _common.resolve_pot_scope(_Host(service)) == ("p1", "linked_repo")
    assert service.index_calls == 1


def test_a_pot_with_two_matching_sources_is_reported_once(in_shop_repo) -> None:
    pots = [_Pot("p1", "shop"), _Pot("p2", "shop-fork")]
    service = _Pots(
        pots,
        {
            "p1": [
                _Source("github.com/acme/shop"),
                _Source("shop-mirror", "https://github.com/acme/shop.git"),
            ],
            "p2": [_Source("github.com/acme/shop")],
        },
        active=None,
    )

    matches = _common._pots_matching_current_repo(_Host(service))

    assert matches == [("p1", "shop"), ("p2", "shop-fork")]


def test_index_rows_that_are_not_repo_sources_never_reach_matching(
    in_shop_repo,
) -> None:
    """Non-repo sources stay out of the index, so they cannot match a repo."""
    pots = [_Pot("p1", "shop")]
    service = _Pots(
        pots,
        {"p1": [_Source("github.com/acme/shop", kind="github")]},
        active=_Pot("p1", "shop", True),
    )

    assert _common._pots_matching_current_repo(_Host(service)) == []
    assert _common.resolve_pot_scope(_Host(service)) == ("p1", "active_pot")


# --- the typed engine's repository selector follows the same rule -----------


async def _select(service, selector: ContextSelector):
    return await LocalContextSelectorResolver(_Host(service)).resolve(selector)


@pytest.mark.anyio
async def test_the_repository_selector_reads_the_index_once() -> None:
    pots, sources = _many_pots(50, linked_index=37)
    service = _Pots(pots, sources, active=None)

    outcome = await _select(
        service, ContextSelector(kind="repository", value="github.com/acme/shop")
    )

    assert isinstance(outcome, Success)
    assert outcome.value.value == "p37"
    assert service.index_calls == 1
    assert service.source_calls == []


@pytest.mark.anyio
async def test_the_repository_selector_reports_ambiguity_once_per_pot() -> None:
    pots = [_Pot("p1", "shop"), _Pot("p2", "shop-fork")]
    match = _Source("github.com/acme/shop")
    service = _Pots(pots, {"p1": [match, match], "p2": [match]}, active=None)

    outcome = await _select(
        service, ContextSelector(kind="repository", value="github.com/acme/shop")
    )

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "ambiguous_pot"
    assert outcome.error.message.count("shop (p1)") == 1


@pytest.mark.anyio
async def test_an_explicit_selector_refuses_an_archived_pot_by_name() -> None:
    service = _Pots([_Pot("p1", "dead", archived=True)], {}, active=None)

    outcome = await _select(service, ContextSelector(kind="explicit", value="dead"))

    assert isinstance(outcome, Failure)
    assert outcome.error.code == "pot_archived"
    assert "--archived" in (outcome.error.recommended_next_action or "")


@pytest.mark.anyio
async def test_an_explicit_selector_names_an_archived_pot_by_its_exact_id() -> None:
    """Ids are never reused, so an exact id is how a retired pot's graph state
    is still reached for clearing; the authorizer limits what may run on it."""
    service = _Pots([_Pot("p1", "dead", archived=True)], {}, active=None)

    outcome = await _select(service, ContextSelector(kind="explicit", value="p1"))

    assert isinstance(outcome, Success)
    assert outcome.value.value == "p1"


@pytest.mark.anyio
async def test_an_explicit_selector_prefers_the_live_pot_sharing_a_name() -> None:
    service = _Pots(
        [_Pot("p1", "app", archived=True), _Pot("p2", "app")], {}, active=None
    )

    outcome = await _select(service, ContextSelector(kind="explicit", value="app"))

    assert isinstance(outcome, Success)
    assert outcome.value.value == "p2"
