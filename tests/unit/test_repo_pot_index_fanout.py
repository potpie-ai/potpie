"""Repo→pot resolution costs one control-plane call, not one call per pot.

``potpie status`` and every command that resolves its pot from the working tree
(``resolve_pot_id``, and so ``graph commit``) used to ask each visible pot for
its sources. In-process that is a dict read per pot; against a hosted control
plane it is a network + database round trip per pot, so a caller with N pots
paid N of them before the command's real work began. Resolution now reads the
control plane's repo source index in one call and matches client-side, and
falls back to the per-pot walk only for a host that does not serve the index.
"""

from __future__ import annotations

import pytest

from potpie.cli.commands import _common

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_host():
    yield
    _common.set_host(None)


class _Pot:
    def __init__(self, pot_id: str, name: str, active: bool = False) -> None:
        self.pot_id = pot_id
        self.name = name
        self.active = active


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


class _Pots:
    """Control plane that serves the repo source index, and counts the traffic."""

    serves_index = True

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
        return None

    def list_sources(self, *, pot_id):
        self.source_calls.append(pot_id)
        return self._sources.get(pot_id, [])

    def list_repo_sources(self):
        if not self.serves_index:
            raise AttributeError("list_repo_sources")
        self.index_calls += 1
        return [
            _RepoSourceRow(pot.pot_id, pot.name, source.name, source.location)
            for pot in self._pots
            for source in self._sources.get(pot.pot_id, [])
            if source.kind == "repo"
        ]


class _LegacyPots(_Pots):
    """A host that predates the index — e.g. a not-yet-upgraded hosted service."""

    serves_index = False


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


def test_resolution_makes_one_index_call_whatever_the_pot_count(monkeypatch) -> None:
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    pots, sources = _many_pots(50, linked_index=37)
    service = _Pots(pots, sources, active=None)

    pot_id, resolved_via = _common.resolve_pot_scope(_Host(service))

    assert (pot_id, resolved_via) == ("p37", "linked_repo")
    assert service.index_calls == 1
    assert service.source_calls == []
    assert service.pot_list_calls == 0


def test_resolution_falls_back_to_the_per_pot_walk_without_the_index(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    pots, sources = _many_pots(5, linked_index=2)
    service = _LegacyPots(pots, sources, active=None)

    pot_id, resolved_via = _common.resolve_pot_scope(_Host(service))

    assert (pot_id, resolved_via) == ("p2", "linked_repo")
    assert service.source_calls == [pot.pot_id for pot in pots]


def test_index_keeps_path_matching_client_side(monkeypatch, tmp_path) -> None:
    """A repo registered by parent path still matches a nested working tree.

    Whether a path contains the cwd is a client-side fact; pushing the match
    into the control plane would have lost it.
    """
    workdir = tmp_path / "repo" / "packages" / "api"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(_common, "_current_git_remote", lambda cwd: None)
    pots = [_Pot("p1", "monorepo")]
    service = _Pots(pots, {"p1": [_Source(str(tmp_path / "repo"))]}, active=None)

    assert _common.resolve_pot_scope(_Host(service)) == ("p1", "linked_repo")
    assert service.index_calls == 1


def test_a_pot_with_two_matching_sources_is_reported_once(monkeypatch) -> None:
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
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
    monkeypatch,
) -> None:
    """Non-repo sources stay out of the index, so they cannot match a repo."""
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda cwd: "github.com/acme/shop"
    )
    pots = [_Pot("p1", "shop")]
    service = _Pots(
        pots,
        {"p1": [_Source("github.com/acme/shop", kind="github")]},
        active=_Pot("p1", "shop", True),
    )

    assert _common._pots_matching_current_repo(_Host(service)) == []
    assert _common.resolve_pot_scope(_Host(service)) == ("p1", "active_pot")
