"""CLI contract coverage for source registration."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, pots
from potpie.cli.repo_location import (
    REPO_MATCH_CONTAINED,
    REPO_MATCH_SELF,
    classify_repo_source_match,
)
from potpie_context_engine.adapters.outbound.pots.local_pot_store import LocalPotStore
from potpie_context_engine.application.services.pot_management import (
    LocalPotManagementService,
)
from potpie_context_engine.domain.ports.services.pot_management import (
    PotInfo,
    SourceInfo,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_cli_state():
    yield
    _common.set_json(False)
    _common.set_host(None)


@dataclass
class _Pots:
    calls: list[dict[str, str | None]] = field(default_factory=list)
    repo_defaults: dict[str, str] = field(default_factory=dict)

    def list_pots(self) -> list[PotInfo]:
        return [PotInfo(pot_id="pot-1", name="default", active=True)]

    def active_pot(self) -> PotInfo:
        return PotInfo(pot_id="pot-1", name="default", active=True)

    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> SourceInfo:
        self.calls.append(
            {"pot_id": pot_id, "kind": kind, "location": location, "name": name}
        )
        return SourceInfo(source_id="src-1", kind=kind, name=name or location)

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        self.repo_defaults[repo] = pot_id


@dataclass
class _Host:
    pots: _Pots


def test_source_add_plain_output_is_registration_only() -> None:
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))

    result = CliRunner().invoke(
        pots.source_app, ["add", "repo", "owner/repo", "--pot", "pot-1"]
    )

    assert result.exit_code == 0, result.output
    assert fake_pots.calls == [
        {
            "pot_id": "pot-1",
            "kind": "repo",
            "location": "owner/repo",
            "name": None,
        }
    ]
    assert "registered source repo:owner/repo (src-1)" in result.output
    assert "no ingestion or scan started" in result.output
    assert fake_pots.repo_defaults == {"owner/repo": "pot-1"}


def test_source_add_json_marks_registration_only() -> None:
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app,
        [
            "add",
            "repo",
            "owner/repo",
            "--name",
            "platform",
            "--pot",
            "pot-1",
            "--no-default",
        ],
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    assert emitted == {
        "source_id": "src-1",
        "kind": "repo",
        "name": "platform",
        "location": "owner/repo",
        "pot_id": "pot-1",
        "registration_only": True,
        "repo_default_set": False,
        "repo_key": "owner/repo",
    }
    assert fake_pots.repo_defaults == {}


# ---------------------------------------------------------------------------
# kind dispatch — every accepted kind has a handler, everything else exits 1
# ---------------------------------------------------------------------------


def test_source_add_registers_an_integration_kind() -> None:
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app, ["add", "notion", "https://notion.so/wiki", "--pot", "pot-1"]
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    assert emitted["kind"] == "notion"
    assert emitted["repo_default_set"] is False
    assert "requested_kind" not in emitted
    assert fake_pots.repo_defaults == {}


def test_source_add_canonicalizes_git_hosts_to_repo() -> None:
    """`github` stores as `repo` — repo-default matching only sees that kind."""
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app, ["add", "github", "owner/repo", "--pot", "pot-1"]
    )

    assert result.exit_code == 0, result.output
    emitted = json.loads(result.output)
    assert emitted["kind"] == "repo"
    assert emitted["requested_kind"] == "github"
    assert fake_pots.calls == [
        {"pot_id": "pot-1", "kind": "repo", "location": "owner/repo", "name": None}
    ]
    assert fake_pots.repo_defaults == {"owner/repo": "pot-1"}


def test_source_add_canonicalization_is_reported_in_human_output() -> None:
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))

    result = CliRunner().invoke(
        pots.source_app, ["add", "gitlab", "owner/repo", "--pot", "pot-1"]
    )

    assert result.exit_code == 0, result.output
    assert "kind 'gitlab' registered as 'repo'" in result.output


@pytest.mark.parametrize("kind", ["document", "pdf", "spreadsheet", "csv", "md"])
def test_source_add_routes_document_kinds_to_resource_import(kind: str) -> None:
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app, ["add", kind, "./q3.pdf", "--pot", "pot-1"]
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    assert emitted["code"] == "source_kind_is_a_document"
    assert "potpie resource import" in emitted["recommended_next_action"]
    assert fake_pots.calls == []


def test_source_add_rejects_an_unknown_kind() -> None:
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app, ["add", "kafka-topic", "orders", "--pot", "pot-1"]
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    assert emitted["code"] == "unknown_source_kind"
    assert "repo" in emitted["detail"]["kinds"]
    assert fake_pots.calls == []


def test_source_add_rejects_explicit_default_on_a_non_repo_kind() -> None:
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app,
        ["add", "linear", "team/PLAT", "--pot", "pot-1", "--default"],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.output)
    assert emitted["code"] == "repo_default_not_applicable"
    assert fake_pots.calls == []


def test_source_add_non_repo_kind_without_the_flag_still_registers() -> None:
    """--default defaults to unset, so a non-repo kind is not tripped by it."""
    fake_pots = _Pots()
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app, ["add", "jira", "PLAT", "--pot", "pot-1"]
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["kind"] == "jira"


def test_source_add_repo_default_reports_unavailable_host() -> None:
    fake_pots = _Pots()
    fake_pots.set_repo_default = None  # type: ignore[method-assign]
    _common.set_host(_Host(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app,
        ["add", "repo", "owner/repo", "--pot", "pot-1"],
    )

    assert result.exit_code != 0
    emitted = json.loads(result.output)
    assert emitted["code"] == "repo_default_unavailable"
    assert fake_pots.calls == []


# ---------------------------------------------------------------------------
# source status — audit-10: no-ID per-pot summary and enriched single-source
# ---------------------------------------------------------------------------


@dataclass
class _StatusPots:
    """Fake pots service with list_sources, source_status, and repo_default."""

    _sources: list[SourceInfo] = field(default_factory=list)
    repo_defaults: dict[str, str] = field(default_factory=dict)

    def list_pots(self) -> list[PotInfo]:
        return [PotInfo(pot_id="pot-1", name="default", active=True)]

    def active_pot(self) -> PotInfo:
        return PotInfo(pot_id="pot-1", name="default", active=True)

    def list_sources(self, *, pot_id: str) -> list[SourceInfo]:
        return self._sources

    def source_status(self, *, pot_id: str, source_id: str) -> SourceInfo:
        for s in self._sources:
            if s.source_id == source_id:
                return s
        raise ValueError(f"no source {source_id}")

    def repo_default(self, *, repo: str) -> str | None:
        return self.repo_defaults.get(repo)

    def add_source(
        self, *, pot_id: str, kind: str, location: str, name: str | None = None
    ) -> SourceInfo:
        src = SourceInfo(
            source_id="src-new",
            kind=kind,
            name=name or location,
            location=location,
        )
        self._sources.append(src)
        return src

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        self.repo_defaults[repo] = pot_id


@dataclass
class _StatusHost:
    pots: _StatusPots
    graph: object = None


def test_source_status_no_id_returns_pot_summary() -> None:
    """No-ID invocation returns per-pot summary with all sources and pot info."""
    src = SourceInfo(
        source_id="src-1",
        kind="repo",
        name="acme/shop",
        location="github.com/acme/shop",
        status="ok",
    )
    fake_pots = _StatusPots(_sources=[src])
    _common.set_host(_StatusHost(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["status", "--pot", "pot-1"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pot_id"] == "pot-1"
    assert payload["source_count"] == 1
    assert len(payload["sources"]) == 1
    row = payload["sources"][0]
    assert row["id"] == "src-1"
    assert row["kind"] == "repo"
    assert row["location"] == "github.com/acme/shop"
    assert row["registration_only"] is True
    assert row["ingestion_status"] == "not_started"
    assert "claim_count" in payload


def test_source_status_no_id_marks_repo_default() -> None:
    """Source whose location is the pot's repo default is marked repo_default=True."""
    src = SourceInfo(
        source_id="src-1",
        kind="repo",
        name="acme/shop",
        location="github.com/acme/shop",
        status="ok",
    )
    fake_pots = _StatusPots(_sources=[src])
    fake_pots.repo_defaults["github.com/acme/shop"] = "pot-1"
    _common.set_host(_StatusHost(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["status", "--pot", "pot-1"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["sources"][0]["repo_default"] is True


def test_source_status_no_id_no_sources_recommends_add() -> None:
    """Per-pot summary with no sources includes a recommended_next_action hint."""
    fake_pots = _StatusPots(_sources=[])
    _common.set_host(_StatusHost(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(pots.source_app, ["status", "--pot", "pot-1"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["source_count"] == 0
    assert payload["recommended_next_action"] is not None
    assert "source add repo" in payload["recommended_next_action"]


def test_source_status_with_id_returns_enriched_row() -> None:
    """Providing a source-id returns a single enriched row, not the old 3-field shape."""
    src = SourceInfo(
        source_id="src-abc",
        kind="repo",
        name=".",
        location="/home/user/project",
        status="ok",
    )
    fake_pots = _StatusPots(_sources=[src])
    _common.set_host(_StatusHost(pots=fake_pots))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app, ["status", "src-abc", "--pot", "pot-1"]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["id"] == "src-abc"
    assert payload["kind"] == "repo"
    assert payload["location"] == "/home/user/project"
    assert payload["registration_only"] is True
    assert payload["ingestion_status"] == "not_started"
    assert "status" in payload


def test_source_status_no_id_human_output_contains_kind_and_location() -> None:
    """Plain-text no-ID output shows kind, location, and registration-only hint."""
    src = SourceInfo(
        source_id="src-1",
        kind="repo",
        name="shop",
        location="github.com/acme/shop",
        status="ok",
    )
    fake_pots = _StatusPots(_sources=[src])
    _common.set_host(_StatusHost(pots=fake_pots))

    result = CliRunner().invoke(pots.source_app, ["status", "--pot", "pot-1"])

    assert result.exit_code == 0, result.output
    assert "repo" in result.output
    assert "github.com/acme/shop" in result.output
    assert "registration-only" in result.output


# ---------------------------------------------------------------------------
# The registry itself — real ``LocalPotManagementService`` over a real
# ``LocalPotStore``. The fakes above answer whatever the CLI asks them, which is
# exactly why they could not see these: a hardcoded ``status`` is invisible
# until something *stores* a different one, and a duplicate row is invisible
# until a store keeps both.
# ---------------------------------------------------------------------------


class _Mutation:
    def reset_pot(self, pot_id: str) -> dict[str, object]:
        del pot_id
        return {"ok": True}


@dataclass(slots=True)
class _Backend:
    mutation: _Mutation


class _RealHost:
    def __init__(self, pots_service: object) -> None:
        self.pots = pots_service


class _BindingRefused:
    """The real registry, with only the repo-default binding refusing.

    ``LocalPotManagementService`` is a slotted dataclass, so the failure is
    injected by wrapping rather than by patching a method onto the instance.
    Everything the registration path touches — ``add_source``, the store, the
    lifecycle guards — is still the real thing.
    """

    def __init__(self, inner: LocalPotManagementService) -> None:
        self._inner = inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def set_repo_default(self, *, repo: str, pot_id: str) -> None:
        raise RuntimeError("control plane refused the binding")


@pytest.fixture()
def registry(tmp_path, monkeypatch) -> LocalPotManagementService:
    # The host registry reads origin state from the home dir; keep it in tmp so
    # the test never consults the developer's real ~/.potpie.
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "home"))
    return LocalPotManagementService(
        store=LocalPotStore(home=tmp_path / "home"),
        backend=_Backend(_Mutation()),
        resources=None,
    )


def _run(registry: object, *args: str):
    _common.set_host(_RealHost(registry))
    _common.set_json(True)
    return CliRunner().invoke(pots.source_app, list(args))


def _stored_rows(registry: LocalPotManagementService, pot_id: str) -> list[dict]:
    return registry.store.list_sources(pot_id=pot_id)


def _mark_stored(
    registry: LocalPotManagementService, pot_id: str, source_id: str, **fields: str
) -> None:
    """Update a stored source row the way a later assessor or ingestor would.

    Written straight to the state file rather than through a service method:
    nothing in the product marks a source stale or ingested yet, and the point
    of the assertions below is that the *reader* reports whatever the row says.
    """
    path = registry.store.home / "pots.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    for row in state["sources"][pot_id]:
        if row["source_id"] == source_id:
            row.update(fields)
    path.write_text(json.dumps(state), encoding="utf-8")


def _git_tree(root: Path, remote: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)  # noqa: S603, S607
    subprocess.run(  # noqa: S603, S607
        ["git", "-C", str(root), "remote", "add", "origin", remote], check=True
    )
    return root


# --- S5-19/20: the row's own status, not a literal ---------------------------


def test_source_status_reports_the_stored_status(registry) -> None:
    """``status`` was the literal ``ok``, so a broken source read as healthy."""
    pot = registry.create_pot(name="shop", use=True)
    src = registry.add_source(pot_id=pot.pot_id, kind="linear", location="team/PLAT")
    _mark_stored(
        registry,
        pot.pot_id,
        src.source_id,
        status="error",
        ingestion_status="completed",
    )

    summary = json.loads(_run(registry, "status", "--pot", pot.pot_id).output)
    row = summary["sources"][0]

    assert row["status"] == "error"
    assert row["ingestion_status"] == "completed"
    # An ingested source is not "registration only" — that label is what the
    # ingestion status *means*, not an independent constant.
    assert row["registration_only"] is False


def test_single_source_status_reports_the_stored_status(registry) -> None:
    pot = registry.create_pot(name="shop", use=True)
    src = registry.add_source(pot_id=pot.pot_id, kind="url", location="https://rb/x")
    _mark_stored(
        registry, pot.pot_id, src.source_id, status="stale", ingestion_status="running"
    )

    payload = json.loads(
        _run(registry, "status", src.source_id, "--pot", pot.pot_id).output
    )

    assert payload["status"] == "stale"
    assert payload["ingestion_status"] == "running"
    assert payload["registration_only"] is False


def test_a_freshly_registered_source_says_registered_and_not_ingested(
    registry,
) -> None:
    pot = registry.create_pot(name="shop", use=True)
    registry.add_source(pot_id=pot.pot_id, kind="jira", location="PLAT")

    row = json.loads(_run(registry, "status", "--pot", pot.pot_id).output)["sources"][0]

    assert row["status"] == "registered"
    assert row["ingestion_status"] == "not_started"
    assert row["registration_only"] is True


# --- S5-18: a missing source is a missing source -----------------------------


def test_status_of_an_unknown_source_id_names_the_source_repair(registry) -> None:
    """``pot_not_found`` sent the operator to ``pot list`` for a pot that resolved."""
    pot = registry.create_pot(name="shop", use=True)

    result = _run(registry, "status", "src_deadbeef", "--pot", pot.pot_id)

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "source_not_found"
    assert "src_deadbeef" in payload["message"]
    assert "potpie source list" in payload["recommended_next_action"]
    assert "pot list" not in payload["recommended_next_action"]


# --- S5-22/23: a registration that identifies nothing is not a registration ---


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_location_registers_nothing(registry, blank: str) -> None:
    pot = registry.create_pot(name="shop", use=True)

    result = _run(registry, "add", "linear", blank, "--pot", pot.pot_id)

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    assert json.loads(result.output)["code"] == "missing_source_location"
    assert _stored_rows(registry, pot.pot_id) == []


def test_the_control_plane_refuses_a_blank_location_too(registry) -> None:
    """The CLI is not the only caller; the registry itself has to hold the line."""
    pot = registry.create_pot(name="shop", use=True)

    with pytest.raises(ValueError, match="cannot be empty"):
        registry.add_source(pot_id=pot.pot_id, kind="linear", location="  ")

    assert _stored_rows(registry, pot.pot_id) == []


def test_a_failed_repo_add_leaves_no_registration_behind(registry, tmp_path) -> None:
    """The row was written *before* the identity check that then failed.

    Exit 1 left a junk ``repo`` row with a blank location, and dedup matched it
    on every retry — so the registration could never be made again.
    """
    pot = registry.create_pot(name="shop", use=True)

    result = _run(registry, "add", "repo", "   ", "--pot", pot.pot_id)

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    assert _stored_rows(registry, pot.pot_id) == []

    # …and the retry that follows a failure actually registers.
    tree = _git_tree(tmp_path / "shop", "git@github.com:acme/shop.git")
    retry = _run(registry, "add", "repo", str(tree), "--pot", pot.pot_id)
    assert retry.exit_code == 0, retry.output
    assert [row["location"] for row in _stored_rows(registry, pot.pot_id)] == [
        str(tree)
    ]


def test_a_repo_add_whose_default_binding_fails_writes_no_row(
    registry, tmp_path
) -> None:
    """The row exists for the binding; it must not outlive the binding failing."""
    tree = _git_tree(tmp_path / "shop", "git@github.com:acme/shop.git")
    pot = registry.create_pot(name="shop", use=True)

    result = _run(
        _BindingRefused(registry), "add", "repo", str(tree), "--pot", pot.pot_id
    )

    assert result.exit_code != 0
    assert _stored_rows(registry, pot.pot_id) == []


# --- S5-28: registration is idempotent for every kind ------------------------


@pytest.mark.parametrize(
    ("kind", "location"),
    [("linear", "team/PLAT"), ("jira", "PLAT"), ("url", "https://runbook/x")],
)
def test_registering_the_same_non_repo_source_twice_keeps_one_row(
    registry, kind: str, location: str
) -> None:
    pot = registry.create_pot(name="shop", use=True)

    first = json.loads(
        _run(registry, "add", kind, location, "--pot", pot.pot_id).output
    )
    second = json.loads(
        _run(registry, "add", kind, location, "--pot", pot.pot_id).output
    )

    assert len(_stored_rows(registry, pot.pot_id)) == 1
    assert second["source_id"] == first["source_id"]
    assert second["already_registered"] is True
    assert "already_registered" not in first


def test_different_locations_of_one_kind_stay_separate_registrations(registry) -> None:
    pot = registry.create_pot(name="shop", use=True)

    _run(registry, "add", "linear", "team/PLAT", "--pot", pot.pot_id)
    _run(registry, "add", "linear", "team/GROWTH", "--pot", pot.pot_id)

    assert len(_stored_rows(registry, pot.pot_id)) == 2


# --- S5-29/30: --default writes the key routing reads ------------------------


def test_default_on_a_path_registered_repo_binds_the_routing_identity(
    registry, tmp_path, monkeypatch
) -> None:
    """``--default`` reported success and changed nothing any command read."""
    tree = _git_tree(tmp_path / "shop", "git@github.com:acme/shop.git")
    pot = registry.create_pot(name="shop", use=True)

    result = _run(registry, "add", "repo", str(tree), "--pot", pot.pot_id, "--default")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["repo_default_set"] is True
    assert registry.list_repo_defaults() == {"github.com/acme/shop": pot.pot_id}

    monkeypatch.chdir(tree)
    routing = _common.repo_effective_pot_info(_RealHost(registry))
    assert routing["default_pot_id"] == pot.pot_id
    assert routing["reason"] == "repo_default"


def test_a_path_registered_repo_is_status_marked_as_the_repo_default(
    registry, tmp_path
) -> None:
    tree = _git_tree(tmp_path / "shop", "git@github.com:acme/shop.git")
    pot = registry.create_pot(name="shop", use=True)
    _run(registry, "add", "repo", str(tree), "--pot", pot.pot_id, "--default")

    row = json.loads(_run(registry, "status", "--pot", pot.pot_id).output)["sources"][0]

    assert row["repo_default"] is True


def test_the_same_repo_by_path_and_by_remote_is_one_registration(
    registry, tmp_path
) -> None:
    tree = _git_tree(tmp_path / "shop", "git@github.com:acme/shop.git")
    pot = registry.create_pot(name="shop", use=True)

    _run(registry, "add", "repo", str(tree), "--pot", pot.pot_id)
    _run(registry, "add", "repo", "git@github.com:acme/shop.git", "--pot", pot.pot_id)

    assert len(_stored_rows(registry, pot.pot_id)) == 1


# --- S5-48/49: a workspace root is not one of the projects under it ----------


def _classified(registry: LocalPotManagementService, cwd: Path) -> dict[str, set[str]]:
    """``{match kind -> pot names}`` for the repo index as seen from ``cwd``.

    The registry half of the workspace-root defect. Resolving the *scope* from
    this classification is a ``_common`` change (see the report's shared-file
    request); what the registry has to supply is the distinction itself, over
    real stored rows.
    """
    out: dict[str, set[str]] = {}
    for row in registry.list_repo_sources():
        match = classify_repo_source_match(row.location or "", cwd=cwd, remote=None)
        if match:
            out.setdefault(match, set()).add(row.pot_name)
    return out


def test_registered_children_are_contained_not_matches_of_the_workspace_root(
    registry, tmp_path
) -> None:
    """Two registered children make a workspace root ambiguous — not the repo.

    From ``~/work``, both ``~/work/alpha`` and ``~/work/beta`` matched, and pot
    resolution reported "Current repo is registered in multiple pots" about a
    directory that is not registered anywhere.
    """
    workspace = tmp_path / "work"
    alpha = workspace / "alpha"
    beta = workspace / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    pot_a = registry.create_pot(name="alpha", use=True)
    pot_b = registry.create_pot(name="beta")
    registry.add_source(pot_id=pot_a.pot_id, kind="repo", location=str(alpha))
    registry.add_source(pot_id=pot_b.pot_id, kind="repo", location=str(beta))

    assert _classified(registry, workspace) == {REPO_MATCH_CONTAINED: {"alpha", "beta"}}
    # …and standing in one of them is an unambiguous self-match on that pot.
    assert _classified(registry, alpha) == {REPO_MATCH_SELF: {"alpha"}}
