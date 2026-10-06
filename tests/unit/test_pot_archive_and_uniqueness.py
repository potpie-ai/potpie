"""``archived`` has to mean something, and a pot name has to mean one pot.

Two defects with one shape: a field the product wrote and never read.

*Archive* stored a flag that nothing enforced. Archived pots kept appearing in
``pot list`` with no marker and no ``archived`` key in the JSON; ``pot use``
still selected them; renames and source registrations still wrote to them; and
a repo default pointing at one kept routing every repo-scoped read and write
into it.

*Uniqueness* was enforced nowhere either. ``pot rename`` would put two pots
under one name, after which every bare ref picked an arbitrary one of them,
including the bare ref handed to ``pot reset <name> --confirm``.

These run the real ``LocalPotManagementService`` over a real ``LocalPotStore``,
because both defects live in the store's ref resolution and the service's
guards. A fake pot service would agree with whatever the CLI asked it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from potpie.cli.commands import _common, pots
from potpie.daemon.http.ui import build_ui_app
from potpie.pots.local_service import LocalPotManagementService
from potpie.pots.local_store import LocalPotStore
from potpie.runtime import (
    ContextSelector,
    DestructiveConfirmation,
    LocalEngineClient,
    OperationCoordinator,
)
from potpie.runtime.local_engine import build_local_resource_manager
from potpie_context_engine import DependencyError, Failure, Success
from potpie_context_engine.core.errors import PotArchived
from potpie_context_engine.requests import ResetContextRequest, SearchRequest
from potpie_context_engine.results import ResetContextResult

pytestmark = pytest.mark.unit

REPO = "git@example.com:acme/shop.git"


class _ResetClient:
    """Stands in for the typed engine client; records every graph reset."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, object]] = []

    async def reset_context(self, request, *, confirmation=None):
        del request
        self.calls.append((self.context_id, confirmation))
        if self.fail:
            return Failure(
                DependencyError(
                    code="unavailable",
                    message="graph store is not reachable",
                    recommended_next_action="run 'potpie daemon restart'",
                )
            )
        return Success(ResetContextResult(context_id=self.context_id, reset=True))


class _Host:
    def __init__(self, pots_service: LocalPotManagementService) -> None:
        self.pots = pots_service


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_runtime(None)
    _common.set_json(False)


@pytest.fixture()
def service(tmp_path) -> LocalPotManagementService:
    return LocalPotManagementService(
        store=LocalPotStore(home=tmp_path / "home"),
        backend=SimpleNamespace(),
    )


@pytest.fixture()
def engine(monkeypatch) -> _ResetClient:
    client = _ResetClient()

    def engine_client(context_id):
        client.context_id = context_id
        return client

    monkeypatch.setattr(pots, "get_engine_client", engine_client)
    return client


def _run(service: LocalPotManagementService, *args: str):
    _common.set_runtime(_Host(service))
    _common.set_json(True)
    return CliRunner().invoke(pots.pot_app, list(args))


def _payload(result) -> dict:
    return json.loads(result.output)


def _archive(service: LocalPotManagementService, ref: str) -> None:
    """Archive the way the CLI does once the graph reset has succeeded."""
    service.archive_pot(ref=ref)


# --- archive is a lifecycle state -------------------------------------------


def test_pot_list_hides_archived_pots_and_says_so(service) -> None:
    service.create_pot(name="live", use=True)
    service.create_pot(name="dead")
    _archive(service, "dead")

    result = _run(service, "list")

    assert result.exit_code == 0, result.output
    assert [row["name"] for row in _payload(result)["pots"]] == ["live"]

    _common.set_json(False)
    human = CliRunner().invoke(pots.pot_app, ["list"]).output
    assert "1 archived" in human and "--archived" in human

    with_archived = _run(service, "list", "--archived")
    rows = {row["name"]: row for row in _payload(with_archived)["pots"]}
    assert rows["dead"]["archived"] is True
    # The key is present on live pots too, so a consumer can tell "not
    # archived" from "this service does not report it".
    assert rows["live"]["archived"] is False


@pytest.mark.parametrize(
    "args",
    [
        ("use", "dead"),
        ("rename", "dead", "reborn"),
        ("reset", "dead", "--confirm"),
        ("default", "set", "dead", "--repo", REPO),
    ],
)
def test_archived_pots_refuse_every_command_that_targets_them(
    service, engine, args
) -> None:
    service.create_pot(name="live", use=True)
    service.create_pot(name="dead")
    _archive(service, "dead")

    result = _run(service, *args)

    assert result.exit_code == 1, result.output
    payload = _payload(result)
    # Not `pot_not_found`: the ref resolved. "No pot matching 'dead'" would
    # send the operator to a listing that hides it on purpose.
    assert payload["code"] == "pot_archived", payload
    assert "archived" in payload["message"]
    assert "--archived" in (payload["recommended_next_action"] or "")
    assert engine.calls == []


def test_source_add_refuses_an_archived_pot(service) -> None:
    service.create_pot(name="live", use=True)
    service.create_pot(name="dead")
    _archive(service, "dead")
    _common.set_runtime(_Host(service))
    _common.set_json(True)

    result = CliRunner().invoke(
        pots.source_app, ["add", "linear", "ACME", "--pot", "dead"]
    )

    assert result.exit_code == 1, result.output
    assert _payload(result)["code"] == "pot_archived"


def test_an_archived_pot_refuses_new_source_registrations(service) -> None:
    pot = service.create_pot(name="dead", use=True)
    _archive(service, "dead")

    with pytest.raises(PotArchived):
        service.add_source(pot_id=pot.pot_id, kind="repo", location="/src/app")


def test_an_archived_pot_stops_being_the_repo_default(service) -> None:
    """The stale pointer is the quiet one: reads come back empty, not wrong."""
    pot = service.create_pot(name="dead", use=True)
    service.set_repo_default(repo=REPO, pot_id=pot.pot_id)
    assert service.repo_default(repo=REPO) == pot.pot_id

    _archive(service, "dead")

    assert service.repo_default(repo=REPO) is None


def test_an_archived_pot_is_not_a_repo_source_candidate(service) -> None:
    pot = service.create_pot(name="dead", use=True)
    service.add_source(pot_id=pot.pot_id, kind="repo", location="/src/app")
    assert [row.pot_id for row in service.list_repo_sources()] == [pot.pot_id]

    _archive(service, "dead")

    assert service.list_repo_sources() == []


def test_repo_scoped_resolution_skips_an_archived_repo_default(
    service, monkeypatch
) -> None:
    monkeypatch.setattr(_common, "_current_repo_identity", lambda: REPO)
    live = service.create_pot(name="live")
    dead = service.create_pot(name="dead")
    service.set_repo_default(repo=REPO, pot_id=dead.pot_id)
    _archive(service, "dead")
    service.use_pot(ref="live")

    assert _common.resolve_pot_scope(_Host(service)) == (live.pot_id, "active_pot")


def test_creating_a_pot_named_after_an_archived_one_starts_fresh(service) -> None:
    """Reuse-by-name must not hand back a pot whose graph state was cleared."""
    old = service.create_pot(name="app", use=True)
    _archive(service, "app")

    new = service.create_pot(name="app", use=True)

    assert new.pot_id != old.pot_id
    assert new.archived is False
    assert new.created is True


def test_a_live_pot_wins_a_name_shared_with_an_archived_one(service) -> None:
    """The archived pot is older, so it comes first in the store; it must still
    not answer the name the live pot now holds."""
    service.create_pot(name="app", use=True)
    _archive(service, "app")
    new = service.create_pot(name="app")

    assert service.use_pot(ref="app").pot_id == new.pot_id
    assert _common.resolve_pot_scope(_Host(service), "app") == (
        new.pot_id,
        "explicit",
    )


def test_create_says_when_it_reused_an_existing_pot(service) -> None:
    first = service.create_pot(name="app")

    result = _run(service, "create", "app")

    assert result.exit_code == 0, result.output
    payload = _payload(result)
    assert payload["id"] == first.pot_id
    assert payload["created"] is False
    assert first.created is True

    _common.set_json(False)
    human = CliRunner().invoke(pots.pot_app, ["create", "app"]).output
    assert "using existing pot 'app'" in human


# --- archive clears the graph first, then retires the pot -------------------


def test_archive_resets_the_graph_before_retiring_the_pot(service, engine) -> None:
    service.create_pot(name="live", use=True)
    dead = service.create_pot(name="dead")

    result = _run(service, "archive", "dead", "--confirm")

    assert result.exit_code == 0, result.output
    assert _payload(result) == {
        "id": dead.pot_id,
        "name": "dead",
        "archived": True,
        "already_archived": False,
        "graph_reset": True,
    }
    assert engine.calls == [(dead.pot_id, DestructiveConfirmation(confirmed=True))]
    archived = {p.pot_id: p.archived for p in service.list_pots()}
    assert archived[dead.pot_id] is True


def test_archive_without_confirmation_changes_nothing(service, engine) -> None:
    service.create_pot(name="dead", use=True)

    result = _run(service, "archive", "dead")

    assert result.exit_code == 1, result.output
    assert _payload(result)["code"] == "destructive_confirmation_required"
    assert engine.calls == []
    assert [p.archived for p in service.list_pots()] == [False]


def test_a_failed_graph_reset_leaves_the_pot_live(service, engine) -> None:
    """Flagging a pot whose graph survived would hide data nothing can clear."""
    service.create_pot(name="dead", use=True)
    engine.fail = True

    result = _run(service, "archive", "dead", "--confirm")

    assert result.exit_code == 2, result.output
    assert _payload(result)["code"] == "unavailable"
    assert [p.archived for p in service.list_pots()] == [False]


# --- archive is idempotent: it can clear an archived pot again ---------------


def _legacy_archived(service: LocalPotManagementService, name: str):
    """A pot archived by flag only, the way archive worked before it cleared
    graph state: hidden from every command, with its data still in place."""
    pot = service.create_pot(name=name)
    service.archive_pot(ref=pot.pot_id)
    return pot


def test_archiving_an_archived_pot_clears_its_graph_state_again(
    service, engine
) -> None:
    service.create_pot(name="live", use=True)
    dead = _legacy_archived(service, "dead")

    result = _run(service, "archive", dead.pot_id, "--confirm")

    assert result.exit_code == 0, result.output
    assert _payload(result) == {
        "id": dead.pot_id,
        "name": "dead",
        "archived": True,
        "already_archived": True,
        "graph_reset": True,
    }
    assert engine.calls == [(dead.pot_id, DestructiveConfirmation(confirmed=True))]
    archived = {p.pot_id: p.archived for p in service.list_pots()}
    assert archived[dead.pot_id] is True

    _common.set_json(False)
    human = CliRunner().invoke(pots.pot_app, ["archive", "dead", "--confirm"]).output
    assert "already archived" in human


def test_re_archiving_still_requires_confirmation(service, engine) -> None:
    dead = _legacy_archived(service, "dead")

    result = _run(service, "archive", dead.pot_id)

    assert result.exit_code == 1, result.output
    assert _payload(result)["code"] == "destructive_confirmation_required"
    assert engine.calls == []


def test_archive_by_name_targets_the_live_pot_when_names_are_shared(
    service, engine
) -> None:
    """Re-archiving reaches an archived pot by name only when no live pot
    answers that name; its id always reaches it."""
    old = _legacy_archived(service, "app")
    new = service.create_pot(name="app", use=True)

    by_name = _run(service, "archive", "app", "--confirm")
    assert _payload(by_name)["id"] == new.pot_id
    assert _payload(by_name)["already_archived"] is False

    by_id = _run(service, "archive", old.pot_id, "--confirm")
    assert _payload(by_id)["id"] == old.pot_id
    assert _payload(by_id)["already_archived"] is True


def test_a_name_shared_by_two_archived_pots_must_be_narrowed_to_an_id(
    service, engine
) -> None:
    first = _legacy_archived(service, "app")
    second = _legacy_archived(service, "app")

    result = _run(service, "archive", "app", "--confirm")

    assert result.exit_code == 1, result.output
    payload = _payload(result)
    assert payload["code"] == "ambiguous_pot"
    assert first.pot_id in payload["message"] and second.pot_id in payload["message"]
    assert engine.calls == []


# --- the typed engine boundary admits only a reset on an archived pot -------


def _engine_client(service: LocalPotManagementService, selector: str):
    services = SimpleNamespace(
        pots=service,
        backend=SimpleNamespace(
            profile="embedded",
            mutation=SimpleNamespace(
                reset_pot=MagicMock(side_effect=lambda pot_id: {"pot_id": pot_id})
            ),
        ),
        agent_context=SimpleNamespace(search=MagicMock(return_value={"matches": 0})),
        graph=SimpleNamespace(),
    )
    client = LocalEngineClient(
        selector=ContextSelector(kind="explicit", value=selector),
        authentication={"kind": "local_cli"},
        resource_manager=build_local_resource_manager(services),
        coordinator=OperationCoordinator(),
    )
    return client, services


@pytest.mark.anyio
async def test_an_archived_pot_id_reaches_only_the_graph_reset(service) -> None:
    dead = _legacy_archived(service, "dead")
    client, services = _engine_client(service, dead.pot_id)

    searched = await client.search(SearchRequest(query="anything"))
    reset = await client.reset_context(
        ResetContextRequest(), confirmation=DestructiveConfirmation(confirmed=True)
    )

    assert isinstance(searched, Failure)
    assert searched.error.code == "pot_archived"
    services.agent_context.search.assert_not_called()
    assert isinstance(reset, Success)
    services.backend.mutation.reset_pot.assert_called_once_with(dead.pot_id)


@pytest.mark.anyio
async def test_an_archived_pot_name_is_refused_at_selection(service) -> None:
    _legacy_archived(service, "dead")
    client, services = _engine_client(service, "dead")

    reset = await client.reset_context(
        ResetContextRequest(), confirmation=DestructiveConfirmation(confirmed=True)
    )

    assert isinstance(reset, Failure)
    assert reset.error.code == "pot_archived"
    services.backend.mutation.reset_pot.assert_not_called()


def test_an_engine_refusal_of_an_archived_pot_exits_like_every_other(
    service, monkeypatch
) -> None:
    """The engine's authorizer raises it as an authorization error; the CLI
    still reports `pot_archived` with exit 1, not as a credential failure."""
    dead = _legacy_archived(service, "dead")
    client, _ = _engine_client(service, dead.pot_id)
    _common.set_json(True)

    with pytest.raises(_common.CliExpectedFailureExit) as exc_info:
        with _common.contract():
            _common.run_engine_operation(client.search(SearchRequest(query="x")))

    assert exc_info.value.exit_code == 1
    assert exc_info.value.error_code == "pot_archived"


# --- one name, one pot -------------------------------------------------------


def test_rename_refuses_a_name_another_live_pot_already_uses(service) -> None:
    service.create_pot(name="alpha", use=True)
    service.create_pot(name="beta")

    result = _run(service, "rename", "beta", "alpha")

    assert result.exit_code == 1, result.output
    payload = _payload(result)
    assert payload["code"] == "pot_name_conflict"
    assert payload["recommended_next_action"]
    assert [p.name for p in service.list_pots()] == ["alpha", "beta"]


def test_rename_may_reuse_a_name_freed_by_archiving(service) -> None:
    service.create_pot(name="alpha", use=True)
    keeper = service.create_pot(name="beta")
    _archive(service, "alpha")

    renamed = service.rename_pot(ref=keeper.pot_id, new_name="alpha")

    assert renamed.name == "alpha"


def test_rename_to_its_own_name_is_not_a_conflict(service) -> None:
    pot = service.create_pot(name="alpha", use=True)

    assert service.rename_pot(ref="alpha", new_name="alpha").pot_id == pot.pot_id


def test_rename_refuses_a_name_that_shadows_a_pot_id(service) -> None:
    """Refs resolve against ids and names, and ids win the lookup, so a pot
    named after another pot's id left that pot unreachable by name."""
    shadowed = service.create_pot(name="alpha", use=True)
    service.create_pot(name="beta")

    result = _run(service, "rename", "beta", shadowed.pot_id)

    assert result.exit_code == 1, result.output
    assert _payload(result)["code"] == "pot_name_conflict"


def test_create_refuses_a_name_that_shadows_a_pot_id(service) -> None:
    shadowed = service.create_pot(name="alpha", use=True)

    result = _run(service, "create", shadowed.pot_id)

    assert result.exit_code == 1, result.output
    assert _payload(result)["code"] == "pot_name_conflict"
    assert len(service.list_pots()) == 1


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_create_refuses_a_blank_name(service, name) -> None:
    """``pot use ''`` is not a command anyone can type."""
    result = _run(service, "create", name)

    assert result.exit_code == 1, result.output
    assert _payload(result)["code"] == "validation_error"
    assert service.list_pots() == []


@pytest.mark.parametrize("name", ["", "   "])
def test_rename_refuses_a_blank_name_and_leaves_the_pot_named(service, name) -> None:
    service.create_pot(name="alpha", use=True)

    result = _run(service, "rename", "alpha", name)

    assert result.exit_code == 1, result.output
    assert _payload(result)["code"] == "validation_error"
    assert [p.name for p in service.list_pots()] == ["alpha"]


# --- the explorer UI applies the same rules ---------------------------------

_TOKEN = "archive-test-daemon-token"  # noqa: S105 - non-secret test fixture


def _ui(service: LocalPotManagementService) -> TestClient:
    graph = SimpleNamespace(
        data_plane_status=lambda pot_id: SimpleNamespace(counts={"claims": 0})
    )
    app = build_ui_app(pots=service, graph=graph, backend=object(), bearer_token=_TOKEN)
    return TestClient(
        app,
        base_url="http://127.0.0.1:8765",
        headers={"Authorization": f"Bearer {_TOKEN}"},
    )


def test_the_explorer_does_not_offer_or_select_archived_pots(service) -> None:
    service.create_pot(name="live", use=True)
    service.create_pot(name="dead")
    _archive(service, "dead")
    client = _ui(service)

    listed = client.get("/ui/api/pots").json()
    selected = client.post("/ui/api/pots/use", json={"ref": "dead"})
    status = client.get("/ui/api/status", params={"pot": "dead"})

    assert [row["name"] for row in listed["pots"]] == ["live"]
    assert selected.status_code == 409
    assert "archived" in selected.json()["detail"]
    assert status.status_code == 409
    assert service.active_pot().name == "live"
