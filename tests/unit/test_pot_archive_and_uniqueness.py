"""``archived`` has to mean something, and a pot name has to mean one pot.

Two defects with one shape: a field the product wrote and never read.

*Archive* was documented as "pot deletion from the control plane" and enforced
nowhere. Archived pots kept appearing in ``pot list`` with no marker and no
``archived`` key in the JSON; ``pot use`` still selected them; claims, renames
and source registrations still wrote to them; and a repo default pointing at one
kept routing every repo-scoped read and write into a pot whose graph and
resource tree had been torn down at archive time — so the answers came back
*empty* rather than wrong, which is far harder to notice.

*Uniqueness* was enforced nowhere either. ``pot rename`` would happily put two
pots under one name, after which every bare ref picked an arbitrary one of them
— including the bare ref handed to ``pot reset <name> --confirm``.

Two more of the same family live here now. *Nameability*: a pot name is the ref
humans type, and both ``create`` and ``rename`` accepted the empty string and
whitespace — a pot that can never be named again. *Targeting*: ``pot reset``
with no ref inferred its victim from the repo binding, so it wiped a pot the
operator had not selected, and found one to wipe even with no active pot at all.

These run the real ``LocalPotManagementService`` over a real ``LocalPotStore``,
because both defects live in the store's ref resolution and the service's
guards. A fake pot service would agree with whatever the CLI asked it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common, pots
from potpie_context_engine.adapters.outbound.pots.local_pot_store import LocalPotStore
from potpie_context_engine.application.services.pot_management import (
    LocalPotManagementService,
)


class _Mutation:
    """The real graph wipe, recorded. ``calls`` is what the targeting tests turn
    on: a refusal that still reached the mutation adapter is not a refusal."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def reset_pot(self, pot_id: str) -> dict[str, object]:
        self.calls.append(pot_id)
        return {"ok": True}


@dataclass(slots=True)
class _Backend:
    mutation: _Mutation


class _Host:
    def __init__(self, pots_service: LocalPotManagementService) -> None:
        self.pots = pots_service


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_host(None)
    _common.set_json(False)


@pytest.fixture()
def service(tmp_path) -> LocalPotManagementService:
    return LocalPotManagementService(
        store=LocalPotStore(home=tmp_path / "home"),
        backend=_Backend(_Mutation()),
        resources=None,
    )


def _run(service: LocalPotManagementService, *args: str):
    _common.set_host(_Host(service))
    _common.set_json(True)
    return CliRunner().invoke(pots.pot_app, list(args))


def _payload(result) -> dict:
    return json.loads(result.output)


# --- archive is a lifecycle state -------------------------------------------


def test_pot_list_hides_archived_pots_and_says_so(service) -> None:
    service.create_pot(name="live", use=True)
    service.create_pot(name="dead")
    service.archive_pot(ref="dead")

    result = _run(service, "list")

    assert result.exit_code == 0, result.output
    assert [row["name"] for row in _payload(result)["pots"]] == ["live"]

    with_archived = _run(service, "list", "--archived")
    rows = {row["name"]: row for row in _payload(with_archived)["pots"]}
    assert rows["dead"]["archived"] is True
    # The key has to be *present* on live pots too, or a consumer cannot tell
    # "not archived" from "this host does not report it".
    assert rows["live"]["archived"] is False


@pytest.mark.parametrize(
    "args",
    [
        ("use", "dead"),
        ("rename", "dead", "reborn"),
        ("reset", "dead", "--confirm"),
        ("archive", "dead", "--confirm"),
        ("default", "set", "dead"),
    ],
)
def test_archived_pots_refuse_every_command_that_targets_them(service, args) -> None:
    service.create_pot(name="live", use=True)
    service.create_pot(name="dead")
    service.archive_pot(ref="dead")

    result = _run(service, *args)

    assert result.exit_code != 0, result.output
    payload = _payload(result)
    # Specifically *not* `pot_not_found`: the ref resolved. Answering "no pot
    # matching 'dead'" sends the operator to a listing that deliberately hides
    # it, to look for something that is not missing.
    assert payload["code"] == "pot_archived", payload
    assert "archived" in payload["message"]
    assert "--archived" in (payload["recommended_next_action"] or "")


def test_an_archived_pot_stops_being_the_repo_default(service, tmp_path) -> None:
    """The stale pointer is the quiet one: reads come back empty, not wrong."""
    pot = service.create_pot(name="dead", use=True)
    service.set_repo_default(repo="git@example.com:acme/app.git", pot_id=pot.pot_id)
    assert service.repo_default(repo="git@example.com:acme/app.git") == pot.pot_id

    service.archive_pot(ref="dead")

    assert service.repo_default(repo="git@example.com:acme/app.git") is None


def test_an_archived_pot_is_not_a_repo_source_candidate(service) -> None:
    pot = service.create_pot(name="dead", use=True)
    service.add_source(pot_id=pot.pot_id, kind="repo", location="/src/app")
    assert [row.pot_id for row in service.list_repo_sources()] == [pot.pot_id]

    service.archive_pot(ref="dead")

    assert service.list_repo_sources() == []


def test_an_archived_pot_refuses_new_source_registrations(service) -> None:
    pot = service.create_pot(name="dead", use=True)
    service.archive_pot(ref="dead")

    with pytest.raises(Exception) as exc:
        service.add_source(pot_id=pot.pot_id, kind="repo", location="/src/app")

    assert "archived" in str(exc.value)


def test_creating_a_pot_named_after_an_archived_one_starts_fresh(service) -> None:
    """Reuse-by-name must not resurrect a pot whose data was destroyed.

    ``create`` is idempotent so ``setup`` can re-run, but handing back an
    archived pot would return an emptied graph under a name the caller believes
    is new — and quietly un-hide it by making it active.
    """
    old = service.create_pot(name="app", use=True)
    service.archive_pot(ref="app")

    new = service.create_pot(name="app", use=True)

    assert new.pot_id != old.pot_id
    assert new.archived is False
    assert new.created is True


def test_create_says_when_it_reused_an_existing_pot(service) -> None:
    first = service.create_pot(name="app")
    again = service.create_pot(name="app")

    assert first.created is True
    assert again.created is False
    assert again.pot_id == first.pot_id


# --- one name, one pot -------------------------------------------------------


def test_rename_refuses_a_name_another_live_pot_already_uses(service) -> None:
    service.create_pot(name="alpha", use=True)
    service.create_pot(name="beta")

    result = _run(service, "rename", "beta", "alpha")

    assert result.exit_code != 0, result.output
    payload = _payload(result)
    assert payload["code"] == "pot_name_conflict"
    assert [p.name for p in service.list_pots()] == ["alpha", "beta"]


def test_rename_may_reuse_a_name_freed_by_archiving(service) -> None:
    service.create_pot(name="alpha", use=True)
    keeper = service.create_pot(name="beta")
    service.archive_pot(ref="alpha")

    renamed = service.rename_pot(ref=keeper.pot_id, new_name="alpha")

    assert renamed.name == "alpha"


def test_rename_refuses_a_name_that_shadows_a_pot_id(service) -> None:
    """Refs resolve against ids *and* names, and ids win the lookup.

    A pot named after another pot's id therefore made the other one permanently
    unreachable by name.
    """
    shadowed = service.create_pot(name="alpha", use=True)
    service.create_pot(name="beta")

    result = _run(service, "rename", "beta", shadowed.pot_id)

    assert result.exit_code != 0, result.output
    assert _payload(result)["code"] == "pot_name_conflict"


def test_create_refuses_a_name_that_shadows_a_pot_id(service) -> None:
    shadowed = service.create_pot(name="alpha", use=True)

    with pytest.raises(Exception) as exc:
        service.create_pot(name=shadowed.pot_id)

    assert "id" in str(exc.value)


# --- a name has to be a name -------------------------------------------------


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_create_refuses_a_blank_name(service, name) -> None:
    """A pot whose name is blank cannot be named again: ``pot use ''`` is not a
    command anyone can type, and the listing prints it as ``* ()``."""
    result = _run(service, "create", name)

    assert result.exit_code != 0, result.output
    assert _payload(result)["code"] == "validation_error"
    assert service.list_pots() == []


@pytest.mark.parametrize("name", ["", "   "])
def test_rename_refuses_a_blank_name_and_leaves_the_pot_named(service, name) -> None:
    """The worse half of the same hole: ``create`` throws away a pot nobody had
    used yet, while ``rename`` does it to one that already holds a project's
    memory."""
    service.create_pot(name="alpha", use=True)

    result = _run(service, "rename", "alpha", name)

    assert result.exit_code != 0, result.output
    assert _payload(result)["code"] == "validation_error"
    assert [p.name for p in service.list_pots()] == ["alpha"]


# --- destructive commands name what they are about to destroy ---------------


def test_the_reset_prompt_names_the_host_it_would_run_against(service) -> None:
    """A prompt saying only ``resetting 'default'`` is not enough to consent to
    when ``default`` exists on two hosts."""
    pot = service.create_pot(name="alpha", use=True)

    result = _run(service, "reset", "alpha")

    assert result.exit_code != 0, result.output
    payload = _payload(result)
    assert payload["code"] == "confirmation_required"
    assert "on the local host" in payload["message"]
    assert pot.pot_id in payload["message"]


def test_reset_accepts_a_host_qualified_ref(service) -> None:
    """``pot list`` prints ``<host>:<name>`` as *the* way to target a host, and
    reset/archive/rename were the only commands that answered it with
    ``pot_not_found``."""
    service.create_pot(name="alpha", use=True)

    result = _run(service, "reset", "local:alpha", "--confirm")

    assert result.exit_code == 0, result.output
    assert _payload(result)["reset"] is True


def test_the_reset_prompt_names_the_pot_not_just_its_id(service) -> None:
    """An id alone is not something the operator can check against the pot they
    believe they are in — least of all on the path that resolved it for them."""
    pot = service.create_pot(name="alpha", use=True)

    result = _run(service, "reset", pot.pot_id)

    assert result.exit_code != 0, result.output
    assert "'alpha'" in _payload(result)["message"]


# --- a bare reset may not pick its own victim --------------------------------

REPO = "git@example.com:acme/shop.git"


@pytest.fixture()
def in_repo(monkeypatch):
    """Run as if cwd were ``REPO``, whatever the developer's checkout is."""
    monkeypatch.setattr(_common, "_current_repo_identity", lambda: REPO)
    return REPO


def test_a_bare_reset_refuses_when_the_repo_pot_is_not_the_active_one(
    service, in_repo
) -> None:
    """The reproduction: ``pot reset --confirm`` with no ref resolved through the
    repo default and wiped a pot the operator had not selected, while
    ``pot info`` was naming a different one as active."""
    active = service.create_pot(name="alpha", use=True)
    bound = service.create_pot(name="beta")
    service.set_repo_default(repo=in_repo, pot_id=bound.pot_id)

    result = _run(service, "reset", "--confirm")

    assert result.exit_code != 0, result.output
    payload = _payload(result)
    assert payload["code"] == "ambiguous_pot"
    # Both pots are named, because the whole failure is not knowing which one
    # the command meant.
    assert "beta" in payload["message"] and "alpha" in payload["message"]
    assert payload["detail"] == {
        "target_pot_id": bound.pot_id,
        "resolved_via": "repo_default",
        "active_pot_id": active.pot_id,
    }
    assert service.backend.mutation.calls == []


def test_a_bare_reset_with_no_active_pot_refuses_instead_of_finding_a_victim(
    service, in_repo
) -> None:
    """Archiving the pot you were in leaves no active pot and a repo default that
    still resolves. A bare ``reset`` took that as a target."""
    service.create_pot(name="alpha", use=True)
    bound = service.create_pot(name="beta")
    service.set_repo_default(repo=in_repo, pot_id=bound.pot_id)
    service.archive_pot(ref="alpha")
    assert service.active_pot() is None
    service.backend.mutation.calls.clear()  # the archive's own wipe

    result = _run(service, "reset", "--confirm")

    assert result.exit_code != 0, result.output
    payload = _payload(result)
    assert payload["code"] == "no_active_pot"
    assert "beta" in payload["message"]
    assert payload["recommended_next_action"].count(bound.pot_id) == 2
    assert service.backend.mutation.calls == []


def test_a_bare_reset_still_works_when_the_repo_pot_is_the_active_one(
    service, in_repo
) -> None:
    """The refusal must not cost the ordinary case its ref: ``setup`` binds the
    repo default to the pot it activates, so the two normally agree."""
    pot = service.create_pot(name="alpha", use=True)
    service.set_repo_default(repo=in_repo, pot_id=pot.pot_id)

    result = _run(service, "reset", "--confirm")

    assert result.exit_code == 0, result.output
    assert _payload(result)["reset"] is True
    assert service.backend.mutation.calls == [pot.pot_id]
