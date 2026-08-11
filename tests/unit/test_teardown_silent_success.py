"""Commands that reported success for work they never did.

All the same shape and the same cause — a lookup miss on an id the caller
typed, handled by falling through to the success message:

``source remove <id>``
    filtered a list that did not contain the id and printed "removed source
    <id>" at exit 0. The likeliest reason for the miss is that the source is
    alive in the pot the caller did not pass, and the message sends them away
    believing it is gone.
``skills install <id>`` / ``skills update <id>`` / ``skills remove <id>``
    skipped an id the catalog does not carry and reported the operation as
    done — ``update`` indistinguishably from "already up to date". The next
    session then runs without the context the harness was told it had.

The dividing line these tests hold: an id the *caller named* must be answered
about; an id that is merely in a set the product chose to iterate — the
recommended bundle, everything currently installed — must be walked past, or a
packaging gap takes the whole bundle down with it.

Both are exercised through the layer that decides — the CLI command and the
skill manager — with the install target faked, because installing for real
writes into the developer's own harness directories.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from potpie.cli.commands import _common, pots, skills
from potpie_context_engine.adapters.outbound.pots.local_pot_store import LocalPotStore
from potpie_context_engine.adapters.outbound.skills.bundle_catalog import catalog_by_id
from potpie_context_engine.application.services.pot_management import (
    LocalPotManagementService,
)
from potpie_context_engine.application.services.skill_manager import DefaultSkillManager


class _Backend:
    """``source remove`` never reaches the graph; nothing here should be called."""

    mutation = None


class _OlderHostPots:
    """A host on the previous contract: ``remove_source`` answers nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def list_pots(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(pot_id="p1", name="shop", active=True)]

    def remove_source(self, *, pot_id: str, source_id: str) -> None:
        self.calls.append((pot_id, source_id))


class _Host:
    def __init__(self, pots_service: Any) -> None:
        self.pots = pots_service


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    _common.set_host(None)
    _common.set_json(False)


@pytest.fixture()
def service(tmp_path, monkeypatch) -> LocalPotManagementService:
    # The host registry reads its origin state from the home dir; keep it in
    # tmp so the test never consults the developer's real ~/.potpie.
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "home"))
    return LocalPotManagementService(
        store=LocalPotStore(home=tmp_path / "home"),
        backend=_Backend(),
    )


def _run(service: LocalPotManagementService, *args: str):
    _common.set_host(_Host(service))
    _common.set_json(True)
    return CliRunner().invoke(pots.source_app, list(args))


# --- source remove ----------------------------------------------------------


def test_removing_a_source_id_the_pot_never_held_fails(service) -> None:
    pot = service.create_pot(name="shop", use=True)

    result = _run(service, "remove", "src_deadbeef", "--pot", pot.pot_id)

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "pot_not_found"
    assert "src_deadbeef" in payload["message"]
    assert pot.pot_id in payload["message"]
    assert "removed" not in payload


def test_removing_a_real_source_id_under_the_wrong_pot_fails(service) -> None:
    """The registration is still there — in the other pot. Saying it was
    removed is how a caller loses track of which pot owns what."""
    owner = service.create_pot(name="shop", use=True)
    other = service.create_pot(name="warehouse")
    source = service.add_source(
        pot_id=owner.pot_id, kind="repo", location="github.com/acme/shop"
    )

    result = _run(service, "remove", source.source_id, "--pot", other.pot_id)

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "pot_not_found"
    assert other.pot_id in payload["message"]
    assert [s.source_id for s in service.list_sources(pot_id=owner.pot_id)] == [
        source.source_id
    ]


def test_removing_a_registered_source_still_succeeds(service) -> None:
    pot = service.create_pot(name="shop", use=True)
    source = service.add_source(
        pot_id=pot.pot_id, kind="repo", location="github.com/acme/shop"
    )

    result = _run(service, "remove", source.source_id, "--pot", pot.pot_id)

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {"removed": source.source_id, "resources_touched": False}
    assert service.list_sources(pot_id=pot.pot_id) == []


def test_a_host_that_answers_nothing_is_still_credited_with_the_removal(
    tmp_path, monkeypatch
) -> None:
    """Only an explicit ``False`` is a miss. A daemon on the previous contract
    returns ``None`` from ``remove_source`` and did remove the row; failing on
    its silence would turn a version skew into a command that never works."""
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path / "home"))
    older = _OlderHostPots()

    _common.set_host(_Host(older))
    _common.set_json(True)
    result = CliRunner().invoke(pots.source_app, ["remove", "src_abc", "--pot", "p1"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["removed"] == "src_abc"
    assert older.calls == [("p1", "src_abc")]


# --- skills install ---------------------------------------------------------


class _FakeTarget:
    """Records installs/removes instead of writing into a real harness directory."""

    skills_root = "/nowhere/skills"

    def __init__(self, installed: dict[str, str] | None = None) -> None:
        self._installed = dict(installed or {})
        self.installs: list[tuple[str, str]] = []
        self.removes: list[str] = []

    def installed(self) -> dict[str, str]:
        return dict(self._installed)

    def install(self, *, skill_id: str, version: str, path: str | None = None) -> None:
        self.installs.append((skill_id, version))

    def remove(self, *, skill_id: str) -> None:
        self.removes.append(skill_id)


def test_installing_an_unknown_skill_id_is_an_error_not_a_skip() -> None:
    target = _FakeTarget()
    manager = DefaultSkillManager(targets={"claude": target})

    with pytest.raises(ValueError) as exc:
        manager.install(agent="claude", skill_id="potpie-does-not-exist")

    message = str(exc.value)
    assert "potpie-does-not-exist" in message
    # Naming the valid ids is the difference between a typo the caller can fix
    # and a dead end.
    assert sorted(catalog_by_id())[0] in message
    assert target.installs == []


def test_the_unknown_skill_id_reaches_the_cli_as_a_validation_error(capsys) -> None:
    manager = DefaultSkillManager(targets={"claude": _FakeTarget()})
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exit_info:
        with _common.contract():
            manager.install(agent="claude", skill_id="potpie-does-not-exist")

    assert exit_info.value.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "validation_error"
    assert "potpie-does-not-exist" in payload["message"]


def test_a_recommended_id_missing_from_the_catalog_is_still_skipped(
    monkeypatch,
) -> None:
    """The bundle install must survive a packaging gap: an id nobody typed is
    not a caller error, and failing the whole install would strand every other
    skill in the bundle."""
    from potpie_context_engine.application.services import skill_manager as module

    packaged = sorted(catalog_by_id())[0]
    monkeypatch.setattr(module, "RECOMMENDED_SKILL_IDS", (packaged, "potpie-unbundled"))
    target = _FakeTarget()

    result = DefaultSkillManager(targets={"claude": target}).install(agent="claude")

    assert result.changed == (packaged,)
    assert [sid for sid, _ in target.installs] == [packaged]


# --- skills update ----------------------------------------------------------

#: An id no packaged bundle carries — a typo, or a skill a later build dropped.
RETIRED = "potpie-retired-skill"


def test_updating_an_unknown_skill_id_is_an_error_not_a_silent_no_op() -> None:
    """The same miss one function below ``install``: ``skills update <typo>``
    exited 0 with ``changed: []``, which is exactly what "already up to date"
    looks like, for a skill the caller never had."""
    target = _FakeTarget()
    manager = DefaultSkillManager(targets={"claude": target})

    with pytest.raises(ValueError) as exc:
        manager.update(agent="claude", skill_id="potpie-does-not-exist")

    message = str(exc.value)
    assert "potpie-does-not-exist" in message
    assert sorted(catalog_by_id())[0] in message
    assert target.installs == []


def test_the_unknown_update_id_reaches_the_cli_as_a_validation_error(capsys) -> None:
    """Exit 0 was the whole defect; the contract has to see a ValueError."""
    manager = DefaultSkillManager(targets={"claude": _FakeTarget()})
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exit_info:
        with _common.contract():
            manager.update(agent="claude", skill_id="potpie-does-not-exist")

    assert exit_info.value.exit_code == _common.EXIT_VALIDATION
    assert json.loads(capsys.readouterr().out)["code"] == "validation_error"


def test_updating_an_installed_id_the_bundle_dropped_names_the_way_out() -> None:
    """Calling it an unknown skill would be false — it is installed, in front
    of the user. There is simply no version to install over it, so the honest
    answer is the command that does work."""
    target = _FakeTarget(installed={RETIRED: "1"})

    with pytest.raises(ValueError) as exc:
        DefaultSkillManager(targets={"claude": target}).update(
            agent="claude", skill_id=RETIRED
        )

    message = str(exc.value)
    assert RETIRED in message
    assert "potpie skills remove" in message
    assert target.installs == []


def test_sweeping_updates_walks_past_an_installed_id_the_bundle_dropped() -> None:
    """Nobody typed the retired id — it is in the sweep only because it is on
    disk. Failing the sweep would strand every other skill's update."""
    packaged = sorted(catalog_by_id())[0]
    target = _FakeTarget(installed={packaged: "0.0-stale", RETIRED: "1"})

    result = DefaultSkillManager(targets={"claude": target}).update(
        agent="claude", all_=True
    )

    assert result.changed == (packaged,)
    assert [sid for sid, _ in target.installs] == [packaged]


# --- skills update, through the command line --------------------------------
#
# The refusals above are the manager's, and for a while nothing could reach
# them: `skills update` took `--all/--agent/--path/--scope` and no id, so
# `potpie skills update <typo>` was answered by click with "Got unexpected extra
# argument" — a statement about the command line, not about the skill — while
# the branch that had something to say sat one call below, pinned by tests that
# drove the manager directly. A behaviour reachable only from a test is not a
# behaviour the product has.


def _skills_cli(monkeypatch, target: _FakeTarget) -> DefaultSkillManager:
    """Point the command group at a fake harness.

    ``skills._skills()`` builds the local host and hands back the real manager,
    which writes into the developer's own ``~/.claude``. The seam is replaced
    rather than the host, so no test here can install a skill for real.

    ``install`` also reports to product analytics. That call already no-ops
    without a telemetry context, which no test sets — but "no context happens to
    be configured" is a property of the environment, and a test that would post
    to the network the moment one is must not be left resting on it.
    """
    manager = DefaultSkillManager(targets={"claude": target})
    monkeypatch.setattr(skills, "_skills", lambda: manager)
    monkeypatch.setattr(skills, "capture_project_binding_event", lambda *a, **k: None)
    return manager


def _run_skills(*args: str):
    _common.set_json(True)
    return CliRunner().invoke(skills.skills_app, list(args))


def test_updating_an_unknown_skill_id_is_reachable_from_the_cli(monkeypatch) -> None:
    target = _FakeTarget()
    _skills_cli(monkeypatch, target)

    result = _run_skills("update", "potpie-does-not-exist")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "potpie-does-not-exist" in payload["message"]
    assert target.installs == []


def test_updating_one_named_skill_updates_that_one(monkeypatch) -> None:
    """The other half of the argument existing: a real id has to *arrive* at the
    manager. An argument the command accepts and drops would report a sweep of
    everything installed as the answer to a question about one skill."""
    packaged, other = sorted(catalog_by_id())[:2]
    target = _FakeTarget(installed={packaged: "0.0-stale", other: "0.0-stale"})
    _skills_cli(monkeypatch, target)

    result = _run_skills("update", packaged)

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["changed"] == [packaged]
    assert [sid for sid, _ in target.installs] == [packaged]


def test_naming_a_skill_and_all_at_once_is_refused(monkeypatch) -> None:
    """``--all`` used to win outright, which meant the id the caller typed was
    discarded and the sweep it did instead was reported as success — the same
    silent-substitution this whole file is about, one flag over."""
    packaged = sorted(catalog_by_id())[0]
    target = _FakeTarget(installed={packaged: "0.0-stale"})
    _skills_cli(monkeypatch, target)

    result = _run_skills("update", packaged, "--all")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    assert "not both" in json.loads(result.output)["message"]
    assert target.installs == []


def test_the_bare_sweep_still_updates_everything_installed(monkeypatch) -> None:
    """The control: naming an id is new, needing one is not."""
    packaged = sorted(catalog_by_id())[0]
    target = _FakeTarget(installed={packaged: "0.0-stale"})
    _skills_cli(monkeypatch, target)

    result = _run_skills("update")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["changed"] == [packaged]


@pytest.mark.parametrize("command", ["install", "update"])
def test_both_id_taking_commands_refuse_a_typo_at_the_command_line(
    monkeypatch, command: str
) -> None:
    """``install`` and ``update`` take an id or the set the product chooses.

    Pinned together because the pair is the invariant: ``update`` was the one
    missing the argument, and the way that is reintroduced is by the two
    signatures drifting again. Asserting the same refusal through both doors
    means a command that quietly stops accepting an id fails here rather than
    at a user's terminal, where it does not look like an error at all — click
    answers a dropped argument with "Got unexpected extra argument", which
    reads as a mistyped command rather than a missing feature.
    """
    target = _FakeTarget()
    _skills_cli(monkeypatch, target)

    result = _run_skills(command, "potpie-does-not-exist")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    # The refusal has to be about the skill, not about the command line.
    assert "potpie-does-not-exist" in payload["message"]
    assert "extra argument" not in payload["message"]
    assert target.installs == []


# --- skills remove ----------------------------------------------------------


def test_removing_an_unknown_skill_id_is_an_error() -> None:
    """Neither the catalog nor the harness has heard of it, so it is a typo,
    and ``removed: []`` for a typo is how a caller comes to believe a skill is
    gone from a harness it was never in."""
    target = _FakeTarget()

    with pytest.raises(ValueError) as exc:
        DefaultSkillManager(targets={"claude": target}).remove(
            agent="claude", skill_id="potpie-does-not-exist"
        )

    assert "potpie-does-not-exist" in str(exc.value)
    assert target.removes == []


def test_removing_a_catalogued_skill_that_is_not_installed_is_a_no_op() -> None:
    """Unlike a typo, this id is real and the end state the caller asked for
    already holds — ``removed: []`` states that exactly. Refusing here would
    make ``skills remove`` fail on its own second run."""
    packaged = sorted(catalog_by_id())[0]
    target = _FakeTarget()

    result = DefaultSkillManager(targets={"claude": target}).remove(
        agent="claude", skill_id=packaged
    )

    assert result.changed == ()
    assert target.removes == []


def test_removing_an_installed_id_the_bundle_dropped_still_removes_it() -> None:
    """The refusal is aimed at typos, not at skills a newer build stopped
    shipping: those are precisely the ones a caller has to clear by hand."""
    target = _FakeTarget(installed={RETIRED: "1"})

    result = DefaultSkillManager(targets={"claude": target}).remove(
        agent="claude", skill_id=RETIRED
    )

    assert result.changed == (RETIRED,)
    assert target.removes == [RETIRED]


def test_removing_everything_installed_stays_lenient() -> None:
    packaged = sorted(catalog_by_id())[0]
    target = _FakeTarget(installed={packaged: "1", RETIRED: "1"})

    result = DefaultSkillManager(targets={"claude": target}).remove(
        agent="claude", all_=True
    )

    assert sorted(result.changed) == sorted((packaged, RETIRED))
    assert sorted(target.removes) == sorted([packaged, RETIRED])
