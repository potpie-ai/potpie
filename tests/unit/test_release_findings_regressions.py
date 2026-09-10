"""Regressions for the potpie 2.0.1 release-test findings.

Each test names the finding it locks down. Grouped by the thing that was
wrong rather than by module, so the report and the suite stay legible
against each other.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
import pytest
from typer.testing import CliRunner

from potpie.cli import main as cli_main
from potpie.cli import repo_location
from potpie.cli.commands import _common, pots, query
from potpie.daemon import discovery as daemon_discovery
from potpie.daemon.lifecycle import Daemon
from potpie.pots.contracts import PotAggregateStatus, PotInfo

pytestmark = pytest.mark.unit

runner = CliRunner()


# --- B1: a stale daemon across an upgrade must not wedge the CLI ------------


def _write_private(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def test_legacy_discovery_record_is_named_as_such_not_merely_invalid(
    tmp_path: Path,
) -> None:
    """A pre-upgrade record must be diagnosed, and marked recoverable."""
    _write_private(
        daemon_discovery.discovery_path(tmp_path),
        json.dumps(
            {
                "transport": "http",
                "base_url": "http://127.0.0.1:60096",
                "token": "x" * 40,
                "pid": 30984,
            }
        ),
    )

    with pytest.raises(daemon_discovery.DaemonDiscoveryError) as raised:
        daemon_discovery.read_daemon_discovery(tmp_path)

    error = raised.value
    assert error.code == daemon_discovery.DISCOVERY_SCHEMA_UNSUPPORTED
    assert "older potpie" in str(error)
    assert error.recoverable_by_replacement is True
    assert "daemon restart" in (error.recommended_next_action or "")


def test_absent_discovery_record_points_at_start_not_restart(tmp_path: Path) -> None:
    with pytest.raises(daemon_discovery.DaemonDiscoveryError) as raised:
        daemon_discovery.load_daemon_connection(tmp_path)

    assert raised.value.code == daemon_discovery.DISCOVERY_ABSENT
    assert "daemon start" in (raised.value.recommended_next_action or "")
    assert raised.value.recoverable_by_replacement is False


def test_status_of_an_unauthenticatable_daemon_names_the_escape_hatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`daemon status` used to report a bare up=True, contradicting `status`."""
    _write_private(daemon_discovery.pid_path(tmp_path), "4242\n")
    _write_private(
        daemon_discovery.discovery_path(tmp_path),
        json.dumps({"base_url": "http://127.0.0.1:1", "token": "x" * 40}),
    )
    monkeypatch.setattr("potpie.daemon.lifecycle._pid_alive", lambda _pid: True)

    status = Daemon(home=tmp_path, in_process=False).status()

    assert status["up"] is True
    assert status["identity"] == "unauthenticated"
    assert status["identity_reason"] == daemon_discovery.DISCOVERY_SCHEMA_UNSUPPORTED
    assert "kill 4242" in status["recovery"]
    assert "daemon restart" in status["recovery"]


def test_stop_refusal_offers_force_and_the_manual_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_private(daemon_discovery.pid_path(tmp_path), "4242\n")
    _write_private(
        daemon_discovery.discovery_path(tmp_path),
        json.dumps({"base_url": "http://127.0.0.1:1", "token": "x" * 40}),
    )
    monkeypatch.setattr("potpie.daemon.lifecycle._pid_alive", lambda _pid: True)

    from potpie.daemon.lifecycle import DaemonStopError

    with pytest.raises(DaemonStopError) as raised:
        Daemon(home=tmp_path, in_process=False).stop()

    next_action = raised.value.error.recommended_next_action or ""
    assert "daemon stop --force" in next_action
    assert "kill 4242" in next_action


# --- B2 / M1: one pot resolution, reported honestly -------------------------


class _Pot:
    def __init__(self, pot_id: str, name: str, active: bool = False) -> None:
        self.pot_id = pot_id
        self.name = name
        self.active = active
        self.archived = False


class _Pots:
    """Active pot is `alpha`; this repo's default binding is `beta`."""

    supports_repo_defaults = True

    def __init__(self) -> None:
        self.alpha = _Pot("pot-alpha", "alpha", active=True)
        self.beta = _Pot("pot-beta", "beta")
        self.reset_calls: list[str] = []

    def list_pots(self):
        return [self.alpha, self.beta]

    def active_pot(self):
        return self.alpha

    def repo_default(self, *, repo: str):
        return "pot-beta"

    def list_sources(self, *, pot_id: str):
        return []

    def aggregate_status(self, *, pot_id: str | None = None):
        target = next(
            (pot for pot in self.list_pots() if pot.pot_id == pot_id), self.alpha
        )
        return PotAggregateStatus(
            active_pot=PotInfo(pot_id="pot-alpha", name="alpha", active=True),
            pot_count=2,
            target_pot=PotInfo(pot_id=target.pot_id, name=target.name),
        )


class _Host:
    def __init__(self, pots_service: _Pots) -> None:
        self.pots = pots_service
        self.profile = "local"


@pytest.fixture
def repo_default_host(monkeypatch: pytest.MonkeyPatch) -> _Pots:
    pots_service = _Pots()
    _common.set_runtime(_Host(pots_service))
    monkeypatch.setattr(
        _common, "_current_repo_identity", lambda: "github.com/acme/shop"
    )
    return pots_service


def test_pot_reset_names_the_target_and_the_diverging_active_pot(
    repo_default_host: _Pots,
) -> None:
    """The most destructive pot op must not target an unreported pot."""
    _common.set_json(True)

    result = runner.invoke(pots.pot_app, ["reset"])

    emitted = json.loads(result.output)
    assert emitted["code"] == "destructive_confirmation_required"
    # The refusal names the repo-default target *and* says the active pot differs.
    assert "pot-beta" in emitted["message"]
    assert "alpha (pot-alpha)" in emitted["message"]
    assert "potpie pot reset pot-beta --confirm" in emitted["recommended_next_action"]


def test_pot_reset_accepts_pot_flag(repo_default_host: _Pots) -> None:
    _common.set_json(True)

    result = runner.invoke(pots.pot_app, ["reset", "--pot", "alpha"])

    emitted = json.loads(result.output)
    assert "potpie pot reset pot-alpha --confirm" in emitted["recommended_next_action"]


def test_pot_reset_rejects_two_conflicting_pot_selections(
    repo_default_host: _Pots,
) -> None:
    _common.set_json(True)

    result = runner.invoke(pots.pot_app, ["reset", "beta", "--pot", "alpha"])

    emitted = json.loads(result.output)
    assert emitted["code"] == "validation_error"
    assert "conflicting pot selection" in emitted["message"]


def test_pot_list_marks_the_pot_commands_here_actually_use(
    repo_default_host: _Pots,
) -> None:
    """`*` alone hid the active-vs-effective split behind every other command."""
    _common.set_json(True)

    result = runner.invoke(pots.pot_app, ["list"])

    emitted = json.loads(result.output)
    rows = {row["name"]: row for row in emitted["pots"]}
    assert rows["alpha"]["active"] is True
    assert rows["alpha"]["effective_for_current_repo"] is False
    assert rows["beta"]["active"] is False
    assert rows["beta"]["effective_for_current_repo"] is True


def test_pot_info_accepts_pot_flag(repo_default_host: _Pots) -> None:
    _common.set_json(True)

    result = runner.invoke(pots.pot_app, ["info", "--pot", "alpha"])

    emitted = json.loads(result.output)
    assert emitted["pot"]["id"] == "pot-alpha"
    assert emitted["pot"]["resolved_via"] == "explicit"
    assert emitted["active_pot"]["id"] == "pot-alpha"


# --- M4: an evidence item must never render as an empty bullet --------------


class _Item:
    def __init__(self, payload: dict, candidate_key: str = "") -> None:
        self.payload = payload
        self.candidate_key = candidate_key
        self.include = "resources"
        self.score = 0.5


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"fact": "a fact"}, "a fact"),
        ({"summary": "a summary"}, "a summary"),
        ({"description": "a description"}, "a description"),
        # The shape that rendered as a blank bullet: no fact, no summary.
        (
            {"label": "Runbook §2", "section_title": "Rollback", "snippet": "…"},
            "Runbook §2",
        ),
        ({"section_title": "Rollback"}, "Rollback"),
        ({"fetch": "potpie resource get potpie://res/a/b/1"}, "potpie resource get"),
    ],
)
def test_envelope_human_never_renders_an_empty_bullet(payload, expected) -> None:
    assert expected in query._item_text(_Item(payload))


def test_envelope_human_falls_back_to_the_candidate_key() -> None:
    text = query._item_text(_Item({}, candidate_key="claim:xyz"))
    assert text == "claim:xyz"


# --- M5: every advertised record type must be writable ----------------------


def test_record_detail_flag_reaches_the_structured_types() -> None:
    from potpie_context_engine.core.context_records import REQUIRED_RECORD_DETAILS

    # The five types the CLI could not reach before `--detail` existed.
    assert set(REQUIRED_RECORD_DETAILS) == {
        "bug_pattern",
        "decision",
        "preference",
        "policy",
        "verification",
    }
    for record_type, fields in REQUIRED_RECORD_DETAILS.items():
        details = {field: "value" for field in fields}
        # `outcome` is enum-constrained; use a legal member.
        if "outcome" in details:
            details["outcome"] = "worked"
        query._require_record_details(record_type, details)


def test_record_missing_detail_error_names_the_flag_to_type(capsys) -> None:
    import typer

    with pytest.raises(typer.Exit):
        query._require_record_details("decision", {})

    assert "--detail rationale" in capsys.readouterr().err


def test_record_rejects_a_value_outside_a_constrained_detail(capsys) -> None:
    import typer

    with pytest.raises(typer.Exit):
        query._require_record_details(
            "verification", {"target_ref": "fix:a", "outcome": "nope"}
        )

    stderr = capsys.readouterr().err
    assert "must be one of" in stderr
    assert "worked" in stderr


def test_record_type_help_lists_every_required_detail() -> None:
    assert "--detail rationale" in query._RECORD_TYPE_HELP
    assert "--detail policy_kind" in query._RECORD_TYPE_HELP
    assert "--detail target_ref" in query._RECORD_TYPE_HELP


def test_parse_details_rejects_a_bare_token(capsys) -> None:
    import typer

    with pytest.raises(typer.Exit):
        query._parse_details(["rationale"])

    assert "key=value" in capsys.readouterr().err


def test_parse_details_builds_the_payload_the_engine_expects() -> None:
    assert query._parse_details(["rationale=cheaper", "outcome=worked"]) == {
        "rationale": "cheaper",
        "outcome": "worked",
    }


# --- m1: --version must name the product --------------------------------


def test_version_reports_potpies_own_version() -> None:
    result = runner.invoke(cli_main.app, ["--version"])

    assert result.exit_code == 0, result.stdout
    lines = result.stdout.splitlines()
    assert lines[0].startswith("potpie ")
    assert lines[1].startswith("potpie-context-engine ")


# --- m2: --verbose must actually produce a traceback ------------------------


def test_verbose_discloses_the_traceback_and_plain_mode_does_not(capsys) -> None:
    import typer

    _common.set_json(True)
    _common.set_verbose(False)
    with pytest.raises(typer.Exit):
        with _common.contract():
            raise RuntimeError("boom in /Users/someone/private")
    quiet = json.loads(capsys.readouterr().out)
    assert quiet["message"] == "Unexpected internal error."
    assert quiet["detail"] is None
    assert "--verbose" in quiet["recommended_next_action"]

    _common.set_verbose(True)
    try:
        with pytest.raises(typer.Exit):
            with _common.contract():
                raise RuntimeError("boom in /Users/someone/private")
        captured = capsys.readouterr()
    finally:
        _common.set_verbose(False)
    loud = json.loads(captured.out)
    assert "RuntimeError" in loud["message"]
    assert loud["detail"]["traceback"]
    assert "Traceback" in captured.err


# --- m3: shipped diagnostics must not name repo-only make targets -----------


def test_shipped_diagnostics_omit_make_targets_for_published_installs() -> None:
    from potpie.cli import cli_install_status as cis

    assert not any("make" in command for command in cis._DIAGNOSTIC_COMMANDS)
    assert "make" not in cis._PUBLISHED_HINT
    status = {"primary_path": "/x/potpie", "running_on_path": True}
    assert "make" not in cis.cli_install_human(status)


# --- m6: required-field errors must name the flag the user types ------------


def test_missing_field_errors_name_the_cli_flag() -> None:
    from potpie_context_engine.core.errors import missing_field_message

    assert missing_field_message("claimed_by") == "claimed_by is required (pass --by)"
    assert missing_field_message("closed_by") == "closed_by is required (pass --by)"
    # An unmapped field still produces the plain message.
    assert missing_field_message("widget") == "widget is required"


# --- m7: skill installs must be relocatable without overriding HOME ---------


def test_harness_home_relocates_skill_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from potpie.skills.targets import ClaudeAgentTarget

    monkeypatch.setenv("POTPIE_HARNESS_HOME", str(tmp_path))
    target = ClaudeAgentTarget()
    assert target.skills_root == tmp_path / ".claude" / "skills"

    monkeypatch.delenv("POTPIE_HARNESS_HOME")
    assert str(ClaudeAgentTarget().skills_root).startswith(str(Path.home()))


def test_unmanaged_potpie_skills_are_reported(tmp_path: Path) -> None:
    """`skills status` was blind to orphaned potpie-* directories."""
    from potpie.skills.catalog import RECOMMENDED_SKILL_IDS
    from potpie.skills.targets import _unmanaged_skill_ids

    for name in (RECOMMENDED_SKILL_IDS[0], "potpie-resource-pdf", "other-tool"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "SKILL.md").write_text("x", encoding="utf-8")

    assert _unmanaged_skill_ids(tmp_path) == ("potpie-resource-pdf",)


def test_unmanaged_skill_scan_tolerates_a_missing_root(tmp_path: Path) -> None:
    from potpie.skills.targets import _unmanaged_skill_ids

    assert _unmanaged_skill_ids(tmp_path / "nope") == ()


# --- c3: repo registration must be deterministic ----------------------------


def test_repo_location_refuses_to_guess_when_the_git_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow git probe used to silently register an absolute path instead."""
    monkeypatch.setattr(
        repo_location,
        "git_remote_or_reason",
        lambda _cwd: (None, "git did not respond within 10s"),
    )

    with pytest.raises(repo_location.RepoLocationError) as raised:
        repo_location.resolve_repo_location(".")

    assert "git did not respond" in str(raised.value)


def test_repo_location_uses_the_path_when_there_is_definitively_no_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        repo_location, "git_remote_or_reason", lambda _cwd: (None, None)
    )

    assert repo_location.resolve_repo_location(".") == str(Path.cwd().resolve())


def test_git_remote_probe_separates_no_remote_from_a_failed_probe(
    tmp_path: Path,
) -> None:
    # Not a work tree at all: git exits 128, a definitive "no origin".
    assert repo_location.git_remote_or_reason(tmp_path) == (None, None)


# --- M7 (local analogue): the persisted backend must not be ignored ---------


def test_default_backend_profile_honours_the_persisted_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from potpie.runtime.composition import default_backend_profile

    monkeypatch.delenv("CONTEXT_ENGINE_BACKEND", raising=False)
    monkeypatch.delenv("GRAPH_DB_BACKEND", raising=False)
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"profile": "local", "backend": "neo4j"}), encoding="utf-8"
    )

    assert default_backend_profile() == "neo4j"

    monkeypatch.setenv("CONTEXT_ENGINE_BACKEND", "falkordb_lite")
    assert default_backend_profile() == "falkordb_lite"


def test_default_backend_profile_falls_back_without_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from potpie.runtime.composition import default_backend_profile

    monkeypatch.delenv("CONTEXT_ENGINE_BACKEND", raising=False)
    monkeypatch.delenv("GRAPH_DB_BACKEND", raising=False)
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))

    assert default_backend_profile() == "falkordb_lite"


# --- c4: an entity row must not render as a bare [?] ------------------------


@pytest.mark.parametrize(
    ("item", "key", "expected"),
    [
        ({"entity_type": "Decision"}, "decision:x", "Decision"),
        ({"labels": ["Feature"]}, "feature:x", "Feature"),
        # No entity_type: the namespaced key names the kind.
        ({}, "decision:retry-budget", "decision"),
        ({}, "service:local:payments-api", "service"),
        # A resource URI's first path segment, not its always-"potpie" scheme.
        ({}, "potpie://res/runbook/rollback/1", "res"),
        # Nothing to go on: the honest unknown marker survives.
        ({}, "opaque", "?"),
        ({}, "", "?"),
    ],
)
def test_entity_label_falls_back_to_the_key_namespace(item, key, expected) -> None:
    from potpie.cli.read_presenter import _entity_label

    assert _entity_label(item, key) == expected


# --- Review follow-ups: the safety edges around the fixes above -------------


def test_pot_reset_accepts_an_id_and_a_name_for_the_same_pot(
    repo_default_host: _Pots,
) -> None:
    """An id and its name select one pot; that is not a conflict."""
    _common.set_json(True)

    result = runner.invoke(pots.pot_app, ["reset", "alpha", "--pot", "pot-alpha"])

    emitted = json.loads(result.output)
    assert emitted["code"] == "destructive_confirmation_required"
    assert "potpie pot reset pot-alpha --confirm" in emitted["recommended_next_action"]


def test_git_probe_separates_a_fatal_error_from_a_missing_origin(
    tmp_path: Path,
) -> None:
    """Exit 128 covers both "no work tree" and a fatal config error.

    Only the former is a definitive "no origin"; treating the latter the same
    way silently registers a path identity for a repo that has a remote.
    """
    broken = tmp_path / "broken"
    broken.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=broken, check=True)
    (broken / ".git" / "config").write_text(
        "[core\n\tbroken = true\n", encoding="utf-8"
    )

    remote, failure = repo_location.git_remote_or_reason(broken)

    assert remote is None
    assert failure and "fatal" in failure

    # A directory that is simply not a work tree stays a definitive answer.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert repo_location.git_remote_or_reason(plain) == (None, None)


@pytest.mark.parametrize(
    ("command", "is_daemon"),
    [
        ("/usr/bin/python3.12 -m potpie.daemon", True),
        ("/opt/python -X dev -m potpie.daemon --flag", True),
        ("/Users/me/.local/bin/potpie-daemon", True),
        ("/usr/bin/potpied", True),
        # Near matches: a substring test would authorise a kill on all of these.
        ("/usr/bin/python -m potpie.daemon_helper", False),
        ("grep -rn potpie.daemon /src", False),
        ("/usr/bin/vim potpie.daemon.py", False),
        ('python -c "import potpie.daemon"', False),
        ("", False),
    ],
)
def test_forced_stop_only_recognises_the_real_daemon_argv(command, is_daemon) -> None:
    from potpie.daemon.lifecycle import _is_daemon_command

    assert _is_daemon_command(command) is is_daemon


def test_force_stop_reports_a_conflict_rather_than_a_false_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup that could not take ownership must not be reported as stopped."""
    from potpie.daemon.lifecycle import DaemonStopError

    daemon = Daemon(home=tmp_path, in_process=False)
    monkeypatch.setattr(
        "potpie.daemon.lifecycle._looks_like_potpie_daemon", lambda _pid: True
    )
    monkeypatch.setattr(
        "potpie.daemon.lifecycle._terminate_pid", lambda _pid: "terminated"
    )
    monkeypatch.setattr(daemon, "_cleanup_runtime_records", lambda **_kwargs: False)

    with pytest.raises(DaemonStopError) as raised:
        daemon._force_stop(4242)

    assert raised.value.error.code == "daemon_cleanup_ownership_conflict"


def test_unreadable_runtime_files_carry_the_recovery_classification(
    tmp_path: Path,
) -> None:
    """A world-readable record fails validation before the classified handlers."""
    record = daemon_discovery.discovery_path(tmp_path)
    record.write_text("{}", encoding="utf-8")
    record.chmod(0o644)

    with pytest.raises(daemon_discovery.DaemonDiscoveryError) as raised:
        daemon_discovery.read_daemon_discovery(tmp_path)

    assert raised.value.code == daemon_discovery.DISCOVERY_UNREADABLE
    assert raised.value.recommended_next_action

    credential = daemon_discovery.credential_path(tmp_path)
    credential.write_text("x" * 40, encoding="utf-8")
    credential.chmod(0o644)

    with pytest.raises(daemon_discovery.DaemonDiscoveryError) as raised:
        daemon_discovery.read_daemon_credential(tmp_path)

    assert raised.value.code == daemon_discovery.DISCOVERY_CREDENTIAL_UNAVAILABLE
    assert raised.value.recommended_next_action


def test_a_non_object_config_root_falls_back_to_the_default_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from potpie.runtime.composition import default_backend_profile

    monkeypatch.delenv("CONTEXT_ENGINE_BACKEND", raising=False)
    monkeypatch.delenv("GRAPH_DB_BACKEND", raising=False)
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text('["not", "an", "object"]', encoding="utf-8")

    assert default_backend_profile() == "falkordb_lite"


def test_each_harness_root_keeps_its_own_skill_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two harness roots sharing one state home must not share versions."""
    from potpie.skills.targets import ClaudeAgentTarget

    state_home = tmp_path / "state"
    monkeypatch.setenv("POTPIE_HARNESS_HOME", str(tmp_path / "a"))
    first = ClaudeAgentTarget(home=state_home)
    monkeypatch.setenv("POTPIE_HARNESS_HOME", str(tmp_path / "b"))
    second = ClaudeAgentTarget(home=state_home)

    assert first._path != second._path

    # The unrelocated root keeps its historical filename, so an existing
    # install's manifest is not orphaned by this change.
    monkeypatch.delenv("POTPIE_HARNESS_HOME")
    assert ClaudeAgentTarget(home=state_home)._path.name == "skills_claude_global.json"
