"""Managed-host routing: two hosts, one CLI.

The risk this feature carries is not "does the listing render": it is running a
command against the wrong graph because a name existed on both hosts, or
silently falling back to the local host when the managed one was asked for.
Most of what is pinned here is routing and refusals; the pot directories are
in-memory doubles (``tests/support/pot_directories.py``).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common
from potpie.cli.main import app
from potpie.config import host_registry
from potpie.runtime import ContextSelector
from potpie.runtime.protocol import ProtocolError
from tests.support.pot_directories import two_hosts


@pytest.fixture
def registry(monkeypatch):
    """A configured managed host beside the local one, both answering."""
    host_registry.reset_for_tests()
    built = two_hosts(monkeypatch)
    yield built
    _common.set_pot_directory_factory(None)
    host_registry.reset_for_tests()


def _run(*args: str, **kwargs: Any):
    return CliRunner().invoke(app, list(args), **kwargs)


# --- ref parsing and the origin pointer -------------------------------------


def test_only_a_known_origin_counts_as_a_prefix() -> None:
    assert host_registry.split_ref("managed:api") == ("managed", "api")
    assert host_registry.split_ref("local:default") == ("local", "default")
    # A pot may be named with a colon; stealing every prefix would make such a
    # name unaddressable.
    assert host_registry.split_ref("team:api") == (None, "team:api")
    assert host_registry.split_ref("managed:") == (None, "managed:")


def test_a_persisted_origin_that_is_no_longer_configured_degrades(
    monkeypatch,
) -> None:
    """Clearing the managed host must not brick every command."""
    host_registry.reset_for_tests()
    monkeypatch.setattr(host_registry, "managed_endpoint", lambda: ("https://s", "t"))
    host_registry.set_persisted_origin(host_registry.MANAGED)
    assert host_registry.current_origin() == host_registry.MANAGED

    monkeypatch.setattr(host_registry, "managed_endpoint", lambda: None)
    assert host_registry.current_origin() == host_registry.LOCAL


# --- listing ----------------------------------------------------------------


def test_pot_list_merges_both_hosts_and_tags_origin(registry) -> None:
    result = _run("--json", "pot", "list")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [(row["origin"], row["name"]) for row in payload["pots"]] == [
        ("local", "default"),
        ("local", "notes"),
        ("managed", "default"),
        ("managed", "api"),
    ]
    assert payload["active_origin"] == "local"
    # Each host keeps its own pointer; only the active origin's is current.
    assert [row["name"] for row in payload["pots"] if row["current"]] == ["default"]
    assert sorted(row["name"] for row in payload["pots"] if row["active"]) == [
        "api",
        "default",
    ]
    assert set(payload["pots"][0]) == {
        "id",
        "name",
        "active",
        "archived",
        "current",
        "origin",
    }


def test_pot_list_managed_reads_only_the_managed_host(registry) -> None:
    result = _run("--json", "pot", "list", "--managed")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {row["origin"] for row in payload["pots"]} == {"managed"}
    assert registry["local"].list_calls == 0


def test_pot_list_local_flag_does_not_touch_the_remote(registry) -> None:
    result = _run("--json", "pot", "list", "--local")

    assert result.exit_code == 0, result.output
    assert {row["origin"] for row in json.loads(result.output)["pots"]} == {"local"}
    assert registry["managed"].list_calls == 0


def test_pot_list_degrades_when_the_remote_is_down(registry) -> None:
    from potpie_context_engine.core.errors import ContextEngineDisabled

    registry["managed"].raises = ContextEngineDisabled("connection refused")

    result = _run("--json", "pot", "list")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {row["origin"] for row in payload["pots"]} == {"local"}
    assert "connection refused" in payload["unavailable"]["managed"]


# --- bare and qualified refs --------------------------------------------------


def test_a_bare_name_on_both_hosts_resolves_on_the_current_origin_only(
    registry,
) -> None:
    """Deliberate deviation from searching every host: a bare ref targets the
    current origin, so 'default' is the local pot here and the managed host is
    not even enumerated."""
    pot = _common.resolve_pot_ref(None, "default")

    assert (pot.pot_id, _common.current_origin()) == ("pot_l1", "local")
    assert registry["managed"].list_calls == 0


def test_a_bare_miss_names_the_pot_on_the_other_host(registry) -> None:
    result = _run("--json", "pot", "use", "api")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "pot_not_found"
    assert "on the local host" in payload["message"]
    assert "'managed:api'" in payload["message"]
    assert "--pot" not in payload["recommended_next_action"]
    assert "potpie pot use managed:api" in payload["recommended_next_action"]
    assert payload["detail"] == {"origin": "local", "elsewhere": ["managed:api"]}
    assert registry["local"].used == []
    assert registry["managed"].used == []


def test_a_bare_miss_on_both_hosts_names_no_alternative(registry) -> None:
    result = _run("--json", "pot", "use", "nowhere")

    payload = json.loads(result.output)
    assert payload["code"] == "pot_not_found"
    assert payload["recommended_next_action"] == "run 'potpie pot list'"
    assert payload["detail"] is None


def test_a_qualified_ref_moves_the_whole_command(registry) -> None:
    pot = _common.resolve_pot_ref(None, "managed:default")

    assert (pot.pot_id, _common.current_origin()) == ("pot_m1", "managed")
    # Not persisted: a qualified ref targets one invocation.
    assert host_registry.persisted_origin() == "local"


def test_a_qualified_ref_for_a_missing_pot_names_the_host(registry) -> None:
    result = _run("--json", "pot", "use", "managed:notes")

    assert result.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(result.output)
    assert payload["code"] == "pot_not_found"
    assert "on the managed host" in payload["message"]
    assert "'local:notes'" in payload["message"]


def test_a_qualified_managed_ref_without_a_managed_host_is_refused(
    monkeypatch,
) -> None:
    host_registry.reset_for_tests()
    two_hosts(monkeypatch, managed_configured=False)

    result = _run("--json", "pot", "use", "managed:api")

    # The same refusal as `--host managed` with no managed host configured,
    # before anything is sent.
    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "No managed host is configured" in payload["message"]


# --- use persists the origin it selected on ---------------------------------


def test_use_moves_the_active_origin_and_persists_it(registry) -> None:
    result = _run("--json", "pot", "use", "managed:api")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert (payload["id"], payload["origin"]) == ("pot_m2", "managed")
    assert registry["managed"].used == ["api"]
    assert host_registry.persisted_origin() == "managed"


def test_a_failed_use_does_not_move_the_pointer(registry) -> None:
    from potpie_context_engine.core.errors import ContextEngineDisabled

    registry["managed"].raises = ContextEngineDisabled("connection refused")

    result = _run("--json", "pot", "use", "managed:api")

    assert result.exit_code == _common.EXIT_UNAVAILABLE
    assert host_registry.persisted_origin() == "local"


def test_also_default_binds_the_repo_on_the_same_host_before_persisting(
    registry, monkeypatch
) -> None:
    monkeypatch.setattr(
        _common, "current_repo_identity_for_cli", lambda: "github.com/acme/shop"
    )

    result = _run(
        "--json", "pot", "use", "managed:api", "--also-default-for-current-repo"
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["repo_default_set"] is True
    assert registry["managed"].defaults == {"github.com/acme/shop": "pot_m2"}
    assert registry["local"].defaults == {}
    assert host_registry.persisted_origin() == "managed"


def test_a_refused_repo_default_leaves_the_pointer_alone(registry, monkeypatch) -> None:
    monkeypatch.setattr(
        _common, "current_repo_identity_for_cli", lambda: "github.com/acme/shop"
    )

    def _refuse(**_kwargs: Any) -> None:
        raise ValueError("refused")

    monkeypatch.setattr(registry["managed"], "set_repo_default", _refuse)

    result = _run(
        "--json", "pot", "use", "managed:api", "--also-default-for-current-repo"
    )

    assert result.exit_code == _common.EXIT_VALIDATION
    assert host_registry.persisted_origin() == "local"


def test_a_path_repo_identity_is_never_bound_on_a_managed_host(
    registry, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(_common, "current_repo_identity_for_cli", lambda: str(tmp_path))

    result = _run(
        "--json", "pot", "use", "managed:api", "--also-default-for-current-repo"
    )

    assert result.exit_code == _common.EXIT_VALIDATION
    assert "git remote" in json.loads(result.output)["message"]
    assert registry["managed"].used == []
    assert registry["managed"].defaults == {}


# --- --host ------------------------------------------------------------------


def test_host_flag_targets_one_invocation_without_persisting(registry) -> None:
    result = _run("--json", "--host", "managed", "pot", "info")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["active_pot"]["id"] == "pot_m2"
    assert host_registry.persisted_origin() == "local"


def test_an_unknown_host_flag_is_rejected_at_the_door(registry) -> None:
    result = _run("--json", "--host", "elsewhere", "pot", "list")

    assert result.exit_code == _common.EXIT_VALIDATION
    assert json.loads(result.output)["code"] == "validation_error"


def test_whoami_answers_for_the_current_host(registry) -> None:
    local = json.loads(_run("--json", "whoami").output)
    managed = json.loads(_run("--json", "--host", "managed", "whoami").output)

    assert local["subject"] == "local-user"
    assert managed == {"subject": "alice", "mode": "managed", "detail": None}


def test_setup_refuses_to_target_a_managed_host(registry) -> None:
    result = _run("--json", "--host", "managed", "setup", "--dry-run")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "local machine" in payload["message"]


# --- grants -------------------------------------------------------------------


def test_grants_are_refused_on_the_local_host(registry) -> None:
    result = _run("--json", "pot", "grants", "default")

    assert result.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(result.output)
    assert payload["code"] == "unsupported"
    assert "single-user" in payload["message"]


def test_grants_round_trip_on_the_managed_host(registry) -> None:
    granted = json.loads(
        _run("--json", "pot", "grant", "managed:api", "bob", "--role", "reader").output
    )
    listed = json.loads(_run("--json", "pot", "grants", "managed:api").output)
    revoked = json.loads(_run("--json", "pot", "revoke", "managed:api", "bob").output)
    again = json.loads(_run("--json", "pot", "revoke", "managed:api", "bob").output)

    assert granted == {"pot_id": "pot_m2", "actor_id": "bob", "role": "reader"}
    assert listed == {
        "pot": "managed:api",
        "grants": [{"pot_id": "pot_m2", "actor_id": "bob", "role": "reader"}],
    }
    assert revoked == {"pot": "managed:api", "actor_id": "bob", "revoked": True}
    assert again["revoked"] is False


# --- engine commands on the managed origin ------------------------------------


class _Captured:
    def __init__(self) -> None:
        self.selectors: list[ContextSelector] = []

    def engine(self, selector: ContextSelector) -> Any:
        self.selectors.append(selector)
        return SimpleNamespace(selector=selector)


@pytest.fixture
def managed_engine(monkeypatch) -> _Captured:
    captured = _Captured()
    monkeypatch.setattr(_common, "_managed_connection", lambda: captured)
    return captured


def test_engine_clients_on_the_managed_origin_go_to_the_host(
    registry, managed_engine
) -> None:
    host_registry.set_current_origin("managed")

    _common.get_engine_client("api")

    assert managed_engine.selectors == [ContextSelector(kind="explicit", value="api")]


def test_a_qualified_engine_ref_reaches_the_host_bare(registry, managed_engine) -> None:
    _common.get_engine_client("managed:api")

    assert managed_engine.selectors == [ContextSelector(kind="explicit", value="api")]
    assert _common.current_origin() == "managed"


def test_a_git_remote_repo_identity_is_sent_as_a_repository_selector(
    registry, managed_engine, monkeypatch
) -> None:
    host_registry.set_current_origin("managed")
    monkeypatch.setattr(
        _common, "_current_git_remote", lambda _cwd: "github.com/acme/shop"
    )

    _common.get_engine_client()

    assert managed_engine.selectors == [
        ContextSelector(kind="repository", value="github.com/acme/shop")
    ]


def test_a_path_repo_identity_is_never_sent_to_a_managed_host(
    registry, managed_engine, monkeypatch
) -> None:
    """No git remote means the active pot: a local path never leaves the machine."""
    host_registry.set_current_origin("managed")
    monkeypatch.setattr(_common, "_current_git_remote", lambda _cwd: None)

    _common.get_engine_client()

    assert managed_engine.selectors == [ContextSelector(kind="active")]


def test_local_engine_clients_stay_local_with_a_local_ref(
    registry, managed_engine
) -> None:
    host_registry.set_current_origin("managed")

    client = _common.get_engine_client("local:default")

    assert managed_engine.selectors == []
    assert client._selector == ContextSelector(kind="explicit", value="default")


# --- presentation of a host's refusals ----------------------------------------


def _boundary_failure(error: ProtocolError | Any) -> Any:
    from potpie.cli.commands._common import EngineClientError, contract

    with contract():
        raise EngineClientError(error)


def test_host_outdated_is_presented_as_not_implemented(capsys) -> None:
    from potpie.runtime.remote import host_outdated_error

    error = host_outdated_error(
        ProtocolError(
            code="request_fields_unsupported",
            message="unknown fields",
            details={"operation": "history", "fields": ["include_claims"]},
        ),
        host="Managed (svc.example)",
        operation="history",
    )
    _common.set_json(True)

    with pytest.raises(_common.CliExpectedFailureExit) as raised:
        _boundary_failure(error)

    assert raised.value.exit_code == _common.EXIT_UNAVAILABLE
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "not_implemented"
    assert payload["message"].startswith("Capability not implemented: history — ")
    assert "older Potpie" in payload["message"]
    assert "'include_claims'" in payload["detail"]
    assert payload["recommended_next_action"] == _common.HOST_OUTDATED_NEXT_ACTION


def test_a_managed_hosts_refusal_through_the_directory_keeps_its_code(
    registry,
) -> None:
    from potpie.runtime.protocol import RuntimeBoundaryError  # noqa: F401
    from potpie.runtime.remote import RemoteHostError
    from potpie.runtime.resource_manager import AuthorizationError

    registry["managed"].raises = RemoteHostError(
        AuthorizationError(code="role_insufficient", message="needs admin")
    )

    result = _run("--json", "pot", "list", "--managed")

    # Enumeration degrades even on an auth refusal; the reason is stated.
    assert result.exit_code == _common.EXIT_UNAVAILABLE
    assert "needs admin" in json.loads(result.output)["message"]

    result = _run("--json", "pot", "use", "managed:api")

    assert result.exit_code == _common.EXIT_AUTH, result.output
    assert json.loads(result.output)["code"] == "role_insufficient"
