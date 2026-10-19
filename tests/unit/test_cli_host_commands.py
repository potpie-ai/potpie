"""``potpie host`` JSON shapes, the global ``--host`` flag, and the probe seam."""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest
from typer.testing import CliRunner

from potpie.cli.commands import _common
from potpie.cli.commands import host as host_cmds
from potpie.cli.main import app
from potpie.config import host_registry as hosts

_URL = "https://your-host.example"


@pytest.fixture
def home(monkeypatch, tmp_path):
    hosts.reset_for_tests()
    monkeypatch.setenv("CONTEXT_ENGINE_HOME", str(tmp_path))
    for name in (
        "POTPIE_MANAGED_URL",
        "POTPIE_MANAGED_TOKEN",
        "POTPIE_ALLOW_INSECURE_HOST",
    ):
        monkeypatch.delenv(name, raising=False)
    yield tmp_path
    hosts.reset_for_tests()


def _run(*args: str, stdin: str | None = None):
    return CliRunner().invoke(app, list(args), input=stdin)


# --- JSON shapes ----------------------------------------------------------------


def test_host_list_reports_both_origins(home) -> None:
    hosts.set_managed_endpoint("http://127.0.0.1:8090", "")

    result = _run("--json", "host", "list")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == {
        "hosts",
        "active_origin",
        "selected_origin",
        "persisted_origin",
        "degraded",
        "problem",
    }
    rows = {row["origin"]: row for row in payload["hosts"]}
    assert set(rows["managed"]) >= {
        "origin",
        "configured",
        "endpoint",
        "problem",
        "active",
    }
    assert rows["local"]["configured"] is True
    assert rows["local"]["endpoint"] == str(home)
    assert rows["managed"]["configured"] is True
    assert rows["managed"]["endpoint"] == "http://127.0.0.1:8090"
    assert payload["active_origin"] == "local"
    assert payload["degraded"] is False


def test_host_list_says_so_when_the_selected_host_is_not_configured(home) -> None:
    hosts.set_persisted_origin(hosts.MANAGED)

    payload = json.loads(_run("--json", "host", "list").output)
    assert payload["degraded"] is True
    assert payload["persisted_origin"] == hosts.MANAGED
    assert payload["selected_origin"] == hosts.MANAGED
    assert payload["active_origin"] == hosts.LOCAL
    assert "but none is configured" in _run("host", "list").output


def test_a_configured_managed_host_is_not_degraded(home) -> None:
    hosts.set_managed_endpoint(_URL, "tok")
    hosts.set_persisted_origin(hosts.MANAGED)

    payload = json.loads(_run("--json", "host", "list").output)
    assert payload["degraded"] is False
    assert payload["active_origin"] == hosts.MANAGED
    assert "but none is configured" not in _run("host", "list").output


def test_host_set_payload_shape(home, monkeypatch) -> None:
    monkeypatch.setattr(host_cmds, "_probe_managed", lambda *_a: [object()])

    checked = _run("--json", "host", "set", _URL, "--token", "-", stdin="tok\n")
    unchecked = _run(
        "--json", "host", "set", _URL, "--no-check", "--token", "-", stdin="tok\n"
    )

    assert checked.exit_code == 0, checked.output
    assert json.loads(checked.output) == {
        "origin": "managed",
        "endpoint": _URL,
        "checked": True,
        "allow_insecure_host": None,
        "pots": 1,
        "shadowed_by_env": None,
    }
    assert unchecked.exit_code == 0, unchecked.output
    assert "pots" not in json.loads(unchecked.output)
    assert json.loads(unchecked.output)["checked"] is False


def test_host_clear_payload_shape(home) -> None:
    hosts.set_managed_endpoint(_URL, "tok")

    result = _run("--json", "host", "clear")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "origin": "managed",
        "endpoint": None,
        "salvaged_registry": None,
    }
    assert hosts.managed_endpoint() is None


def test_host_use_persists_the_origin(home) -> None:
    hosts.set_managed_endpoint(_URL, "tok")

    result = _run("--json", "host", "use", "Managed")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"active_origin": "managed"}
    assert hosts.persisted_origin() == hosts.MANAGED

    back = _run("--json", "host", "use", "local")
    assert json.loads(back.output) == {"active_origin": "local"}
    assert hosts.persisted_origin() == hosts.LOCAL


def test_host_use_refuses_an_unknown_or_unconfigured_origin(home) -> None:
    unknown = _run("--json", "host", "use", "prod")
    assert unknown.exit_code == _common.EXIT_VALIDATION, unknown.output
    assert json.loads(unknown.output)["code"] == "validation_error"
    assert "potpie host list" in json.loads(unknown.output)["recommended_next_action"]

    unconfigured = _run("--json", "host", "use", "managed")
    assert unconfigured.exit_code == _common.EXIT_VALIDATION, unconfigured.output
    error = json.loads(unconfigured.output)
    assert error["message"] == "No managed host is configured."
    assert "host set" in error["recommended_next_action"]
    assert hosts.persisted_origin() == hosts.LOCAL


# --- the global --host flag -------------------------------------------------------


def test_host_flag_targets_one_invocation_without_persisting(home) -> None:
    hosts.set_managed_endpoint(_URL, "tok")

    result = _run("--json", "--host", "managed", "host", "list")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["active_origin"] == "managed"
    assert payload["selected_origin"] == "managed"
    assert payload["persisted_origin"] == "local"
    assert hosts.persisted_origin() == "local"

    # The next invocation in the same process does not inherit it.
    later = json.loads(_run("--json", "host", "list").output)
    assert later["active_origin"] == "local"


def test_an_unknown_host_flag_is_rejected_at_the_door(home) -> None:
    result = _run("--json", "--host", "prod", "host", "list")

    assert not isinstance(result.exception, ValueError), result.exception
    assert "Traceback" not in result.output
    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "prod" in payload["message"]
    assert "potpie host list" in payload["recommended_next_action"]


def test_host_override_refuses_an_unconfigured_managed_host(home) -> None:
    result = _run("--json", "--host", "managed", "host", "list")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert payload["message"] == "No managed host is configured."
    assert "host set" in payload["recommended_next_action"]
    assert "hosts" not in payload


def test_host_override_names_an_unusable_override_instead_of_the_file(
    home, monkeypatch
) -> None:
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://127.0.0.1:8090x")

    result = _run("--json", "--host", "managed", "host", "list")

    assert result.exit_code == _common.EXIT_VALIDATION, result.output
    payload = json.loads(result.output)
    assert payload["code"] == "validation_error"
    assert "unusable" in payload["message"]
    assert "POTPIE_MANAGED_URL" in payload["message"]
    assert "host set" not in payload["recommended_next_action"]


def test_host_override_with_an_unreadable_registry_is_the_unavailable_envelope(
    home,
) -> None:
    (home / "cli_hosts.json").write_text("{broken")

    result = _run("--json", "--host", "managed", "host", "list")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    assert json.loads(result.output)["code"] == "unavailable"


# --- the probe seam ---------------------------------------------------------------


def test_check_without_a_remote_client_fails_clearly_and_writes_nothing(
    home, monkeypatch
) -> None:
    monkeypatch.setitem(sys.modules, "potpie.runtime.remote", None)

    result = _run("--json", "host", "set", _URL, "--token", "-", stdin="tok\n")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    error = json.loads(result.output)
    assert error["code"] == "not_implemented"
    assert "--no-check" in error["recommended_next_action"]
    assert not (home / "cli_hosts.json").exists()


def test_any_probe_failure_is_unavailable_and_names_the_endpoint(
    home, monkeypatch
) -> None:
    def _boom(*_args: Any) -> Any:
        raise RuntimeError("401 unauthorized")

    monkeypatch.setattr(host_cmds, "_probe_managed", _boom)

    result = _run("--json", "host", "set", _URL, "--token", "-", stdin="tok\n")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    error = json.loads(result.output)
    assert error["code"] == "unavailable"
    assert _URL in error["message"]
    assert not (home / "cli_hosts.json").exists()


def test_the_probe_handshakes_and_lists_pots_on_the_offered_endpoint(
    home, monkeypatch
) -> None:
    """The probe builds the offered endpoint, handshakes, then lists pots."""
    from potpie.runtime import remote

    calls: list[Any] = []

    class _Endpoint:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("endpoint", kwargs))

    class _Listing:
        pots = ["p1", "p2"]

    class _Client:
        def __init__(self, endpoint: Any) -> None:
            calls.append(("client", type(endpoint).__name__))

        async def handshake(self) -> None:
            calls.append("handshake")

        async def pot_list(self) -> _Listing:
            calls.append("host.pot_list")
            return _Listing()

    monkeypatch.setattr(remote, "ManagedHostEndpoint", _Endpoint)
    monkeypatch.setattr(remote, "ManagedHostProbe", _Client)

    result = _run("--json", "host", "set", _URL, "--token", "-", stdin="tok\n")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["pots"] == 2
    assert calls == [
        (
            "endpoint",
            {"base_url": _URL, "token": "tok", "allow_insecure_host": None},
        ),
        ("client", "_Endpoint"),
        "handshake",
        "host.pot_list",
    ]


# --- the real probe against a managed host ------------------------------------------


def _reference_probe(monkeypatch, tmp_path):
    """Point ``host set --check`` at the reference host through the real client."""
    import asyncio
    import functools

    from potpie.runtime import remote
    from tests.support.reference_host import ReferenceHost

    reference = ReferenceHost.build(tmp_path / "reference-host")
    monkeypatch.setattr(
        remote,
        "ManagedHostProbe",
        functools.partial(
            remote.ManagedHostProbe, http_transport=reference.httpx_transport()
        ),
    )
    return reference, lambda: asyncio.run(reference.close())


def test_host_set_check_probes_a_real_managed_host(home, monkeypatch, tmp_path) -> None:
    from tests.support.reference_host import ALICE_KEY

    _reference, close = _reference_probe(monkeypatch, tmp_path)

    result = _run("--json", "host", "set", _URL, "--token", "-", stdin=f"{ALICE_KEY}\n")

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["checked"] is True
    assert payload["pots"] == 2
    assert hosts.managed_endpoint() == (_URL, ALICE_KEY)
    close()


def test_host_set_check_refused_by_the_host_stores_nothing(
    home, monkeypatch, tmp_path
) -> None:
    _reference, close = _reference_probe(monkeypatch, tmp_path)

    result = _run("--json", "host", "set", _URL, "--token", "-", stdin="wrong-key\n")

    assert result.exit_code == _common.EXIT_UNAVAILABLE, result.output
    error = json.loads(result.output)
    assert error["code"] == "unavailable"
    assert "API key was refused" in error["message"]
    assert not (home / "cli_hosts.json").exists()
    close()
