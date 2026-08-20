"""One error contract for the whole CLI: one envelope, one exit-code table.

The failures these lock down were all the *same* failure wearing different
clothes — the shape of an error depended on who raised it, and the exit code
depended on who was reading it:

- a mistyped flag exited 2 for a human and 1 under ``--json``;
- a managed host that answered 401 was reported as "unavailable" at exit 2,
  indistinguishable from a daemon that was never started, while ``EXIT_AUTH``
  sat unused;
- ``graph`` errors and every other error shared no JSON key, so ``resp['code']``
  raised ``KeyError`` on exactly the commands agents call most;
- ``--json`` after the subcommand was rejected in prose, at exit 2, to a caller
  that had just said it could only read JSON — it is now hoisted and honoured,
  so the position a caller guesses cannot change the answer;
- ``potpie --json --version`` answered a machine with two lines of prose;
- a missing *source* was reported as a missing *pot*, because the error type
  subclasses the one above it in the boundary.

Everything here drives ``run_cli`` (the shipped entrypoint) or the real error
boundary. ``CliRunner`` invokes the app in Click's standalone mode and never
enters ``run_cli``, so it cannot defend this contract.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from potpie.cli import hosts
from potpie.cli import main as host_cli
from potpie.cli.commands import _common
from potpie.cli.commands import graph as graph_cmds
from potpie_context_core.errors import ContextEngineDisabled

CLI_FLOW_DOC = Path(__file__).resolve().parents[2] / "docs/context-graph/cli-flow.md"


@pytest.fixture(autouse=True)
def _reset_cli_output_state() -> Iterator[None]:
    """Keep the argv-flag memory from leaking between tests.

    ``run_cli`` remembers what its argv scan found so a trailing ``--json`` is
    not undone by the root callback; that memory is scoped to one command line
    and this fixture is the test-side half of the same promise.
    """
    _common.set_json(False)
    _common.set_verbose(False)
    _common.clear_argv_output_flags()
    _common.set_host(None)
    yield
    _common.set_json(False)
    _common.set_verbose(False)
    _common.clear_argv_output_flags()
    _common.set_host(None)


# --- the table --------------------------------------------------------------


@pytest.mark.parametrize(
    "code, expected",
    [
        ("unavailable", _common.EXIT_UNAVAILABLE),
        ("not_implemented", _common.EXIT_UNAVAILABLE),
        ("degraded", _common.EXIT_DEGRADED),
        ("auth_error", _common.EXIT_AUTH),
        ("validation_error", _common.EXIT_VALIDATION),
        ("usage_error", _common.EXIT_VALIDATION),
        ("pot_not_found", _common.EXIT_VALIDATION),
        ("ambiguous_pot", _common.EXIT_VALIDATION),
        ("no_active_pot", _common.EXIT_VALIDATION),
        ("unexpected_cli_error", _common.EXIT_VALIDATION),
        ("something_nobody_classified", _common.EXIT_VALIDATION),
    ],
)
def test_the_exit_code_table_is_a_function_of_the_error_code(
    code: str, expected: int
) -> None:
    assert _common.exit_code_for(code) == expected


@pytest.mark.parametrize(
    "code, expected",
    [
        ("unavailable", _common.EXIT_UNAVAILABLE),
        ("not_implemented", _common.EXIT_UNAVAILABLE),
        ("auth_error", _common.EXIT_AUTH),
        ("validation_error", _common.EXIT_VALIDATION),
    ],
)
def test_fail_without_an_explicit_exit_code_uses_the_table(
    code: str, expected: int, capsys: pytest.CaptureFixture[str]
) -> None:
    """The class of drift where one code meant two exit codes is unrepresentable."""
    _common.set_json(True)
    with pytest.raises(typer.Exit) as exc_info:
        _common.fail(code=code, message="boom")

    assert exc_info.value.exit_code == expected
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == code
    assert payload["ok"] is False


def test_the_documented_table_matches_the_code() -> None:
    """Cheap doc-drift guard, so "stated once" keeps meaning stated once."""
    rows = re.findall(
        r"^\s*\|[^|]+\|\s*`([a-z_]+)`\s*\|\s*`(\d)`\s*\|\s*$",
        CLI_FLOW_DOC.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert rows, "the exit table vanished from cli-flow.md"
    for code, exit_code in rows:
        assert _common.exit_code_for(code) == int(exit_code), code
    documented = {code for code, _ in rows}
    assert {"unavailable", "not_implemented", "auth_error", "usage_error"} <= documented


# --- usage errors: one exit code in both output modes -----------------------


USAGE_ARGV = [
    ["pot", "create"],  # missing argument
    ["nosuchcmd"],  # unknown command
    ["pot"],  # missing subcommand
    ["host", "list", "--nope"],  # unknown option
]


@pytest.mark.parametrize("argv", USAGE_ARGV, ids=lambda a: "-".join(a))
def test_usage_errors_exit_the_same_in_both_output_modes(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(typer.Exit) as human:
        host_cli.run_cli(list(argv))
    human_out = capsys.readouterr()

    with pytest.raises(typer.Exit) as machine:
        host_cli.run_cli(["--json", *argv])
    machine_out = capsys.readouterr()

    assert human.value.exit_code == _common.EXIT_VALIDATION
    assert machine.value.exit_code == _common.EXIT_VALIDATION
    assert human_out.out == ""
    payload = json.loads(machine_out.out)
    assert payload["ok"] is False
    assert payload["code"] == "usage_error"
    assert payload["recommended_next_action"]


# --- a global flag after the command means the same thing --------------------


def _drive(argv: list[str]) -> int:
    """Run the shipped entrypoint and return its exit code."""
    try:
        host_cli.run_cli(list(argv))
    except typer.Exit as exc:
        return int(exc.exit_code or 0)
    return 0


@pytest.mark.parametrize(
    "command",
    [
        ["pot", "list"],  # succeeds
        ["graph", "catalog"],  # the workbench envelope
        ["pot", "create"],  # fails, before the command body
    ],
    ids=lambda c: "-".join(c),
)
def test_a_global_flag_means_the_same_thing_in_either_position(
    command: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """One accepted position is a coin flip every agent consumer has to win.

    The losing side got Click's prose on stderr from a command line that had
    just said it could only read JSON, so both spellings are driven here and
    required to answer with the same exit code and the same envelope keys.
    """
    leading = _drive(["--json", *command])
    leading_out = capsys.readouterr().out
    trailing = _drive([*command, "--json"])
    trailing_out = capsys.readouterr().out

    assert trailing == leading
    # json.loads is the assertion that matters: Click's refusal is not JSON.
    assert sorted(json.loads(trailing_out)) == sorted(json.loads(leading_out))


@pytest.mark.parametrize("flag", ["--verbose", "-v"])
def test_a_trailing_verbose_flag_is_applied_rather_than_rejected(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _drive(["pot", "list", flag]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert _common.is_verbose()


def test_a_trailing_host_flag_keeps_its_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Hoisting a value-taking flag has to move the value with it.

    ``potpie status --host managed`` used to reach the caller as "Got unexpected
    extra argument(s) (managed)" — a message naming neither the flag nor the
    fix — because ``status`` declared a boolean ``--host`` of its own that ate
    the flag and left the value stranded.
    """
    assert (
        _drive(["--json", "pot", "list", "--host", "bogus"]) == _common.EXIT_VALIDATION
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "validation_error"
    assert "bogus" in payload["message"]


def test_a_value_taking_global_with_no_value_still_names_the_position(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The one shape that cannot be hoisted keeps the old refusal, with a metavar."""
    assert _drive(["--json", "pot", "list", "--host"]) == _common.EXIT_VALIDATION
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "usage_error"
    assert "global flag" in payload["message"]
    assert "--host <local|managed>" in payload["recommended_next_action"]


def test_a_command_flag_is_not_mistaken_for_a_misplaced_global(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The match is keyed on Click's ``NoSuchOption``, not on a scan of argv.

    ``gitbucket login`` owns a ``--host`` of its own, so Click never raises
    ``NoSuchOption`` for it there and the root flag cannot shadow it.
    """
    from typer._click.exceptions import NoSuchOption

    assert host_cli._misplaced_root_flag(NoSuchOption("--pot")) is None
    assert host_cli._misplaced_root_flag(NoSuchOption("--json")) == "--json"
    assert host_cli._misplaced_root_flag(NoSuchOption("--host")) == "--host"

    from typer.testing import CliRunner

    result = CliRunner().invoke(host_cli.app, ["gitbucket", "login", "--help"])
    assert "--host" in result.output

    with pytest.raises(typer.Exit):
        host_cli.run_cli(["--json", "host", "list", "--nope"])
    payload = json.loads(capsys.readouterr().out)
    assert "global flag" not in payload["message"]
    assert "--nope" in payload["message"]


def test_the_root_callback_does_not_downgrade_an_argv_json_request(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Names the exact mechanism the trailing-``--json`` bug rode in on."""
    _common.bootstrap_output_flags_from_argv(["pot", "list", "--json"])
    assert _common.argv_requested_json()

    assert _drive(["pot", "list", "--json"]) == 0

    # The root callback ran (it is what parses `pot`), and JSON survived it.
    assert _common.is_json()
    json.loads(capsys.readouterr().out)


def test_a_trailing_version_flag_still_answers_in_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The eager callback fires on the *second* parse, after the scan is spent."""
    assert _drive(["pot", "list", "--version", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["name"] == "potpie"


# --- a host that answered and refused the credential ------------------------


class _RefusingHandler(BaseHTTPRequestHandler):
    status = 401

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        body = json.dumps({"detail": "Invalid API key"}).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # noqa: A002 - silence the server
        return


@pytest.fixture(params=[401, 403])
def refusing_host_url(request: pytest.FixtureRequest) -> Iterator[str]:
    """A real HTTP server that answers and rejects the credential.

    Deliberately not a fake at the adapter boundary: the whole contract under
    test is that the transport notices a status code and the CLI boundary reads
    it back off the exception, and a stubbed exception would model exactly the
    coupling that is supposed to be verified.
    """
    handler = type("_H", (_RefusingHandler,), {"status": request.param})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_daemon_rpc_client_marks_a_refused_credential_on_the_exception(
    refusing_host_url: str,
) -> None:
    host = hosts.build_managed_host(refusing_host_url, "wrong-token")

    with pytest.raises(ContextEngineDisabled) as exc_info:
        host.pots.list_pots()

    exc = exc_info.value
    assert exc.status_code in _common.CREDENTIAL_REFUSED_STATUSES
    assert "refused the credential" in str(exc)
    assert "Managed (" in str(exc)
    assert refusing_host_url in str(exc)
    assert "host set" in (exc.recommended_next_action or "")


def test_a_refused_credential_is_exit_auth_not_unavailable(
    refusing_host_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    host = hosts.build_managed_host(refusing_host_url, "wrong-token")
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        with _common.contract():
            host.pots.list_pots()

    assert exc_info.value.exit_code == _common.EXIT_AUTH
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "auth_error"
    assert "Managed (" in payload["message"]
    assert refusing_host_url in payload["message"]
    assert "host set" in payload["recommended_next_action"]


def test_an_unreachable_host_is_still_unavailable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The auth branch must not have swallowed the rest of the family."""
    server = HTTPServer(("127.0.0.1", 0), _RefusingHandler)
    dead_url = f"http://127.0.0.1:{server.server_port}"
    server.server_close()

    host = hosts.build_managed_host(dead_url, "any-token")
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        with _common.contract():
            host.pots.list_pots()

    assert exc_info.value.exit_code == _common.EXIT_UNAVAILABLE
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "unavailable"


# --- one envelope across every emitter --------------------------------------


SHARED_ERROR_KEYS = {"ok", "code", "message", "detail", "recommended_next_action"}


def _graph_error_payload(capsys: pytest.CaptureFixture[str]) -> dict:
    _common.set_json(True)
    with pytest.raises(typer.Exit):
        with graph_cmds._graph_command("graph.read"):
            raise ValueError("no such view: 'bogus'")
    return json.loads(capsys.readouterr().out)


def test_every_error_path_shares_one_envelope(
    refusing_host_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    payloads: list[dict] = []

    _common.set_json(True)
    with pytest.raises(typer.Exit):
        with _common.contract():
            raise ValueError("no such view: 'bogus'")
    payloads.append(json.loads(capsys.readouterr().out))

    with pytest.raises(typer.Exit):
        with _common.contract():
            raise ContextEngineDisabled("backend is down")
    payloads.append(json.loads(capsys.readouterr().out))

    host = hosts.build_managed_host(refusing_host_url, "wrong-token")
    with pytest.raises(typer.Exit):
        with _common.contract():
            host.pots.list_pots()
    payloads.append(json.loads(capsys.readouterr().out))

    with pytest.raises(typer.Exit):
        host_cli.run_cli(["--json", "pot", "create"])
    payloads.append(json.loads(capsys.readouterr().out))

    payloads.append(_graph_error_payload(capsys))

    for payload in payloads:
        assert SHARED_ERROR_KEYS <= set(payload), payload
        assert payload["ok"] is False, payload
        assert payload["code"], payload
        assert payload["message"], payload


def test_graph_errors_keep_their_workbench_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The flattening is additive: the workbench contract is untouched."""
    payload = _graph_error_payload(capsys)

    assert payload["command"] == "graph.read"
    assert payload["request_id"].startswith("req:")
    assert payload["graph_contract_version"]
    assert payload["error"]["code"] == payload["code"] == "validation_error"
    assert payload["error"]["message"] == payload["message"]
    assert payload["error"]["detail"] == payload["detail"]


@pytest.mark.parametrize(
    "status, expected",
    [
        ("unavailable", _common.EXIT_UNAVAILABLE),
        ("not_implemented", _common.EXIT_UNAVAILABLE),
        ("validation_error", _common.EXIT_VALIDATION),
        ("invalid", _common.EXIT_VALIDATION),
    ],
)
def test_graph_service_failures_use_the_shared_exit_table(
    status: str, expected: int, capsys: pytest.CaptureFixture[str]
) -> None:
    ctx = graph_cmds._GraphCliCommandContext("graph.read")
    _common.set_json(True)

    with pytest.raises(typer.Exit) as exc_info:
        graph_cmds._emit_graph_result(
            ctx,
            {"ok": False, "status": status, "message": "nope"},
            human="nope",
        )

    assert exc_info.value.exit_code == expected
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == status
    assert payload["error"]["code"] == status


# --- nothing escapes as a traceback -----------------------------------------


def test_an_unhandled_exception_becomes_the_envelope_not_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch.object(host_cli, "app", side_effect=RuntimeError("boom")):
        with pytest.raises(typer.Exit) as exc_info:
            host_cli.run_cli(["--json", "pot", "list"])

    captured = capsys.readouterr()
    assert exc_info.value.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["code"] == "unexpected_cli_error"
    assert payload["detail"] is None
    assert "Traceback" not in captured.out


def test_verbose_keeps_the_traceback_on_an_unhandled_exception(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch.object(host_cli, "app", side_effect=RuntimeError("boom")):
        with pytest.raises(typer.Exit):
            host_cli.run_cli(["--json", "--verbose", "pot", "list"])

    payload = json.loads(capsys.readouterr().out)
    assert "RuntimeError" in payload["detail"]
    assert "boom" in payload["detail"]


# --- --version ---------------------------------------------------------------


@pytest.mark.parametrize(
    "argv", [["--json", "--version"], ["--version", "--json"]], ids=["before", "after"]
)
def test_json_version_emits_json(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    import platform
    import sys

    host_cli.run_cli(list(argv))

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    # The distribution that owns the command, not the library under it. The
    # engine is still reported, under its own key, and the build rev is what
    # actually identifies the code -- both version numbers are constants.
    assert payload["name"] == "potpie"
    assert payload["version"]
    assert set(payload["build"]) == {"rev", "dirty", "built_at"}
    assert payload["engine"]["name"] == "potpie-context-engine"
    assert payload["engine"]["version"]
    assert payload["python"] == platform.python_version()
    assert payload["executable"] == sys.executable


def test_human_version_is_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    import platform
    import sys

    host_cli.run_cli(["--version"])

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("potpie ")  # the distribution, with its build rev
    assert "(" in lines[0] and lines[0].endswith(")")
    assert "potpie-context-engine " in out
    assert f"python {platform.python_version()} ({sys.executable})" in out


# --- a missing source is not a missing pot ----------------------------------


def test_the_boundary_reports_a_missing_source_as_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``SourceNotFound`` subclasses ``PotNotFound``, so order decides the noun.

    The subclassing exists so a boundary that has not learned the type still
    degrades to a sensible refusal rather than "unexpected internal error".
    Without a branch of its own here, every command that has not mapped the
    error itself sends the operator to ``pot list`` to hunt for a pot that was
    never missing.
    """
    from potpie_context_core.errors import SourceNotFound

    _common.set_json(True)
    with pytest.raises(typer.Exit) as exc_info:
        with _common.contract():
            raise SourceNotFound(
                "No source 'src-typo' in pot 'p1'.",
                recommended_next_action="list them with 'potpie source list'",
            )

    assert exc_info.value.exit_code == _common.EXIT_VALIDATION
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "source_not_found"
    assert payload["recommended_next_action"] == "list them with 'potpie source list'"


def test_the_boundary_still_reports_a_missing_pot_as_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from potpie_context_core.errors import PotNotFound

    _common.set_json(True)
    with pytest.raises(typer.Exit):
        with _common.contract():
            raise PotNotFound("No pot matching 'nope'.")

    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "pot_not_found"
    assert "pot list" in payload["recommended_next_action"]
