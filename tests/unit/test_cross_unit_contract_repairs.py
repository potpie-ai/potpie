"""Contracts that only break when two units' changes are combined.

Every case here passed inside the unit that introduced it and failed once the
four P1 branches shared a working tree, or was a hole that no single unit owned
because the fix straddled two of them:

- a global flag typed after the command is hoisted and honoured, and the hint
  left for the one shape that cannot be hoisted reconstructed its advice from
  raw argv, so it printed the secret that ``config set`` redacts one line later;
- ``setup`` moved in-process (cold-start deadline fix) and stopped consulting
  the origin, turning ``--host managed setup`` from a refusal into a silent
  local provision at exit 0;
- the auth surface claimed the shared error envelope while emitting a different
  shape on a different stream;
- three unavailability codes reached exit 2 only because their call sites passed
  the number by hand, so the one that forgot exited 1 for the same condition;
- two copies of the repo normalizer disagreed about credentials in a remote URL,
  which both defeated dedup and wrote the credential to disk;
- ``doctor`` survived a host that *refuses* a surface but not one whose payload
  shape had drifted, which is the same cross-repo skew it exists to report.

These are integration-level invariants: keep them with the combined surface
rather than in any one unit's module.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
import typer
from typer.testing import CliRunner

from potpie.cli import hosts
from potpie.cli import main as host_cli
from potpie.cli.commands import _common
from potpie.cli.ui import output as ui_output


@pytest.fixture(autouse=True)
def _reset_cli_state() -> Iterator[None]:
    _common.set_json(False)
    _common.set_verbose(False)
    _common.clear_argv_output_flags()
    _common.set_host(None)
    hosts.set_current_origin(None)
    yield
    _common.set_json(False)
    _common.set_verbose(False)
    _common.clear_argv_output_flags()
    _common.set_host(None)
    hosts.set_current_origin(None)


def _run_cli(argv: list[str], capsys) -> tuple[int, str, str]:
    """Drive the shipped entrypoint; ``CliRunner`` never enters ``run_cli``."""
    code = 0
    try:
        host_cli.run_cli(argv)
    except typer.Exit as exc:
        code = int(exc.exit_code)
    except SystemExit as exc:  # click.Abort and friends land here
        code = int(exc.code or 0)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --- a trailing global flag, and the hint that survives it -------------------

SECRET = "ghp_thisisasecrettokenvalue"


@pytest.mark.parametrize("flag", ["--json", "--verbose"])
def test_a_trailing_flag_never_echoes_the_value_it_followed(flag, capsys) -> None:
    """The whole point of redacting the echo is lost if the parse layer prints it.

    A trailing global is now hoisted and honoured, so this command line reaches
    the config catalog gate instead of the position refusal — and neither of
    them may repeat the value back. The gate fires *before* the write, so the
    key is named and the secret is not.
    """
    code, out, err = _run_cli(["config", "set", "github_token", SECRET, flag], capsys)

    assert code == _common.exit_code_for("validation_error")
    assert SECRET not in out
    assert SECRET not in err
    assert "github_token" in (out + err), "the refusal must still name the key"


def test_the_unhoistable_hint_still_names_the_command_without_its_values(
    capsys,
) -> None:
    """A value-taking global with no value is the one shape still refused by position.

    It is also the shape that reaches `_reordered_invocation`, whose first
    version echoed every remaining token — turning a parse error into a
    next-action line that printed the secret `config set` redacts one step
    later.
    """
    code, out, err = _run_cli(
        ["config", "set", "github_token", SECRET, "--host"], capsys
    )

    assert code == _common.exit_code_for("usage_error")
    assert SECRET not in out
    assert SECRET not in err
    assert "config set" in (out + err), "the hint must still name the command"


def test_a_trailing_global_flag_runs_the_command_it_was_typed_after(capsys) -> None:
    """`potpie pot list --json` answers with the listing, not with advice about it."""
    code, out, _ = _run_cli(["pot", "list", "--json"], capsys)
    payload = json.loads(out)

    assert code == 0
    assert "pots" in payload


def test_a_global_flag_between_a_group_and_its_subcommand_is_honoured(capsys) -> None:
    """Click stops resolving at the group; the flag still belongs to the root."""
    _, out, _ = _run_cli(["graph", "--json", "catalog"], capsys)
    payload = json.loads(out)

    assert payload["command"] == "graph.catalog"


# --- setup targets this machine, and says so ---------------------------------


def test_setup_refuses_an_explicit_remote_target(monkeypatch, capsys) -> None:
    """An error must not degrade into a wrong-host success.

    Before this, ``--host managed setup`` provisioned a *local* backend, pot and
    daemon and exited 0 without ever contacting the host it was aimed at.
    """
    monkeypatch.setenv("POTPIE_MANAGED_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("POTPIE_MANAGED_TOKEN", "sk-test")

    code, out, _ = _run_cli(["--json", "--host", "managed", "setup"], capsys)
    payload = json.loads(out)

    assert code == _common.exit_code_for("validation_error")
    assert payload["ok"] is False
    assert "cannot target" in payload["message"]


def test_setup_still_runs_for_a_persisted_managed_pointer(monkeypatch) -> None:
    """Only an *explicit* target refuses; the persisted pointer must not strand anyone.

    Someone whose active host is managed still has to be able to provision the
    machine they are typing on.
    """
    monkeypatch.setattr(hosts, "origin_overridden", lambda: False)
    monkeypatch.setattr(hosts, "selected_origin", lambda: hosts.MANAGED)

    from potpie.cli.commands.bootstrap import _refuse_remote_setup_target

    _refuse_remote_setup_target()  # must not raise


# --- one envelope, on one stream ---------------------------------------------


def test_auth_errors_use_the_shared_envelope_on_stdout(capsys) -> None:
    """``potpie --json login`` used to put nothing at all on stdout."""
    ui_output.configure_error_output(as_json=True)
    try:
        ui_output.emit_error("Potpie login failed", "bad key", hint="use sk-")
    finally:
        ui_output.configure_error_output(as_json=False)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert captured.err == ""
    assert set(payload) == {
        "ok",
        "code",
        "message",
        "detail",
        "recommended_next_action",
    }
    assert payload["ok"] is False


# --- one exit-code table ------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    ["daemon_unavailable", "daemon_start_failed", "telemetry_preference_write_failed"],
)
def test_narrow_unavailability_codes_are_in_the_table(code) -> None:
    """`potpie ui` reported a dead daemon as 1 while `potpie status` reported 2."""
    assert _common.exit_code_for(code) == _common.EXIT_UNAVAILABLE


def test_no_fail_call_site_passes_an_exit_code_the_table_disagrees_with() -> None:
    """``exit_code=`` stays an override, never a second table.

    The parameter exists for paths that mean something the code alone does not
    say, which makes it the one way a number could quietly acquire a second
    meaning — the exact drift ``exit_code_for`` was introduced to end. Read off
    the source rather than exercised, because the point is that no such call
    site may exist, not that the ones we happened to test agree.
    """
    import ast
    import pathlib

    cli_root = pathlib.Path(_common.__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in sorted(cli_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name != "fail":
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            code, override = kwargs.get("code"), kwargs.get("exit_code")
            if override is None or not isinstance(code, ast.Constant):
                continue
            if isinstance(override, ast.Name):
                value = getattr(_common, override.id, None)
            elif isinstance(override, ast.Constant):
                value = override.value
            else:
                continue
            if value is not None and value != _common.exit_code_for(code.value):
                offenders.append(f"{path.name}:{node.lineno} {code.value} -> {value}")

    assert not offenders, offenders


def test_no_cli_call_site_hardcodes_an_exit_code_the_table_disagrees_with() -> None:
    """The table is only authoritative if nothing quietly overrides it."""
    import inspect

    from potpie.cli.commands import daemon as daemon_cmds
    from potpie.cli.commands import host as host_cmds
    from potpie.cli.commands import pots as pot_cmds
    from potpie.cli.commands import telemetry as telemetry_cmds
    from potpie.cli.commands import ui as ui_cmds

    for module in (daemon_cmds, telemetry_cmds, host_cmds, pot_cmds, ui_cmds):
        source = inspect.getsource(module)
        assert "exit_code=EXIT_UNAVAILABLE" not in source, module.__name__


# --- one repo normalizer ------------------------------------------------------


def test_the_cli_and_the_engine_share_one_repo_normalizer() -> None:
    from potpie.cli import repo_location
    from potpie_context_engine.domain import repo_identity

    assert repo_location.repo_identity_key is repo_identity.repo_identity_key
    assert repo_location.normalize_repo_ref is repo_identity.normalize_repo_ref


@pytest.mark.parametrize(
    "ref",
    [
        "https://user:pw@github.com/acme/shop",
        "https://token@github.com/acme/shop",
        "https://github.com/Acme/Shop.git",
        "git@github.com:Acme/Shop.git",
    ],
)
def test_repo_identity_drops_credentials_and_still_dedups(ref) -> None:
    """A credential is not part of a repository's identity — and it reached pots.json."""
    from potpie_context_engine.domain.repo_identity import repo_identity_key

    assert repo_identity_key(ref) == "github.com/acme/shop"


def test_repo_identity_keeps_a_non_default_port() -> None:
    """Dropping userinfo must not also drop the part that distinguishes hosts."""
    from potpie_context_engine.domain.repo_identity import repo_identity_key

    assert (
        repo_identity_key("https://user:pw@ghe.internal:8443/acme/shop.git")
        == "ghe.internal:8443/acme/shop"
    )


# --- a credential inside a value ---------------------------------------------


def test_config_redacts_url_userinfo_in_both_directions() -> None:
    """`is_secret_config_key` reads the key, so it cannot see this one."""
    from potpie_context_engine.application.services.config_service import (
        public_config_value,
    )

    assert (
        public_config_value("ledger.url", "https://user:tok@ledger.internal/x")
        == "https://<redacted>@ledger.internal/x"
    )
    # An ordinary URL is untouched, or every operator reads a redaction that
    # withheld nothing.
    assert (
        public_config_value("ledger.url", "https://ledger.internal/x")
        == "https://ledger.internal/x"
    )


# --- doctor survives shape drift, not just refusal ---------------------------


class _DriftedHost:
    """A managed host built from a revision where these payloads were DTOs."""

    profile = "local"

    class backend:
        profile = "falkordb_lite"

        @staticmethod
        def capabilities() -> dict:
            return {"mutation": True}

        class mutation:
            @staticmethod
            def readiness(pot_id: str) -> dict:
                return {"ready": True}

    class pots:
        @staticmethod
        def active_pot():
            class _Pot:
                pot_id = "pot_1"
                name = "default"
                active = True

            return _Pot()

        @staticmethod
        def list_pots() -> list:
            return []

    class resources:
        @staticmethod
        def status(pot_id=None) -> dict:
            return {"kind": "local"}

        @staticmethod
        def index_status(pot_id=None) -> dict:
            return {"profile": "x"}

    class ledger:
        @staticmethod
        def status() -> dict:
            return {"available": True}

    class daemon:
        in_process = True

        @staticmethod
        def status() -> dict:
            return {"up": True, "mode": "in_process", "home": "/x", "pid": 1}

        @staticmethod
        def discovery():
            return None

    class auth:
        @staticmethod
        def whoami():
            raise NotImplementedError


def test_doctor_reports_payload_drift_instead_of_dying_on_it() -> None:
    """A renamed field on a managed host used to cost the operator every block.

    ``section()`` wrapped the call but not the unwrapping, so the report died
    with ``unexpected_cli_error`` at exit 1 — from the one command whose entire
    purpose is to describe a broken host.
    """
    _common.set_host(_DriftedHost())
    _common.set_json(True)

    result = CliRunner().invoke(
        host_cli.app, ["--json", "doctor"], catch_exceptions=False
    )
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload["ok"] is False
    assert "resources" in payload["degraded_sections"]
    # The blocks that never needed a host must survive the ones that did.
    assert payload["daemon"]["pid"] == 1
    assert payload["cli_install"]
    # Degraded blocks keep their whole key set, so consumers index, not guess.
    assert {"kind", "ready", "location", "documents", "detail"} <= set(
        payload["resources"]
    )
