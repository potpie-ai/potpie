"""CLI install diagnostics for uv-tool installs."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from potpie.cli import cli_install_status as cis
from potpie.cli.commands import bootstrap
from potpie.cli import main as cli_main

pytestmark = pytest.mark.unit

runner = CliRunner()


def test_collect_cli_install_status_from_uv_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cis.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None
    )
    monkeypatch.setattr(
        cis,
        "_potpie_paths_on_path",
        lambda: ["/Users/me/.local/bin/potpie"],
    )
    monkeypatch.setattr(
        cis,
        "_python_from_script",
        lambda _path: "/Users/me/.local/share/uv/tools/potpie-context-engine/bin/python3",
    )
    monkeypatch.setattr(cis, "_python_version", lambda _path: "3.12.12")
    monkeypatch.setattr(cis, "_installed_package_version", lambda: "0.1.0")
    monkeypatch.setattr(
        cis.subprocess,
        "run",
        lambda *args, **kwargs: MagicMock(
            returncode=0,
            stdout="potpie-context-engine v0.1.0\n- potpie\n",
            stderr="",
        ),
    )

    status = cis.collect_cli_install_status()

    assert status["on_path"] is True
    assert status["primary_path"] == "/Users/me/.local/bin/potpie"
    assert status["uv_tool_installed"] is True
    assert status["uv_tool_version"] == "0.1.0"
    assert status["install_method"] == "uv_tool"
    assert status["python_version"] == "3.12.12"
    assert "uv tool list" in status["diagnostic_commands"]
    assert "make cli-status" in status["diagnostic_commands"]
    assert "pip show" in status["pip_show_note"]


def test_cli_install_human_when_missing_from_path() -> None:
    human = cis.cli_install_human({"on_path": False})
    assert "NOT on PATH" in human
    assert "make cli-install" in human


def test_python_from_script_ignores_binary_executable(tmp_path) -> None:
    binary = tmp_path / "potpie"
    binary.write_bytes(b"\xff\xfe\xfd\xfc")
    assert cis._python_from_script(str(binary)) is None


def test_doctor_includes_cli_install(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_host = MagicMock()
    mock_host.daemon.status.return_value = {"mode": "in_process", "up": True}
    mock_host.backend.profile = "falkordb"
    mock_host.backend.capabilities.return_value.implemented.return_value = [
        "graph.read"
    ]
    mock_host.backend.mutation.readiness.return_value = MagicMock(
        ready=True,
        profile="falkordb",
        capability_ready={"mutation": True},
        detail=None,
    )
    mock_host.pots.active_pot.return_value = None
    mock_host.ledger.status.return_value = MagicMock(available=True, binding="local")
    monkeypatch.setattr(bootstrap, "get_host", lambda: mock_host)
    monkeypatch.setattr(
        bootstrap,
        "collect_cli_install_status",
        lambda: {
            "package_name": "potpie-context-engine",
            "package_version": "0.1.0",
            "on_path": True,
            "primary_path": "/Users/me/.local/bin/potpie",
            "python_version": "3.12.12",
            "install_method": "uv_tool",
        },
    )

    result = runner.invoke(cli_main.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["cli_install"]["install_method"] == "uv_tool"
    assert payload["cli_install"]["primary_path"] == "/Users/me/.local/bin/potpie"

    result = runner.invoke(cli_main.app, ["doctor"])
    assert result.exit_code == 0, result.stdout
    assert "cli: potpie-context-engine 0.1.0" in result.stdout
    assert "via=uv_tool" in result.stdout
