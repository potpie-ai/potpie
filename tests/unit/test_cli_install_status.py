"""CLI install diagnostics for uv-tool installs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from potpie.cli import cli_install_status as cis
from potpie.cli.commands import bootstrap
from potpie.cli import main as cli_main

pytestmark = pytest.mark.unit

runner = CliRunner()


def _fake_uv_tool_env(
    tmp_path: Path,
    *,
    tool_name: str = "potpie",
    editable: bool = True,
) -> tuple[Path, Path]:
    """Create ``…/uv/tools/<tool>/bin/potpie`` (+ optional editable markers)."""
    tool_root = tmp_path / "uv" / "tools" / tool_name
    bin_dir = tool_root / "bin"
    bin_dir.mkdir(parents=True)
    script = bin_dir / "potpie"
    python = bin_dir / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    script.write_text(f"#!{python}\n", encoding="utf-8")
    script.chmod(0o755)
    python.chmod(0o755)

    if editable:
        (tool_root / "uv-receipt.toml").write_text(
            'requirements = [{ name = "potpie", editable = "/repo/potpie" }]\n',
            encoding="utf-8",
        )
        dist = (
            tool_root
            / "lib"
            / "python3.12"
            / "site-packages"
            / "potpie-2.0.0.dist-info"
        )
        dist.mkdir(parents=True)
        (dist / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": "file:///repo/potpie",
                    "dir_info": {"editable": True},
                }
            ),
            encoding="utf-8",
        )
    else:
        (tool_root / "uv-receipt.toml").write_text(
            'requirements = [{ name = "potpie" }]\n',
            encoding="utf-8",
        )
    return tool_root, script


def test_collect_cli_install_status_from_uv_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _tool_root, script = _fake_uv_tool_env(tmp_path, editable=True)
    link = tmp_path / "bin" / "potpie"
    link.parent.mkdir(parents=True)
    link.symlink_to(script)

    monkeypatch.setattr(
        cis.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None
    )
    monkeypatch.setattr(cis, "_potpie_paths_on_path", lambda: [str(link)])
    monkeypatch.setattr(cis, "_python_version", lambda _path: "3.12.12")
    monkeypatch.setattr(cis, "_package_version_via_interpreter", lambda _path: "0.1.0")
    monkeypatch.setattr(cis, "_installed_package_version", lambda: "0.1.0")
    monkeypatch.setattr(
        cis.subprocess,
        "run",
        lambda *args, **kwargs: MagicMock(
            returncode=0,
            stdout="potpie v0.1.0\n- potpie\n",
            stderr="",
        ),
    )

    status = cis.collect_cli_install_status()

    assert status["on_path"] is True
    assert status["primary_path"] == str(link)
    assert status["uv_tool_installed"] is True
    assert status["install_method"] == "uv_tool"
    assert status["editable"] is True
    assert status["uv_tool_name"] == "potpie"
    assert status["python_version"] == "3.12.12"
    assert "uv tool list" in status["diagnostic_commands"]
    assert "make cli-status" in status["diagnostic_commands"]
    assert "make cli-install" in status["diagnostic_commands"]
    assert status["hint"] is not None
    assert "make cli-install" in status["hint"]
    assert "pip show" in status["pip_show_note"]
    assert "make cli-install" in cis.cli_install_human(status)
    assert "editable=true" in cis.cli_install_human(status)


def test_collect_skips_uv_tool_when_path_executable_is_not_uv_backed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """uv tool list can show potpie while PATH points at a different binary."""
    other = tmp_path / "elsewhere" / "bin" / "potpie"
    other.parent.mkdir(parents=True)
    other.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    other.chmod(0o755)

    monkeypatch.setattr(
        cis.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None
    )
    monkeypatch.setattr(cis, "_potpie_paths_on_path", lambda: [str(other)])
    monkeypatch.setattr(cis, "_python_version", lambda _path: "3.12.12")
    monkeypatch.setattr(cis, "_package_version_via_interpreter", lambda _path: "9.9.9")
    monkeypatch.setattr(cis, "_installed_package_version", lambda: "9.9.9")
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

    assert status["uv_tool_installed"] is True  # listed, but not active
    assert status["install_method"] is None
    assert status["editable"] is None
    assert status["hint"] is None
    human = cis.cli_install_human(status)
    assert "via=uv_tool" not in human
    assert "local reinstall: make cli-install" not in human
    assert "published: uv tool install potpie" not in human


def test_sibling_editable_dependency_does_not_mark_tool_editable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Editable sibling deps must not make the potpie tool look editable."""
    tool_root, script = _fake_uv_tool_env(tmp_path, editable=False)
    (tool_root / "uv-receipt.toml").write_text(
        "\n".join(
            [
                "requirements = [",
                '  { name = "potpie" },',
                '  { name = "helper-lib", editable = "/repo/helper" },',
                "]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    site = tool_root / "lib" / "python3.12" / "site-packages"
    sibling = site / "helper_lib-1.0.0.dist-info"
    sibling.mkdir(parents=True)
    (sibling / "direct_url.json").write_text(
        json.dumps(
            {
                "url": "file:///repo/helper",
                "dir_info": {"editable": True},
            }
        ),
        encoding="utf-8",
    )
    # Non-editable potpie dist-info (no editable flag).
    potpie_dist = site / "potpie-2.0.0.dist-info"
    potpie_dist.mkdir(parents=True)
    (potpie_dist / "direct_url.json").write_text(
        json.dumps({"url": "https://pypi.org/simple/potpie/"}),
        encoding="utf-8",
    )

    link = tmp_path / "bin" / "potpie"
    link.parent.mkdir(parents=True)
    link.symlink_to(script)

    monkeypatch.setattr(
        cis.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None
    )
    monkeypatch.setattr(cis, "_potpie_paths_on_path", lambda: [str(link)])
    monkeypatch.setattr(cis, "_python_version", lambda _path: "3.12.12")
    monkeypatch.setattr(cis, "_package_version_via_interpreter", lambda _path: "2.0.0")
    monkeypatch.setattr(
        cis.subprocess,
        "run",
        lambda *args, **kwargs: MagicMock(
            returncode=0,
            stdout="potpie v2.0.0\n- potpie\n",
            stderr="",
        ),
    )

    status = cis.collect_cli_install_status()

    assert status["install_method"] == "uv_tool"
    assert status["editable"] is False
    assert status["hint"] is not None
    assert "make cli-install" not in status["hint"]
    assert "uv tool install potpie" in status["hint"]


def test_published_uv_tool_hint_omits_make_cli_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _tool_root, script = _fake_uv_tool_env(tmp_path, editable=False)
    link = tmp_path / "bin" / "potpie"
    link.parent.mkdir(parents=True)
    link.symlink_to(script)

    monkeypatch.setattr(
        cis.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None
    )
    monkeypatch.setattr(cis, "_potpie_paths_on_path", lambda: [str(link)])
    monkeypatch.setattr(cis, "_python_version", lambda _path: "3.12.12")
    monkeypatch.setattr(cis, "_package_version_via_interpreter", lambda _path: "2.0.0")
    monkeypatch.setattr(
        cis.subprocess,
        "run",
        lambda *args, **kwargs: MagicMock(
            returncode=0,
            stdout="potpie v2.0.0\n- potpie\n",
            stderr="",
        ),
    )

    status = cis.collect_cli_install_status()

    assert status["install_method"] == "uv_tool"
    assert status["editable"] is False
    assert status["hint"] is not None
    assert "uv tool install potpie" in status["hint"]
    assert "make cli-install" not in status["hint"]
    human = cis.cli_install_human(status)
    assert "via=uv_tool" in human
    assert "published: uv tool install potpie" in human
    assert "local reinstall: make cli-install" not in human


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
            "editable": True,
        },
    )

    result = runner.invoke(cli_main.app, ["--json", "doctor"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["cli_install"]["install_method"] == "uv_tool"
    assert payload["cli_install"]["primary_path"] == "/Users/me/.local/bin/potpie"

    result = runner.invoke(cli_main.app, ["doctor"])
    assert result.exit_code == 0, result.stdout
    human = " ".join(result.stdout.split())
    assert "cli: potpie-context-engine 0.1.0" in human
    assert "via=uv_tool" in human
    assert "make cli-status" in human
    assert "make cli-install" in human


def test_cli_install_human_uv_tool_includes_reinstall_hint() -> None:
    human = cis.cli_install_human(
        {
            "on_path": True,
            "package_name": "potpie-context-engine",
            "package_version": "0.1.0",
            "primary_path": "/Users/me/.local/bin/potpie",
            "install_method": "uv_tool",
            "editable": True,
            "python_version": "3.12.12",
        }
    )
    assert "via=uv_tool" in human
    assert "make cli-status" in human
    assert "make cli-install" in human


def test_collect_cli_install_status_omits_hint_without_uv_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cis.shutil, "which", lambda _name: None)
    monkeypatch.setattr(cis, "_potpie_paths_on_path", lambda: [])
    monkeypatch.setattr(cis, "_installed_package_version", lambda: None)

    status = cis.collect_cli_install_status()

    assert status["uv_tool_installed"] is False
    assert status["hint"] is None
    assert "make cli-install" in status["diagnostic_commands"]
