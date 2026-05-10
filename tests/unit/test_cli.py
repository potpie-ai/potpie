from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli.local_dev import app


pytestmark = pytest.mark.unit

runner = CliRunner()


def test_parse_submits_repo_and_polls_until_ready(monkeypatch, tmp_path: Path):
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if method == "POST" and path == "/api/v1/parse":
            assert kwargs["json_body"] == {
                "repo_path": str(tmp_path.resolve()),
                "branch_name": "main",
            }
            return {"project_id": "project-1", "status": " submitted "}
        if method == "GET" and path == "/api/v1/parsing-status/project-1":
            return {"status": "READY", "latest": True}
        raise AssertionError(f"Unexpected request {method} {path}")

    monkeypatch.setattr("adapters.inbound.cli.local_dev._request", fake_request)
    monkeypatch.setattr("adapters.inbound.cli.local_dev.time.sleep", lambda _: None)

    result = runner.invoke(
        app,
        [
            "parse",
            str(tmp_path),
            "--branch",
            "main",
            "--api-key",
            "test-key",
            "--interval",
            "0.1",
            "--timeout",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Project: project-1" in result.output
    assert "Parsing complete" in result.output
    assert len(calls) == 2


def test_parse_rejects_file_path(tmp_path: Path):
    file_path = tmp_path / "README.md"
    file_path.write_text("not a repo")

    result = runner.invoke(app, ["parse", str(file_path), "--api-key", "test-key"])

    assert result.exit_code != 0
    assert "Directory" in result.output or "Invalid" in result.output


def test_chat_creates_conversation_and_sends_single_message(monkeypatch):
    requests = []

    def fake_request(method, path, **kwargs):
        requests.append((method, path, kwargs.get("json_body")))
        if method == "GET" and path == "/api/v1/parsing-status/project-1":
            return {"status": " READY ", "latest": True}
        if method == "POST" and path == "/api/v1/conversations/":
            assert kwargs["json_body"] == {
                "project_ids": ["project-1"],
                "agent_ids": ["codebase_qna_agent"],
            }
            return {"conversation_id": "conversation-1"}
        if method == "POST" and path == "/api/v1/conversations/conversation-1/message/":
            assert kwargs["json_body"] == {"content": "What does this repo do?"}
            return {"message": "It answers questions about code."}
        raise AssertionError(f"Unexpected request {method} {path}")

    monkeypatch.setattr("adapters.inbound.cli.local_dev._request", fake_request)

    result = runner.invoke(
        app,
        [
            "chat",
            "project-1",
            "--agent",
            "codebase_qna_agent",
            "--message",
            "What does this repo do?",
            "--api-key",
            "test-key",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Conversation: conversation-1" in result.output
    assert "It answers questions about code." in result.output
    assert [request[0] for request in requests] == ["GET", "POST", "POST"]


def test_chat_rejects_project_that_is_not_ready(monkeypatch):
    def fake_request(method, path, **kwargs):
        assert method == "GET"
        assert path == "/api/v1/parsing-status/project-1"
        return {"status": "submitted", "latest": False}

    monkeypatch.setattr("adapters.inbound.cli.local_dev._request", fake_request)

    result = runner.invoke(
        app,
        [
            "chat",
            "project-1",
            "--agent",
            "codebase_qna_agent",
            "--message",
            "Hello",
            "--api-key",
            "test-key",
        ],
    )

    assert result.exit_code == 1
    assert "not ready yet" in result.output


def test_start_uses_start_script(monkeypatch):
    called = {}

    monkeypatch.setattr(
        "adapters.inbound.cli.local_dev.platform.system", lambda: "Linux"
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.local_dev.shutil.which", lambda name: "/bin/bash"
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.local_dev._repo_root",
        lambda: Path("C:/Users/frogc/potpie-bounty-work"),
    )

    def fake_call(command, cwd):
        called["command"] = command
        called["cwd"] = cwd
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)

    result = runner.invoke(app, ["start"])

    assert result.exit_code == 0
    assert called["command"][0] == "/bin/bash"
    script_path = Path(called["command"][1])
    assert script_path.parts[-2:] == ("scripts", "start.sh")
    assert called["cwd"].name == "potpie-bounty-work"


@pytest.mark.parametrize(
    ("system_name", "expected_engine", "expected_script"),
    [
        ("Linux", "/bin/bash", "stop.sh"),
        ("Windows", "powershell", "stop.ps1"),
    ],
)
def test_stop_uses_stop_script(
    monkeypatch, tmp_path: Path, system_name, expected_engine, expected_script
):
    called = {}
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "stop.sh").write_text("")
    (scripts_dir / "stop.ps1").write_text("")

    monkeypatch.setattr(
        "adapters.inbound.cli.local_dev.platform.system", lambda: system_name
    )
    monkeypatch.setattr(
        "adapters.inbound.cli.local_dev.shutil.which", lambda name: "/bin/bash"
    )
    monkeypatch.setattr("adapters.inbound.cli.local_dev._repo_root", lambda: tmp_path)

    def fake_call(command, cwd):
        called["command"] = command
        called["cwd"] = cwd
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)

    result = runner.invoke(app, ["stop"])

    assert result.exit_code == 0
    assert called["command"][0] == expected_engine
    script_path = Path(called["command"][-1])
    assert script_path.parts[-2:] == ("scripts", expected_script)
    assert called["cwd"] == tmp_path


def test_context_cli_registers_local_dev_commands():
    from adapters.inbound.cli.main import app as context_app

    command_names = {command.name for command in context_app.registered_commands}

    assert {"start", "stop", "parse", "chat"}.issubset(command_names)
