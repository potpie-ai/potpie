"""Local Potpie dev commands: start/stop/parse/chat."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from adapters.inbound.cli import main as cli_main

pytestmark = pytest.mark.unit

runner = CliRunner()


class _FakeLocalClient:
    def __init__(self) -> None:
        self.parse_calls: list[dict[str, Any]] = []
        self.status_calls: list[str] = []
        self.project_rows: list[dict[str, Any]] = [
            {
                "id": "proj-1",
                "repo_name": "repo",
                "branch_name": "main",
                "status": "ready",
            }
        ]
        self.agent_rows: list[dict[str, Any]] = [
            {"id": "codebase_qna_agent", "name": "Codebase Q&A Agent"}
        ]
        self.created_conversations: list[dict[str, Any]] = []
        self.messages: list[tuple[str, str]] = []
        self.status_payloads: list[dict[str, Any]] = [
            {"status": "submitted", "latest": False},
            {"status": "ready", "latest": True},
        ]

    def get_health(self) -> tuple[int, dict[str, Any] | None]:
        return 200, {"status": "ok"}

    def parse_directory(
        self,
        *,
        repo_path: str,
        branch_name: str | None = None,
        repo_name: str | None = None,
    ) -> dict[str, Any]:
        self.parse_calls.append(
            {
                "repo_path": repo_path,
                "branch_name": branch_name,
                "repo_name": repo_name,
            }
        )
        return {"project_id": "proj-1", "status": "submitted"}

    def get_parsing_status(self, project_id: str) -> dict[str, Any]:
        self.status_calls.append(project_id)
        if self.status_payloads:
            return self.status_payloads.pop(0)
        return {"status": "ready", "latest": True}

    def list_projects(self) -> list[dict[str, Any]]:
        return list(self.project_rows)

    def list_available_agents(self) -> list[dict[str, Any]]:
        return list(self.agent_rows)

    def create_conversation(
        self,
        *,
        project_id: str,
        agent_id: str,
        title: str = "CLI Chat",
        hidden: bool = True,
        user_id: str = "cli",
    ) -> dict[str, Any]:
        self.created_conversations.append(
            {
                "project_id": project_id,
                "agent_id": agent_id,
                "title": title,
                "hidden": hidden,
                "user_id": user_id,
            }
        )
        return {"conversation_id": "conv-1"}

    def send_message(self, conversation_id: str, content: str) -> dict[str, Any]:
        self.messages.append((conversation_id, content))
        return {"message": "hello from potpie"}


@pytest.fixture
def fake_local_client(monkeypatch: pytest.MonkeyPatch) -> _FakeLocalClient:
    client = _FakeLocalClient()
    monkeypatch.setattr(cli_main, "_local_client_or_exit", lambda _verbose: client)
    return client


def test_start_launches_background_process(
    fake_local_client: _FakeLocalClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(cli_main, "_local_repo_root_or_exit", lambda _v: tmp_path)
    monkeypatch.setattr(
        cli_main, "_start_command", lambda root: ["bash", str(root / "scripts" / "start.sh")]
    )
    monkeypatch.setattr(
        cli_main,
        "_launch_background_process_or_exit",
        lambda command, *, cwd, log_path, verbose: calls.update(
            {
                "command": command,
                "cwd": cwd,
                "log_path": log_path,
                "verbose": verbose,
            }
        )
        or 4242,
    )
    monkeypatch.setattr(cli_main, "_wait_for_local_health", lambda client, *, timeout_seconds: True)

    result = runner.invoke(cli_main.app, ["start"])
    assert result.exit_code == 0, result.stdout
    assert calls["cwd"] == tmp_path
    assert calls["command"][0] == "bash"
    assert "Potpie start requested." in result.stdout


def test_stop_runs_platform_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr(cli_main, "_local_repo_root_or_exit", lambda _v: tmp_path)
    monkeypatch.setattr(
        cli_main, "_stop_command", lambda root: ["bash", str(root / "scripts" / "stop.sh")]
    )
    monkeypatch.setattr(
        cli_main,
        "_run_cli_subprocess_or_exit",
        lambda command, *, cwd, verbose: calls.update(
            {"command": command, "cwd": cwd, "verbose": verbose}
        ),
    )

    result = runner.invoke(cli_main.app, ["stop"])
    assert result.exit_code == 0, result.stdout
    assert calls["cwd"] == tmp_path
    assert calls["command"][0] == "bash"


def test_parse_polls_until_ready(
    fake_local_client: _FakeLocalClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_main, "_resolve_repo_path_or_exit", lambda raw, verbose: Path("/tmp/repo")
    )
    monkeypatch.setattr(
        cli_main, "_resolve_branch_or_exit", lambda repo_path, branch, verbose: "main"
    )

    result = runner.invoke(
        cli_main.app,
        ["parse", "/tmp/repo", "--poll-interval", "0.1", "--timeout", "5"],
    )
    assert result.exit_code == 0, result.stdout
    assert fake_local_client.parse_calls == [
        {
            "repo_path": str(Path("/tmp/repo")),
            "branch_name": "main",
            "repo_name": "repo",
        }
    ]
    assert fake_local_client.status_calls == ["proj-1", "proj-1"]
    assert "Parse ready: proj-1" in result.stdout


def test_chat_creates_conversation_and_sends_messages(
    fake_local_client: _FakeLocalClient,
) -> None:
    fake_local_client.status_payloads = [{"status": "ready", "latest": True}]

    result = runner.invoke(
        cli_main.app,
        ["chat", "proj-1", "--agent", "Codebase Q&A Agent"],
        input="hello\n/exit\n",
    )
    assert result.exit_code == 0, result.stdout
    assert fake_local_client.created_conversations == [
        {
            "project_id": "proj-1",
            "agent_id": "codebase_qna_agent",
            "title": "CLI Chat proj-1",
            "hidden": True,
            "user_id": "cli",
        }
    ]
    assert fake_local_client.messages == [("conv-1", "hello")]
    assert "potpie> hello from potpie" in result.stdout
