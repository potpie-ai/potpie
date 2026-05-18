"""Unit tests for the local Potpie CLI."""

from __future__ import annotations

import subprocess
import shutil
from importlib.metadata import entry_points
from pathlib import Path

import pytest
from typer.testing import CliRunner

from potpie import cli

pytestmark = pytest.mark.unit

runner = CliRunner()


def _make_git_repo(path: Path) -> Path:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is required for CLI repository tests")

    path.mkdir()
    subprocess.run([git, "init", "-b", "main"], cwd=path, check=True)
    (path / "README.md").write_text("test\n")
    subprocess.run([git, "add", "README.md"], cwd=path, check=True)
    subprocess.run(
        [
            git,
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test User",
            "commit",
            "-m",
            "init",
        ],
        cwd=path,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return path


def test_cli_command_registration() -> None:
    scripts = entry_points(group="console_scripts")
    assert any(
        script.name == "potpie" and script.value == "potpie.cli:main"
        for script in scripts
    )

    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "start" in result.output
    assert "stop" in result.output
    assert "parse" in result.output
    assert "chat" in result.output


def test_parse_invalid_repo_path() -> None:
    result = runner.invoke(cli.app, ["parse", "/does/not/exist"])

    assert result.exit_code == 1
    assert "Repository path does not exist" in result.output


def test_parse_calls_submission_and_polls_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _make_git_repo(tmp_path / "repo")
    calls: list[tuple[str, object]] = []

    class FakeClient:
        def __init__(self, base_url: str | None = None):
            calls.append(("init", base_url))

        def submit_parse(self, repo_path: Path, branch: str) -> dict[str, str]:
            calls.append(("submit", (repo_path, branch)))
            return {"project_id": "project-1", "status": "submitted"}

        def get_parsing_status(self, project_id: str) -> dict[str, str]:
            calls.append(("status", project_id))
            return {"status": "ready", "latest": False}

    monkeypatch.setattr(cli, "PotpieApiClient", FakeClient)

    result = runner.invoke(
        cli.app,
        ["parse", str(repo), "--poll-interval", "0", "--api-url", "http://test"],
    )

    assert result.exit_code == 0
    assert ("init", "http://test") in calls
    assert ("submit", (repo.resolve(), "main")) in calls
    assert ("status", "project-1") in calls
    assert "Parsing complete" in result.output


def test_parse_failure_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = _make_git_repo(tmp_path / "repo")

    class FakeClient:
        def __init__(self, base_url: str | None = None):
            pass

        def submit_parse(self, repo_path: Path, branch: str) -> dict[str, str]:
            return {"project_id": "project-1", "status": "submitted"}

        def get_parsing_status(self, project_id: str) -> dict[str, str]:
            return {"status": "error"}

    monkeypatch.setattr(cli, "PotpieApiClient", FakeClient)

    result = runner.invoke(cli.app, ["parse", str(repo), "--poll-interval", "0"])

    assert result.exit_code == 1
    assert "Parsing failed" in result.output


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda client: client.submit_parse(Path("/tmp/repo"), "main"),
            "POST /api/v2/parse response was not an object.",
        ),
        (
            lambda client: client.get_parsing_status("project-1"),
            "GET /api/v2/parsing-status/project-1 response was not an object.",
        ),
        (
            lambda client: client.create_conversation("project-1", "agent"),
            "POST /api/v2/conversations/ response was not an object.",
        ),
    ],
)
def test_api_object_methods_reject_non_object(
    monkeypatch: pytest.MonkeyPatch, call: object, message: str
) -> None:
    client = cli.PotpieApiClient.__new__(cli.PotpieApiClient)
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: [])

    with pytest.raises(cli.CLIError, match=message):
        call(client)


def test_chat_rejects_not_ready_project(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeClient:
        def __init__(self, base_url: str | None = None):
            pass

        def get_parsing_status(self, project_id: str) -> dict[str, str]:
            return {"status": "submitted"}

        def create_conversation(self, project_id: str, agent_name: str) -> str:
            calls.append("create")
            return "conversation-1"

    monkeypatch.setattr(cli, "PotpieApiClient", FakeClient)

    result = runner.invoke(cli.app, ["chat", "project-1", "--agent", "agent"])

    assert result.exit_code == 1
    assert "not ready" in result.output
    assert calls == []


def test_chat_branch_rejects_missing_project(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeClient:
        def __init__(self, base_url: str | None = None):
            pass

        def get_parsing_status(self, project_id: str) -> dict[str, str]:
            return {"status": "ready"}

        def list_projects(self) -> list[dict[str, str]]:
            return []

        def create_conversation(self, project_id: str, agent_name: str) -> str:
            calls.append("create")
            return "conversation-1"

    monkeypatch.setattr(cli, "PotpieApiClient", FakeClient)

    result = runner.invoke(
        cli.app,
        ["chat", "project-1", "--agent", "agent", "--branch", "main"],
    )

    assert result.exit_code == 1
    assert "was not found in the project list" in result.output
    assert calls == []


def test_chat_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    sent_messages: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, base_url: str | None = None):
            pass

        def get_parsing_status(self, project_id: str) -> dict[str, str]:
            return {"status": "ready"}

        def create_conversation(self, project_id: str, agent_name: str) -> str:
            return "conversation-1"

        def send_message(self, conversation_id: str, content: str) -> dict[str, str]:
            sent_messages.append((conversation_id, content))
            return {"message": "hello from potpie"}

    monkeypatch.setattr(cli, "PotpieApiClient", FakeClient)

    result = runner.invoke(
        cli.app,
        ["chat", "project-1", "--agent", "codebase_qna_agent"],
        input="hello\nquit\n",
    )

    assert result.exit_code == 0
    assert sent_messages == [("conversation-1", "hello")]
    assert "hello from potpie" in result.output
    assert "Exiting chat" in result.output
