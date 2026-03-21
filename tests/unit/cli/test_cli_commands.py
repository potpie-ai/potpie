"""Unit tests for potpie CLI command functions."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from potpie.cli.client import PotpieClientError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client_mock(**kwargs):
    """Return a MagicMock configured as a PotpieClient."""
    mock = MagicMock()
    for k, v in kwargs.items():
        setattr(mock, k, MagicMock(return_value=v))
    return mock


# ---------------------------------------------------------------------------
# parse command
# ---------------------------------------------------------------------------


class TestParseCommand:
    def test_parse_valid_path_ready_immediately(self, tmp_path, capsys):
        from potpie.cli.commands.parse import parse_repo

        mock_client = _make_client_mock()
        mock_client.parse.return_value = {"project_id": "proj-1", "status": "ready"}

        with patch("potpie.cli.commands.parse.PotpieClient", return_value=mock_client):
            parse_repo(str(tmp_path), branch="main")

        out = capsys.readouterr().out
        assert "proj-1" in out
        assert "ready" in out.lower()

    def test_parse_polls_until_ready(self, tmp_path, capsys):
        from potpie.cli.commands.parse import parse_repo

        mock_client = _make_client_mock()
        mock_client.parse.return_value = {"project_id": "proj-2", "status": "submitted"}
        mock_client.poll_parsing_status.return_value = iter([
            {"status": "submitted", "latest": False},
            {"status": "cloned", "latest": False},
            {"status": "ready", "latest": True},
        ])

        with patch("potpie.cli.commands.parse.PotpieClient", return_value=mock_client):
            parse_repo(str(tmp_path), branch="main")

        out = capsys.readouterr().out
        assert "proj-2" in out
        assert "complete" in out.lower()

    def test_parse_exits_on_nonexistent_path(self, capsys):
        from potpie.cli.commands.parse import parse_repo

        with pytest.raises(SystemExit) as exc_info:
            parse_repo("/nonexistent/path/that/does/not/exist")
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "does not exist" in err

    def test_parse_exits_when_path_is_file(self, tmp_path, capsys):
        from potpie.cli.commands.parse import parse_repo

        test_file = tmp_path / "file.txt"
        test_file.write_text("hello")

        with pytest.raises(SystemExit) as exc_info:
            parse_repo(str(test_file))
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "not a directory" in err

    def test_parse_exits_on_client_error(self, tmp_path, capsys):
        from potpie.cli.commands.parse import parse_repo

        mock_client = _make_client_mock()
        mock_client.parse.side_effect = PotpieClientError("server error")

        with patch("potpie.cli.commands.parse.PotpieClient", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                parse_repo(str(tmp_path))
        assert exc_info.value.code == 1

    def test_parse_exits_on_failed_status(self, tmp_path, capsys):
        from potpie.cli.commands.parse import parse_repo

        mock_client = _make_client_mock()
        mock_client.parse.return_value = {"project_id": "proj-3", "status": "submitted"}
        mock_client.poll_parsing_status.return_value = iter([
            {"status": "error", "latest": False},
        ])

        with patch("potpie.cli.commands.parse.PotpieClient", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                parse_repo(str(tmp_path))
        assert exc_info.value.code == 1

    def test_parse_missing_project_id(self, tmp_path, capsys):
        from potpie.cli.commands.parse import parse_repo

        mock_client = _make_client_mock()
        mock_client.parse.return_value = {"status": "submitted"}  # no project_id

        with patch("potpie.cli.commands.parse.PotpieClient", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                parse_repo(str(tmp_path))
        assert exc_info.value.code == 1

    def test_parse_uses_branch_argument(self, tmp_path):
        from potpie.cli.commands.parse import parse_repo

        mock_client = _make_client_mock()
        mock_client.parse.return_value = {"project_id": "proj-4", "status": "ready"}

        with patch("potpie.cli.commands.parse.PotpieClient", return_value=mock_client):
            parse_repo(str(tmp_path), branch="feature/my-branch")

        mock_client.parse.assert_called_once_with(
            repo_path=str(tmp_path.resolve()),
            branch_name="feature/my-branch",
        )


# ---------------------------------------------------------------------------
# list_projects command
# ---------------------------------------------------------------------------


class TestListProjectsCommand:
    def test_list_projects_output(self, capsys):
        from potpie.cli.commands.list_projects import list_projects

        projects = [
            {"id": "proj-1", "repo_name": "owner/repo", "branch_name": "main", "status": "ready"},
            {"id": "proj-2", "repo_name": "owner/repo2", "branch_name": "dev", "status": "submitted"},
        ]
        mock_client = _make_client_mock(list_projects=projects)

        with patch("potpie.cli.commands.list_projects.PotpieClient", return_value=mock_client):
            list_projects()

        out = capsys.readouterr().out
        assert "proj-1" in out
        assert "owner/repo" in out
        assert "ready" in out

    def test_list_projects_empty(self, capsys):
        from potpie.cli.commands.list_projects import list_projects

        mock_client = _make_client_mock(list_projects=[])

        with patch("potpie.cli.commands.list_projects.PotpieClient", return_value=mock_client):
            list_projects()

        out = capsys.readouterr().out
        assert "No projects found" in out

    def test_list_projects_exits_on_client_error(self, capsys):
        from potpie.cli.commands.list_projects import list_projects

        mock_client = MagicMock()
        mock_client.list_projects.side_effect = PotpieClientError("connection refused")

        with patch("potpie.cli.commands.list_projects.PotpieClient", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                list_projects()
        assert exc_info.value.code == 1

    def test_list_projects_shows_repo_path_when_no_repo_name(self, capsys):
        from potpie.cli.commands.list_projects import list_projects

        projects = [
            {"id": "proj-3", "repo_path": "/local/repo", "branch_name": "main", "status": "ready"},
        ]
        mock_client = _make_client_mock(list_projects=projects)

        with patch("potpie.cli.commands.list_projects.PotpieClient", return_value=mock_client):
            list_projects()

        out = capsys.readouterr().out
        assert "/local/repo" in out


# ---------------------------------------------------------------------------
# list_agents command
# ---------------------------------------------------------------------------


class TestListAgentsCommand:
    def test_list_agents_output(self, capsys):
        from potpie.cli.commands.list_agents import list_agents

        agents = [
            {"id": "codebase_qna_agent", "name": "Codebase QnA", "status": "active", "description": "Q&A about the codebase"},
            {"id": "debug_agent", "name": "Debugging", "status": "active", "description": "Helps debug code"},
        ]
        mock_client = _make_client_mock(list_agents=agents)

        with patch("potpie.cli.commands.list_agents.PotpieClient", return_value=mock_client):
            list_agents()

        out = capsys.readouterr().out
        assert "codebase_qna_agent" in out
        assert "Codebase QnA" in out

    def test_list_agents_empty(self, capsys):
        from potpie.cli.commands.list_agents import list_agents

        mock_client = _make_client_mock(list_agents=[])

        with patch("potpie.cli.commands.list_agents.PotpieClient", return_value=mock_client):
            list_agents()

        out = capsys.readouterr().out
        assert "No agents found" in out

    def test_list_agents_exits_on_client_error(self, capsys):
        from potpie.cli.commands.list_agents import list_agents

        mock_client = MagicMock()
        mock_client.list_agents.side_effect = PotpieClientError("connection refused")

        with patch("potpie.cli.commands.list_agents.PotpieClient", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                list_agents()
        assert exc_info.value.code == 1

    def test_list_agents_description_shown(self, capsys):
        from potpie.cli.commands.list_agents import list_agents

        agents = [
            {"id": "agent1", "name": "Agent One", "status": "active",
             "description": "This agent does something useful"},
        ]
        mock_client = _make_client_mock(list_agents=agents)

        with patch("potpie.cli.commands.list_agents.PotpieClient", return_value=mock_client):
            list_agents()

        out = capsys.readouterr().out
        assert "This agent does something useful" in out


# ---------------------------------------------------------------------------
# chat command
# ---------------------------------------------------------------------------


class TestChatCommand:
    def test_chat_creates_conversation_and_sends_messages(self, capsys):
        from potpie.cli.commands.chat import start_chat

        mock_client = MagicMock()
        mock_client.create_conversation.return_value = {
            "conversation_id": "conv-1",
            "message": "ok",
        }
        mock_client.send_message.return_value = {"message": "Here is the answer."}

        inputs = iter(["What is this project?", "exit"])
        with patch("potpie.cli.commands.chat.PotpieClient", return_value=mock_client):
            with patch("builtins.input", side_effect=inputs):
                start_chat("proj-1", "codebase_qna_agent")

        mock_client.create_conversation.assert_called_once_with(
            project_id="proj-1", agent_id="codebase_qna_agent"
        )
        mock_client.send_message.assert_called_once_with(
            "conv-1", "What is this project?"
        )
        out = capsys.readouterr().out
        assert "Here is the answer." in out

    def test_chat_exits_on_empty_project_id(self, capsys):
        from potpie.cli.commands.chat import start_chat

        with pytest.raises(SystemExit) as exc_info:
            start_chat("", "agent-1")
        assert exc_info.value.code == 1

    def test_chat_exits_on_empty_agent_id(self, capsys):
        from potpie.cli.commands.chat import start_chat

        with pytest.raises(SystemExit) as exc_info:
            start_chat("proj-1", "")
        assert exc_info.value.code == 1

    def test_chat_exits_when_conversation_not_created(self, capsys):
        from potpie.cli.commands.chat import start_chat

        mock_client = MagicMock()
        mock_client.create_conversation.side_effect = PotpieClientError("server error")

        with patch("potpie.cli.commands.chat.PotpieClient", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                start_chat("proj-1", "agent-1")
        assert exc_info.value.code == 1

    def test_chat_handles_eoferror_gracefully(self, capsys):
        from potpie.cli.commands.chat import start_chat

        mock_client = MagicMock()
        mock_client.create_conversation.return_value = {
            "conversation_id": "conv-2",
            "message": "ok",
        }

        with patch("potpie.cli.commands.chat.PotpieClient", return_value=mock_client):
            with patch("builtins.input", side_effect=EOFError):
                # Should not raise
                start_chat("proj-1", "agent-1")

        out = capsys.readouterr().out
        assert "Ending" in out

    def test_chat_skips_empty_input(self, capsys):
        from potpie.cli.commands.chat import start_chat

        mock_client = MagicMock()
        mock_client.create_conversation.return_value = {
            "conversation_id": "conv-3",
            "message": "ok",
        }
        mock_client.send_message.return_value = {"message": "reply"}

        inputs = iter(["", "  ", "hello", "quit"])
        with patch("potpie.cli.commands.chat.PotpieClient", return_value=mock_client):
            with patch("builtins.input", side_effect=inputs):
                start_chat("proj-1", "agent-1")

        # Only one non-empty, non-quit message should have been sent
        assert mock_client.send_message.call_count == 1

    def test_chat_send_error_continues_session(self, capsys):
        from potpie.cli.commands.chat import start_chat

        mock_client = MagicMock()
        mock_client.create_conversation.return_value = {
            "conversation_id": "conv-4",
            "message": "ok",
        }
        mock_client.send_message.side_effect = [
            PotpieClientError("timeout"),
            {"message": "success"},
        ]

        inputs = iter(["first message", "second message", "exit"])
        with patch("potpie.cli.commands.chat.PotpieClient", return_value=mock_client):
            with patch("builtins.input", side_effect=inputs):
                start_chat("proj-1", "agent-1")

        # Both messages attempted, session continued after error
        assert mock_client.send_message.call_count == 2

    def test_chat_exits_when_no_conversation_id_returned(self, capsys):
        from potpie.cli.commands.chat import start_chat

        mock_client = MagicMock()
        mock_client.create_conversation.return_value = {"message": "ok"}  # missing conversation_id

        with patch("potpie.cli.commands.chat.PotpieClient", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                start_chat("proj-1", "agent-1")
        assert exc_info.value.code == 1

    def test_chat_quit_variants(self, capsys):
        from potpie.cli.commands.chat import start_chat, _QUIT_COMMANDS

        for quit_cmd in _QUIT_COMMANDS:
            mock_client = MagicMock()
            mock_client.create_conversation.return_value = {
                "conversation_id": "conv-5",
                "message": "ok",
            }
            inputs = iter([quit_cmd])
            with patch("potpie.cli.commands.chat.PotpieClient", return_value=mock_client):
                with patch("builtins.input", side_effect=inputs):
                    # Should exit cleanly without raising
                    start_chat("proj-1", "agent-1")


# ---------------------------------------------------------------------------
# start / stop commands (subprocess-level)
# ---------------------------------------------------------------------------


class TestStartCommand:
    def test_start_runs_script_if_exists(self, tmp_path):
        from potpie.cli.commands.start import start_server

        script = tmp_path / "scripts" / "start.sh"
        script.parent.mkdir()
        script.write_text("#!/bin/bash\necho started\n")

        with patch("potpie.cli.commands.start._find_project_root", return_value=tmp_path):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                start_server()

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert any("bash" in a for a in args)
        assert str(script) in args

    def test_start_falls_back_to_direct_when_no_script(self, tmp_path):
        from potpie.cli.commands.start import start_server

        # No scripts/start.sh in tmp_path
        with patch("potpie.cli.commands.start._find_project_root", return_value=tmp_path):
            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value.pid = 12345
                start_server()

        assert mock_popen.call_count == 2  # gunicorn + celery


class TestStopCommand:
    def test_stop_runs_script_if_exists(self, tmp_path):
        from potpie.cli.commands.stop import stop_server

        script = tmp_path / "scripts" / "stop.sh"
        script.parent.mkdir()
        script.write_text("#!/bin/bash\necho stopped\n")

        with patch("potpie.cli.commands.stop._find_project_root", return_value=tmp_path):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                stop_server()

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert any("bash" in a for a in args)
        assert str(script) in args

    def test_stop_falls_back_to_pkill_when_no_script(self, tmp_path):
        from potpie.cli.commands.stop import stop_server

        with patch("potpie.cli.commands.stop._find_project_root", return_value=tmp_path):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                stop_server()

        # pkill for gunicorn and celery
        calls = [str(c) for c in mock_run.call_args_list]
        joined = " ".join(calls)
        assert "gunicorn" in joined
        assert "celery" in joined
