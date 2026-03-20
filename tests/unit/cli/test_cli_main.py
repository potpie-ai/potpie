"""Unit tests for potpie.cli.main (argument parsing and dispatch)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from potpie.cli.main import _build_parser, main

pytestmark = pytest.mark.unit


class TestBuildParser:
    def test_parser_has_all_commands(self):
        parser = _build_parser()
        # argparse stores subparsers in _subparsers
        subparser_actions = [
            a for a in parser._subparsers._group_actions
            if hasattr(a, "choices")
        ]
        assert subparser_actions, "No subparsers found"
        commands = set(subparser_actions[0].choices.keys())
        assert "start" in commands
        assert "stop" in commands
        assert "parse" in commands
        assert "chat" in commands
        assert "list-projects" in commands
        assert "list-agents" in commands

    def test_parse_requires_repo_path(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["parse"])

    def test_chat_requires_project_id_and_agent(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["chat"])
        with pytest.raises(SystemExit):
            parser.parse_args(["chat", "proj-1"])  # missing --agent

    def test_chat_parses_correctly(self):
        parser = _build_parser()
        args = parser.parse_args(["chat", "proj-1", "--agent", "codebase_qna_agent"])
        assert args.project_id == "proj-1"
        assert args.agent == "codebase_qna_agent"

    def test_parse_default_branch(self):
        parser = _build_parser()
        args = parser.parse_args(["parse", "/some/path"])
        assert args.branch == "main"

    def test_parse_custom_branch(self):
        parser = _build_parser()
        args = parser.parse_args(["parse", "/some/path", "--branch", "develop"])
        assert args.branch == "develop"

    def test_url_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--url", "http://localhost:9000", "list-agents"])  # NOSONAR — localhost test fixture
        assert args.url == "http://localhost:9000"  # NOSONAR

    def test_no_command_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1


class TestMainDispatch:
    def test_dispatches_start(self):
        with patch("potpie.cli.commands.start.start_server") as mock_fn:
            main(["start"])
        mock_fn.assert_called_once()

    def test_dispatches_stop(self):
        with patch("potpie.cli.commands.stop.stop_server") as mock_fn:
            main(["stop"])
        mock_fn.assert_called_once()

    def test_dispatches_parse(self, tmp_path):
        with patch("potpie.cli.commands.parse.parse_repo") as mock_fn:
            main(["parse", str(tmp_path)])
        mock_fn.assert_called_once_with(
            str(tmp_path), branch="main", base_url=None
        )

    def test_dispatches_parse_with_branch(self, tmp_path):
        with patch("potpie.cli.commands.parse.parse_repo") as mock_fn:
            main(["parse", str(tmp_path), "--branch", "dev"])
        mock_fn.assert_called_once_with(
            str(tmp_path), branch="dev", base_url=None
        )

    def test_dispatches_chat(self):
        with patch("potpie.cli.commands.chat.start_chat") as mock_fn:
            main(["chat", "proj-1", "--agent", "debug_agent"])
        mock_fn.assert_called_once_with(
            "proj-1", agent_id="debug_agent", base_url=None
        )

    def test_dispatches_list_projects(self):
        with patch("potpie.cli.commands.list_projects.list_projects") as mock_fn:
            main(["list-projects"])
        mock_fn.assert_called_once_with(base_url=None)

    def test_dispatches_list_agents(self):
        with patch("potpie.cli.commands.list_agents.list_agents") as mock_fn:
            main(["list-agents"])
        mock_fn.assert_called_once_with(base_url=None)

    def test_url_flag_passed_to_commands(self):
        with patch("potpie.cli.commands.list_agents.list_agents") as mock_fn:
            main(["--url", "http://myserver:9000", "list-agents"])  # NOSONAR — test fixture URL
        mock_fn.assert_called_once_with(base_url="http://myserver:9000")  # NOSONAR

    def test_url_flag_passed_to_parse(self, tmp_path):
        with patch("potpie.cli.commands.parse.parse_repo") as mock_fn:
            main(["--url", "http://myserver:9000", "parse", str(tmp_path)])  # NOSONAR — test fixture URL
        mock_fn.assert_called_once_with(
            str(tmp_path), branch="main", base_url="http://myserver:9000"  # NOSONAR
        )

    def test_url_flag_passed_to_chat(self):
        with patch("potpie.cli.commands.chat.start_chat") as mock_fn:
            main(["--url", "http://myserver:9000", "chat", "proj-1", "--agent", "agent"])  # NOSONAR — test fixture URL
        mock_fn.assert_called_once_with(
            "proj-1", agent_id="agent", base_url="http://myserver:9000"  # NOSONAR
        )

    def test_url_flag_passed_to_list_projects(self):
        with patch("potpie.cli.commands.list_projects.list_projects") as mock_fn:
            main(["--url", "http://myserver:9000", "list-projects"])  # NOSONAR — test fixture URL
        mock_fn.assert_called_once_with(base_url="http://myserver:9000")  # NOSONAR
