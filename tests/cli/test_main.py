"""Tests for the CLI main module."""

from __future__ import annotations

import sys
from unittest.mock import patch, MagicMock

import pytest

from potpie.cli.main import build_parser, main


class TestParser:
    """Test argument parsing."""

    def test_no_args_shows_help(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_parse_command(self):
        parser = build_parser()
        args = parser.parse_args(["parse", "/tmp/repo"])
        assert args.command == "parse"
        assert args.repo_path == "/tmp/repo"
        assert args.branch == "main"

    def test_parse_with_branch(self):
        parser = build_parser()
        args = parser.parse_args(["parse", "/tmp/repo", "--branch", "develop"])
        assert args.branch == "develop"

    def test_chat_command(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "project-123"])
        assert args.command == "chat"
        assert args.project_id == "project-123"
        assert args.agent == "codebase_qna_agent"

    def test_chat_with_agent(self):
        parser = build_parser()
        args = parser.parse_args(["chat", "project-123", "--agent", "debug_agent"])
        assert args.agent == "debug_agent"

    def test_status_command(self):
        parser = build_parser()
        args = parser.parse_args(["status", "project-456"])
        assert args.command == "status"
        assert args.project_id == "project-456"

    def test_projects_command(self):
        parser = build_parser()
        args = parser.parse_args(["projects"])
        assert args.command == "projects"

    def test_agents_command(self):
        parser = build_parser()
        args = parser.parse_args(["agents"])
        assert args.command == "agents"

    def test_start_command(self):
        parser = build_parser()
        args = parser.parse_args(["start"])
        assert args.command == "start"

    def test_stop_command(self):
        parser = build_parser()
        args = parser.parse_args(["stop"])
        assert args.command == "stop"

    def test_custom_url(self):
        parser = build_parser()
        args = parser.parse_args(["--url", "http://myhost:9000", "projects"])
        assert args.url == "http://myhost:9000"
        assert args.command == "projects"


class TestMainDispatch:
    """Test that main() dispatches to the correct command."""

    def test_no_command_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    @patch("potpie.cli.main.cmd_projects")
    @patch("potpie.cli.main._get_client")
    def test_projects_dispatched(self, mock_client, mock_cmd):
        mock_cmd.return_value = None
        main(["projects"])
        mock_cmd.assert_called_once()

    @patch("potpie.cli.main.cmd_agents")
    @patch("potpie.cli.main._get_client")
    def test_agents_dispatched(self, mock_client, mock_cmd):
        mock_cmd.return_value = None
        main(["agents"])
        mock_cmd.assert_called_once()
