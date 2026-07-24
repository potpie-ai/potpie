"""Tests for root help: first-run guidance + de-emphasize auth/cloud."""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from potpie.cli import main as host_cli

pytestmark = pytest.mark.unit

# Rich help on CI injects ANSI between glyphs (e.g. "╭─" + codes + " Legacy"),
# so assertions must run on a stripped view of the text.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _root_help() -> str:
    result = CliRunner().invoke(
        host_cli.app,
        ["--help"],
        color=False,
        env={"NO_COLOR": "1", "TERM": "dumb"},
    )
    assert result.exit_code == 0, result.output
    return _strip_ansi(result.stdout)


def _panel_body(help_text: str, title: str) -> str:
    """Return the body of a Rich help panel titled ``title`` (until the next panel)."""
    pattern = rf"╭─\s*{re.escape(title)}\s*─+╮\n(.*?)(?=\n╭─|\Z)"
    match = re.search(pattern, help_text, flags=re.DOTALL)
    assert match is not None, f"panel {title!r} not found in:\n{help_text}"
    return match.group(1)


class TestAudit32RootHelp:
    def test_root_help_includes_first_run_block(self) -> None:
        help_text = _root_help()

        assert "First run:" in help_text
        assert "potpie setup --repo . --agent <harness>" in help_text
        assert "potpie doctor" in help_text
        assert "potpie status" in help_text

    def test_auth_and_cloud_are_outside_default_commands_panel(self) -> None:
        help_text = _root_help()
        commands = _panel_body(help_text, "Commands")

        assert re.search(r"^\s*│\s*auth\b", commands, flags=re.MULTILINE) is None
        assert re.search(r"^\s*│\s*cloud\b", commands, flags=re.MULTILINE) is None

    def test_auth_lives_in_legacy_panel(self) -> None:
        help_text = _root_help()
        legacy = _panel_body(help_text, "Legacy")

        assert re.search(r"^\s*│\s*auth\b", legacy, flags=re.MULTILINE)
        assert "auth status" in legacy.lower()
        assert "legacy" in legacy.lower() or "prefer" in legacy.lower()

    def test_cloud_lives_in_coming_soon_panel(self) -> None:
        help_text = _root_help()
        coming_soon = _panel_body(help_text, "Coming soon")

        assert re.search(r"^\s*│\s*cloud\b", coming_soon, flags=re.MULTILINE)
        assert "in development" in coming_soon.lower()
        assert "not implemented" not in coming_soon.lower()

    def test_legacy_and_coming_soon_panels_follow_commands(self) -> None:
        help_text = _root_help()

        commands_idx = help_text.index("╭─ Commands")
        legacy_idx = help_text.index("╭─ Legacy")
        coming_soon_idx = help_text.index("╭─ Coming soon")

        assert commands_idx < legacy_idx < coming_soon_idx
