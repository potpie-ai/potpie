"""Step 12b: the Claude Code plugin is well-formed and model-free.

Pins the plugin's manifest, hook wiring, and the no-model invariant. The adapter's
behavior is covered by ``test_nudge_adapter``; this covers the static contract:
events are mapped to the adapter, every hook command is a deterministic CLI call,
and the bundled skill does not drift from the other harness bundles.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import potpie.context_engine.adapters.inbound.cli as _clipkg

pytestmark = pytest.mark.unit

TEMPLATES = Path(_clipkg.__file__).resolve().parent / "templates"
PLUGIN = TEMPLATES / "claude_plugin"


def test_plugin_manifest_required_fields() -> None:
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    for key in ("name", "description", "version", "hooks"):
        assert key in manifest, f"plugin.json missing {key}"
    assert manifest["name"] == "potpie"
    assert manifest["hooks"] == "./hooks/hooks.json"


def test_marketplace_descriptor_points_at_plugin() -> None:
    mkt = json.loads(
        (PLUGIN / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8")
    )
    assert mkt["plugins"], "marketplace must list the plugin"
    plugin = mkt["plugins"][0]
    assert plugin["name"] == "potpie"
    assert plugin["source"] == "./"


def _hooks() -> dict:
    return json.loads((PLUGIN / "hooks" / "hooks.json").read_text(encoding="utf-8"))["hooks"]


def test_hooks_cover_the_four_v15_event_classes() -> None:
    hooks = _hooks()
    assert set(hooks) == {"SessionStart", "PreToolUse", "PostToolUse", "Stop"}
    # PreToolUse wires both an edit matcher and a Bash matcher.
    matchers = {entry.get("matcher") for entry in hooks["PreToolUse"]}
    assert any(m and "Write" in m and "Edit" in m for m in matchers)
    assert "Bash" in matchers
    # PostToolUse wires Bash (the red→green / failure path).
    assert {entry.get("matcher") for entry in hooks["PostToolUse"]} == {"Bash"}


def _all_hook_commands() -> list[str]:
    commands: list[str] = []
    for entries in _hooks().values():
        for entry in entries:
            for hook in entry.get("hooks", []):
                commands.append(hook["command"])
    return commands


def test_every_hook_calls_the_adapter_via_plugin_root() -> None:
    commands = _all_hook_commands()
    assert len(commands) == 5  # SessionStart, 2×PreToolUse, PostToolUse, Stop
    for cmd in commands:
        assert "potpie_nudge.py" in cmd
        assert "${CLAUDE_PLUGIN_ROOT}" in cmd
        assert "--event" in cmd


def test_no_hook_command_invokes_a_model() -> None:
    forbidden = ("anthropic", "openai", "claude -p", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    for cmd in _all_hook_commands():
        low = cmd.lower()
        for token in forbidden:
            assert token.lower() not in low, f"hook command calls a model: {cmd!r}"


def test_adapter_and_skill_files_exist() -> None:
    assert (PLUGIN / "hooks" / "potpie_nudge.py").is_file()
    assert (PLUGIN / "skills" / "potpie-graph" / "SKILL.md").is_file()
    for skill_id in (
        "potpie-project-preferences",
        "potpie-infra-architecture",
        "potpie-change-timeline",
        "potpie-debug-memory",
        "potpie-source-ingestion",
        "potpie-repo-baseline",
    ):
        assert (PLUGIN / "skills" / skill_id / "SKILL.md").is_file()
    assert (PLUGIN / "commands" / "potpie-feature.md").is_file()
    assert (PLUGIN / "commands" / "potpie-record.md").is_file()


def test_adapter_is_model_free() -> None:
    source = (PLUGIN / "hooks" / "potpie_nudge.py").read_text(encoding="utf-8").lower()
    for token in ("import anthropic", "import openai", "anthropic_api_key", "openai_api_key"):
        assert token not in source, f"adapter references a model client: {token}"


def test_potpie_graph_skill_does_not_drift_across_bundles() -> None:
    paths = [
        TEMPLATES / "agent_bundle" / ".agents" / "skills" / "potpie-graph" / "SKILL.md",
        TEMPLATES / "claude_bundle" / ".claude" / "skills" / "potpie-graph" / "SKILL.md",
        PLUGIN / "skills" / "potpie-graph" / "SKILL.md",
    ]
    bodies = {p.read_text(encoding="utf-8") for p in paths}
    assert len(bodies) == 1, "potpie-graph SKILL.md must be identical across all bundles"


def test_shared_plugin_and_agent_skills_do_not_drift() -> None:
    for skill_id in (
        "potpie-change-timeline",
        "potpie-debug-memory",
        "potpie-infra-architecture",
        "potpie-project-preferences",
        "potpie-repo-baseline",
        "potpie-source-ingestion",
    ):
        agent = (
            TEMPLATES
            / "agent_bundle"
            / ".agents"
            / "skills"
            / skill_id
            / "SKILL.md"
        )
        plugin = PLUGIN / "skills" / skill_id / "SKILL.md"
        assert agent.read_text(encoding="utf-8") == plugin.read_text(encoding="utf-8"), (
            f"{skill_id} SKILL.md must be identical in agent_bundle and claude_plugin"
        )
