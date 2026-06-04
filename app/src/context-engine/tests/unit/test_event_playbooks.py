"""Tests for the event-playbook registry and prompt rendering."""

from __future__ import annotations

import pytest

from domain.event_playbooks import (
    EventPlaybook,
    all_registered_playbooks,
    find_playbook,
    render_playbooks_section,
)

pytestmark = pytest.mark.unit


def test_exact_match_returns_specific_playbook() -> None:
    pb = find_playbook("github", "repository", "added")
    assert pb.source_system == "github"
    assert pb.event_type == "repository"
    assert pb.action == "added"
    assert pb.max_tool_calls >= 60  # bootstrap is the heavy one


def test_unregistered_action_falls_through_to_default() -> None:
    """No (github, repository, deleted) entry → falls to ('*','*','*') default."""
    pb = find_playbook("github", "repository", "deleted")
    assert pb.source_system == "*"
    assert pb.event_type == "*"
    assert pb.action == "*"


def test_unknown_source_falls_through_to_default() -> None:
    pb = find_playbook("zoom", "meeting", "ended")
    assert pb.source_system == "*"


def test_render_includes_each_playbook_kind() -> None:
    """Section text names every (source/event_type/action) tuple it was given."""
    pbs = [
        find_playbook("github", "repository", "added"),
        find_playbook("github", "pull_request", "merged"),
    ]
    out = render_playbooks_section(pbs)
    assert "github / repository / added" in out
    assert "github / pull_request / merged" in out
    assert "WHAT TO EXTRACT" in out
    assert "USEFUL TOOLS" in out


def test_render_empty_list_returns_empty_string() -> None:
    assert render_playbooks_section([]) == ""


def test_all_registered_playbooks_includes_repo_bootstrap() -> None:
    keys = {(pb.source_system, pb.event_type, pb.action) for pb in all_registered_playbooks()}
    assert ("github", "repository", "added") in keys
    assert ("github", "pull_request", "merged") in keys


def test_playbook_is_frozen() -> None:
    pb = find_playbook("github", "repository", "added")
    with pytest.raises(Exception):
        pb.summary = "mutated"  # type: ignore[misc]
    assert isinstance(pb, EventPlaybook)
