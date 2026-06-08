"""Contract checks for Jira and Linear diff-sync ingestion playbooks."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


PLAYBOOKS_DIR = Path(__file__).resolve().parents[2] / "domain" / "playbooks"


def _read_skill(filename: str) -> tuple[dict[str, str], str]:
    raw = (PLAYBOOKS_DIR / filename).read_text(encoding="utf-8")
    assert raw.startswith("---\n")
    end = raw.find("\n---\n", 4)
    assert end > 0
    frontmatter: dict[str, str] = {}
    for line in raw[4:end].splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        frontmatter[key.strip()] = value.strip()
    return frontmatter, raw[end + 5 :]


@pytest.mark.parametrize(
    ("filename", "source_system", "event_type"),
    [
        ("linear_team_diff_sync.md", "linear", "linear_team"),
        ("jira_project_diff_sync.md", "jira", "jira_project"),
    ],
)
def test_diff_sync_frontmatter_targets_incremental_event(
    filename: str, source_system: str, event_type: str
) -> None:
    frontmatter, _ = _read_skill(filename)

    assert frontmatter["source_system"] == source_system
    assert frontmatter["event_type"] == event_type
    assert frontmatter["action"] == "diff_sync"
    assert frontmatter["enables_planner"].lower() == "true"


@pytest.mark.parametrize(
    ("filename", "history_path"),
    [
        (
            "linear_team_diff_sync.md",
            "context-sync-history/linear-team-<team-slug>.jsonl",
        ),
        (
            "jira_project_diff_sync.md",
            "context-sync-history/jira-project-<project-key-lowered>.jsonl",
        ),
    ],
)
def test_diff_sync_skills_require_append_only_history(
    filename: str, history_path: str
) -> None:
    _, body = _read_skill(filename)
    lowered = body.lower()

    assert history_path in body
    assert "append one json line per run" in lowered
    assert "do not overwrite or compact the history file" in lowered
    assert "only advance `new_cursor` after the context graph has been checked" in lowered
    assert "`graph_checked`" in body
    assert "`graph_missing`" in body
    assert "`graph_stale`" in body


def test_linear_diff_sync_uses_updated_since_enumeration() -> None:
    _, body = _read_skill("linear_team_diff_sync.md")

    assert "linear_list_projects(team_id=team, updated_since=query_since, limit=count)" in body
    assert "linear_list_documents(team_id=team, updated_since=query_since, limit=count)" in body
    assert "linear_list_issues(team_id=team, updated_since=query_since, limit=count)" in body
    assert "updatedAt" in body
    assert "activity:linear:issue:<identifier-lowered>" in body


def test_jira_diff_sync_uses_jql_updated_search() -> None:
    _, body = _read_skill("jira_project_diff_sync.md")

    assert "jira_search_issues(jql, limit=count)" in body
    assert "updated >= <query_since> ORDER BY updated ASC" in body
    assert "jira_bulk_fetch_changelogs" in body
    assert "activity:jira:issue:<issue-key-lowered>" in body


@pytest.mark.parametrize(
    "filename",
    ["linear_team_diff_sync.md", "jira_project_diff_sync.md"],
)
def test_diff_sync_audits_context_graph_before_hydrating_source(filename: str) -> None:
    _, body = _read_skill(filename)
    normalized = " ".join(body.split())

    assert "### Phase 2 - Audit current context graph" in body
    assert 'context_search(query=<activity-key>, node_labels=["Activity"], limit=1)' in body
    assert "before hydrating source details" in normalized
    assert "missing or stale" in body
    assert "source_updated_at" in body


@pytest.mark.parametrize(
    ("source_system", "event_type", "expected_heading"),
    [
        ("linear", "linear_team", "# Linear team diff sync"),
        ("jira", "jira_project", "# Jira project diff sync"),
    ],
)
def test_diff_sync_playbooks_are_registered(
    source_system: str, event_type: str, expected_heading: str
) -> None:
    from domain.event_playbooks import find_playbook, is_default_playbook

    pb = find_playbook(source_system, event_type, "diff_sync")

    assert not is_default_playbook(pb)
    assert pb.action == "diff_sync"
    assert pb.enables_planner is True
    assert pb.max_tool_calls >= 400
    assert expected_heading in pb.extract
    assert "context_search" in pb.tool_hints
